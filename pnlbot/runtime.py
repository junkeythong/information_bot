import atexit
import datetime
import time
from typing import Optional

import pytz
import requests

from pnlbot.commands import check_telegram_commands
from pnlbot.config import load_bot_settings, load_env_config, log_lunar_vn_version
from pnlbot.constants import (
    ASCII_STARTUP_BANNER,
    STATE_FILE_PATH,
    TELEGRAM_API_URL,
    TELEGRAM_POLL_TIMEOUT,
)
from pnlbot.freqtrade import (
    check_freqtrade_bots,
    enrich_pnl_with_freqtrade_exit_reasons,
    format_freqtrade_status_section,
)
from pnlbot.http import create_retry_session
from pnlbot.logging import configure_runtime_logging
from pnlbot.messages import compose_status_message
from pnlbot.models import BotSettings, BotState, EnvConfig
from pnlbot.monitoring import (
    monitor_loop,
    notify_exit,
    start_system_monitor_worker,
)
from pnlbot import portfolio
from pnlbot.persistence import (
    apply_persisted_configuration,
    load_persisted_state,
    persist_runtime_state,
)
from pnlbot.telegram import (
    is_telegram_polling_conflict,
    pin_telegram_message,
    sanitize_telegram_error,
    send_telegram_message,
    unpin_telegram_message,
)
from pnlbot.time_utils import get_lunar_date_string, should_send_daily_spot_report, should_send_daily_status


def compose_startup_status_message(
    session: requests.Session,
    config: EnvConfig,
    state: BotState,
    pnl: object,
    *,
    spot_balance: object,
) -> str:
    pnl = enrich_pnl_with_freqtrade_exit_reasons(session, config, state.freqtrade_ports, pnl)

    message = compose_status_message(state, config, None, pnl, spot_balance=spot_balance)
    if state.freqtrade_ports:
        freqtrade_results = check_freqtrade_bots(session, config, state.freqtrade_ports)
        message += format_freqtrade_status_section(freqtrade_results)
    return message


def init_last_update_id(
    session: requests.Session,
    config: EnvConfig,
    settings: BotSettings,
    state: Optional[BotState] = None,
) -> Optional[int]:
    try:
        url = f"{TELEGRAM_API_URL}/bot{config.telegram_token}/getUpdates"
        response = session.get(url, params={"timeout": 1}, timeout=6)
        response.raise_for_status()
        updates = response.json().get("result") or []
        if updates:
            return max(update["update_id"] for update in updates if "update_id" in update)
    except Exception as exc:
        if is_telegram_polling_conflict(exc) and state is not None:
            state.telegram_command_polling_enabled = False
            print(
                "Startup disabled Telegram command polling: another getUpdates consumer is already active "
                f"({sanitize_telegram_error(exc, config)})",
                flush=True,
            )
        else:
            print(f"Startup could not fetch update_id: {sanitize_telegram_error(exc, config)}", flush=True)
    return None


def send_daily_lunar_pin(
    session: requests.Session,
    config: EnvConfig,
    settings: BotSettings,
    state: BotState,
    now: datetime.datetime,
) -> bool:
    if not should_send_daily_status(state, now):
        return False

    lunar_str = get_lunar_date_string(config.timezone)
    alert_msg = f"📅 *Hôm nay là:* `{lunar_str}`"

    if state.pinned_daily_message_id:
        unpin_telegram_message(session, config, state.pinned_daily_message_id)

    resp = send_telegram_message(
        session,
        config,
        settings,
        alert_msg,
        state=state,
        force_send=True,
    )

    if resp and "result" in resp:
        msg_id = resp["result"].get("message_id")
        if msg_id:
            pin_telegram_message(session, config, msg_id)
            state.pinned_daily_message_id = msg_id

    state.last_lunar_alert_date = now.strftime("%Y-%m-%d")
    persist_runtime_state(STATE_FILE_PATH, state, settings)
    return True


def send_daily_spot_report(
    session: requests.Session,
    config: EnvConfig,
    settings: BotSettings,
    state: BotState,
    now: datetime.datetime,
) -> bool:
    if not should_send_daily_spot_report(state, now):
        return False

    spot_balance, _ = portfolio.refresh_spot_balance(session, config, state)
    send_telegram_message(
        session,
        config,
        settings,
        portfolio.format_spot_balance_summary(
            state,
            spot_balance,
        ),
        state=state,
        force_send=True,
    )
    state.last_spot_report_date = now.strftime("%Y-%m-%d")
    persist_runtime_state(STATE_FILE_PATH, state, settings)
    return True


def main() -> None:
    configure_runtime_logging()
    lunar_vn_version = log_lunar_vn_version()
    try:
        config = load_env_config()
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return

    try:
        settings = load_bot_settings()
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return

    session = create_retry_session()
    state = BotState(
        interval_seconds=settings.default_interval_seconds,
        night_mode_enabled=settings.default_night_mode_enabled,
        night_mode_window=settings.night_mode_window,
        init_capital=settings.init_capital,
        outage_street_filter=config.outage_street_filter,
    )

    persisted = load_persisted_state(STATE_FILE_PATH)
    if persisted:
        apply_persisted_configuration(persisted, state, settings)
        config.outage_street_filter = state.outage_street_filter

    atexit.register(lambda: notify_exit(session, config, settings))

    try:
        state.last_update_id = init_last_update_id(session, config, settings, state)
        snapshot = portfolio.refresh_portfolio_snapshot(session, config, state)
        pnl = snapshot.pnl
        if isinstance(pnl, str):
            send_telegram_message(
                session,
                config,
                settings,
                f"❌ Failed to retrieve PnL: {pnl}",
                state=state,
                force_send=True,
            )
            pnl = 0.0

        persist_runtime_state(STATE_FILE_PATH, state, settings)

        send_telegram_message(
            session,
            config,
            settings,
            ASCII_STARTUP_BANNER,
            state=state,
            force_send=True,
        )

        send_telegram_message(
            session,
            config,
            settings,
            "🤖 Binance PnL bot is starting...",
            state=state,
            force_send=True,
        )
        send_telegram_message(
            session,
            config,
            settings,
            f"📦 lunar-vn version: `{lunar_vn_version}`",
            state=state,
            force_send=True,
        )

        spot_balance = snapshot.spot_balance

        # Status retrieval is done on demand
        send_telegram_message(
            session,
            config,
            settings,
            compose_startup_status_message(
                session,
                config,
                state,
                pnl,
                spot_balance=spot_balance,
            ),
            state=state,
            force_send=True,
        )

        start_system_monitor_worker(session, config, settings, state)

        last_run = time.time()
        while True:
            poll_timeout = min(TELEGRAM_POLL_TIMEOUT, max(1, state.interval_seconds // 2))
            if state.telegram_command_polling_enabled:
                state.last_update_id = check_telegram_commands(
                    session,
                    config,
                    settings,
                    state,
                    state.last_update_id,
                    poll_timeout,
                )
            else:
                time.sleep(poll_timeout)

            tz_now = datetime.datetime.now(pytz.timezone(config.timezone))
            start_hour, end_hour = state.night_mode_window
            if start_hour <= end_hour:
                in_night_window = state.night_mode_enabled and start_hour <= tz_now.hour < end_hour
            else:
                in_night_window = state.night_mode_enabled and (
                    tz_now.hour >= start_hour or tz_now.hour < end_hour
                )

            if in_night_window and not state.night_mode_active:
                state.night_mode_active = True
                send_telegram_message(
                    session,
                    config,
                    settings,
                    "🌙 *Night mode started.* Notifications are paused.",
                    state=state,
                    force_send=True,
                )
            elif not in_night_window and state.night_mode_active:
                state.night_mode_active = False
                send_telegram_message(
                    session,
                    config,
                    settings,
                    "🌅 *Night mode ended.* Resuming regular notifications.",
                    state=state,
                    force_send=True,
                )

            # 8:00 AM daily lunar pin and Spot report
            send_daily_lunar_pin(session, config, settings, state, tz_now)
            send_daily_spot_report(session, config, settings, state, tz_now)

            now = time.time()
            if state.is_running and now - last_run >= state.interval_seconds:
                monitor_loop(session, config, settings, state)
                persist_runtime_state(STATE_FILE_PATH, state, settings)
                last_run = now

    except Exception as exc:
        send_telegram_message(
            session,
            config,
            settings,
            f"❌ The bot encountered an error and stopped: `{exc}`",
            state=state,
            force_send=True,
        )
        print(f"Unhandled error: {exc}")
