import threading
import time
from typing import List

import requests

from . import portfolio
from .constants import POWER_OUTAGE_REFRESH_SECONDS, STATE_FILE_PATH
from .freqtrade import (
    check_freqtrade_bots,
    enrich_pnl_with_freqtrade_exit_reasons,
    format_freqtrade_alert,
)
from .messages import format_closed_trade_lines
from .models import BotSettings, BotState, EnvConfig
from .outages import get_power_outages
from .persistence import persist_runtime_state
from .system_info import get_system_info_text
from .telegram import send_telegram_message


def start_system_monitor_worker(
    session: requests.Session, config: EnvConfig, settings: BotSettings, state: BotState
) -> threading.Thread:
    def worker():
        last_alert_time = 0
        while True:
            try:
                time.sleep(5)  # Near real-time check every 5 seconds
                if not state.is_running:
                    continue

                system_info = get_system_info_text(config, settings)
                if system_info:
                    now = time.time()
                    # 5 minutes cooldown to avoid alert spam
                    if now - last_alert_time > 300:
                        send_telegram_message(session, config, settings, system_info, state=state, force_send=True)
                        last_alert_time = now

                if maybe_send_freqtrade_health_alert(session, config, settings, state):
                    persist_runtime_state(STATE_FILE_PATH, state, settings)
            except Exception as exc:
                print(f"System monitor worker error: {exc}")
                time.sleep(10)

    thread = threading.Thread(target=worker, name="system-monitor-worker", daemon=True)
    thread.start()
    return thread


def notify_exit(session: requests.Session, config: EnvConfig, settings: BotSettings) -> None:
    try:
        send_telegram_message(session, config, settings, "❌ Bot has been stopped.", force_send=True)
    except Exception as exc:
        print(f"Error while sending shutdown notification: {exc}")


def maybe_send_freqtrade_health_alert(
    session: requests.Session,
    config: EnvConfig,
    settings: BotSettings,
    state: BotState,
    *,
    now: float = None,
) -> bool:
    if not state.freqtrade_ports:
        return False

    current_time = time.time() if now is None else now
    freqtrade_results = check_freqtrade_bots(session, config, state.freqtrade_ports)
    freqtrade_alert = format_freqtrade_alert(freqtrade_results)
    if not freqtrade_alert:
        return False

    if current_time - state.last_freqtrade_alert_time < state.freqtrade_alert_cooldown_seconds:
        return False

    send_telegram_message(session, config, settings, freqtrade_alert, state=state, force_send=True)
    state.last_freqtrade_alert_time = current_time
    return True


def refresh_power_outages(session: requests.Session, config: EnvConfig, state: BotState) -> List[dict]:
    """Updates the state with new power outages and returns only the NEW ones."""
    current_outages = get_power_outages(session, config)
    state.last_outage_check = time.time()
    if not current_outages:
        return []

    seen_ids = {o["id"] for o in state.power_outages}
    new_outages = [o for o in current_outages if o["id"] not in seen_ids]

    # Keep only current and future outages in state
    # Since we don't have perfect date parsing here, we'll just keep the latest results
    # and filter by ID to find truly new ones.
    state.power_outages = current_outages

    return new_outages


def _closed_trade_key(trade: dict) -> str:
    return ":".join(
        [
            str(trade.get("symbol", "UNKNOWN")),
            str(trade.get("position_side", "BOTH")),
            str(trade.get("side", "UNKNOWN")),
            str(trade.get("time", 0)),
            f"{float(trade.get('pnl', 0.0)):.8f}",
        ]
    )


def find_new_closed_futures_trades(state: BotState, pnl: object) -> tuple[List[dict], bool]:
    if not isinstance(pnl, dict):
        return [], False

    closed_trades = [trade for trade in pnl.get("closed_trades", []) if isinstance(trade, dict)]
    if not closed_trades:
        return [], False

    current_keys = [_closed_trade_key(trade) for trade in closed_trades]
    if not state.seen_futures_closed_trade_keys:
        state.seen_futures_closed_trade_keys = current_keys[:50]
        return [], True

    seen_keys = set(state.seen_futures_closed_trade_keys)
    new_trades = [
        trade for key, trade in zip(current_keys, closed_trades)
        if key not in seen_keys
    ]

    merged_keys = []
    for key in current_keys + state.seen_futures_closed_trade_keys:
        if key not in merged_keys:
            merged_keys.append(key)
        if len(merged_keys) >= 50:
            break

    state_changed = merged_keys != state.seen_futures_closed_trade_keys
    state.seen_futures_closed_trade_keys = merged_keys
    return new_trades, state_changed


def format_closed_futures_trade_alert(closed_trades: List[dict]) -> str:
    title = "✅ *Futures trade closed*" if len(closed_trades) == 1 else "✅ *Futures trades closed*"
    lines = [title]
    for trade in closed_trades:
        lines.extend(format_closed_trade_lines(trade))
    return "\n".join(lines)


def monitor_loop(session: requests.Session, config: EnvConfig, settings: BotSettings, state: BotState) -> None:
    now = time.time()
    pnl, pnl_changed = portfolio.refresh_futures_pnl(session, config, state)
    _, spot_changed = portfolio.refresh_spot_balance(session, config, state)
    snapshot = portfolio.PortfolioSnapshot(
        pnl=pnl,
        spot_balance=None,
        state_changed=pnl_changed or spot_changed,
    )

    if isinstance(snapshot.pnl, str):
        send_telegram_message(session, config, settings, snapshot.pnl, state=state, force_send=True)
        return

    snapshot.pnl = enrich_pnl_with_freqtrade_exit_reasons(session, config, state.freqtrade_ports, snapshot.pnl)
    new_closed_trades, closed_trade_state_changed = find_new_closed_futures_trades(state, snapshot.pnl)
    snapshot.state_changed = snapshot.state_changed or closed_trade_state_changed

    if new_closed_trades:
        send_telegram_message(
            session,
            config,
            settings,
            format_closed_futures_trade_alert(new_closed_trades),
            state=state,
        )

    message = portfolio.format_monitoring_message(session, config, state, snapshot)
    if message:
        send_telegram_message(
            session,
            config,
            settings,
            message,
            state=state,
        )

    # Power Outage Check
    if now - state.last_outage_check >= POWER_OUTAGE_REFRESH_SECONDS:
        new_outages = refresh_power_outages(session, config, state)
        if new_outages:
            lines = ["⚡ *NEW Power Outage detected!*"]
            for o in new_outages:
                lines.append(f"• `{o['time']}`")
                lines.append(f"  ▫️ Area: `{o['area']}`")
                lines.append(f"  ▫️ Reason: `{o['reason']}`")
            send_telegram_message(session, config, settings, "\n".join(lines), state=state, force_send=True)
        snapshot.state_changed = True

    if snapshot.state_changed:
        persist_runtime_state(STATE_FILE_PATH, state, settings)
