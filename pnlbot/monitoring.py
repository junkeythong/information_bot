import threading
import time
from typing import List

import requests

from .constants import POWER_OUTAGE_REFRESH_SECONDS, STATE_FILE_PATH
from .market_data import get_air_quality, get_futures_pnl, get_spot_balance
from .messages import get_pnl_icon
from .models import BotSettings, BotState, EnvConfig
from .outages import get_power_outages
from .persistence import persist_runtime_state
from .state import update_pnl_range, update_spot_balance_range
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


def monitor_loop(session: requests.Session, config: EnvConfig, settings: BotSettings, state: BotState) -> None:
    pnl = get_futures_pnl(session, config)
    spot_balance = get_spot_balance(session, config)

    if isinstance(pnl, str):
        send_telegram_message(session, config, settings, pnl, state=state, force_send=True)
        return

    state_changed = False

    # Update Futures Stats
    if isinstance(pnl, (int, float)):
        if update_pnl_range(state, pnl):
            state_changed = True

    # Update Spot Stats
    total_spot = 0.0
    if isinstance(spot_balance, dict):
        total_spot = spot_balance.get("total", 0.0)
    elif isinstance(spot_balance, (int, float)):
        total_spot = float(spot_balance)

    if total_spot > 0:
        if update_spot_balance_range(state, total_spot):
            state_changed = True

    # Format spot balance message
    spot_msg = ""
    if total_spot > 0:
        pnl_perc_info = ""
        if state.init_capital:
            pnl_perc = (total_spot - state.init_capital) / state.init_capital * 100
            icon = get_pnl_icon(pnl_perc)
            pnl_perc_info = f" {icon} ({pnl_perc:+.2f}%)"

        spot_msg = f"💰 *Spot:* `{total_spot:,.2f} USDT`{pnl_perc_info}\n📊 *Range:* `[{state.min_spot_balance:,.2f}, {state.max_spot_balance:,.2f}]`\n"
        if isinstance(spot_balance, dict):
            breakdown = spot_balance.get("breakdown", [])
            for item in breakdown[:5]:  # Show top 5 assets
                price_str = f" @ {item['price']:,.4f}" if item['asset'] != "USDT" else ""
                spot_msg += f"  ▫️ `{item['asset']}`: `{item['usdt_value']:,.2f} USDT`{price_str}\n"
        spot_msg += "\n"
    elif isinstance(spot_balance, str) and not ("0.0" in spot_balance or "0 USDT" in spot_balance):
        spot_msg = f"⚠️ {spot_balance}\n\n"

    # Format futures message
    futures_msg = ""
    if pnl != 0.0:
        if pnl <= state.pnl_alert_low:
            futures_msg = f"Heavy loss: 🔻 `{pnl:,.2f} USDT`\n📊 *Range:* `[{state.min_pnl:,.2f}, {state.max_pnl:,.2f}]`"
        elif pnl >= state.pnl_alert_high:
            futures_msg = f"High profit: 🟢 `{pnl:,.2f} USDT`\n📊 *Range:* `[{state.min_pnl:,.2f}, {state.max_pnl:,.2f}]`"
        else:
            icon = get_pnl_icon(pnl)
            futures_msg = f"💰 *Futures:* `{pnl:,.2f} USDT` {icon}\n📊 *Range:* `[{state.min_pnl:,.2f}, {state.max_pnl:,.2f}]`"

    # Format AQI message
    aqi_msg = ""
    if config.iqair_api_key:
        aqi_data = get_air_quality(session, config)
        if isinstance(aqi_data, dict):
            aqi_us = aqi_data.get("aqi_us", 0)
            # Determine AQI emoji
            if aqi_us <= 50:
                aqi_emoji = "🟢"
            elif aqi_us <= 100:
                aqi_emoji = "🟡"
            elif aqi_us <= 150:
                aqi_emoji = "🟠"
            elif aqi_us <= 200:
                aqi_emoji = "🔴"
            elif aqi_us <= 300:
                aqi_emoji = "🟣"
            else:
                aqi_emoji = "🟤"

            aqi_msg = f"\n{aqi_emoji} *AQI:* `{aqi_us}` ({aqi_data.get('city', 'N/A')}), Temp: `{aqi_data.get('temperature', 0)}°C`"

    if spot_msg or futures_msg or aqi_msg:
        send_telegram_message(
            session,
            config,
            settings,
            f"{spot_msg}{futures_msg}{aqi_msg}".strip(),
            state=state,
        )

    # Power Outage Check
    if time.time() - state.last_outage_check >= POWER_OUTAGE_REFRESH_SECONDS:
        new_outages = refresh_power_outages(session, config, state)
        if new_outages:
            lines = ["⚡ *NEW Power Outage detected!*"]
            for o in new_outages:
                lines.append(f"• `{o['time']}`")
                lines.append(f"  ▫️ Area: `{o['area']}`")
                lines.append(f"  ▫️ Reason: `{o['reason']}`")
            send_telegram_message(session, config, settings, "\n".join(lines), state=state, force_send=True)
        state_changed = True

    if state_changed:
        persist_runtime_state(STATE_FILE_PATH, state, settings)
