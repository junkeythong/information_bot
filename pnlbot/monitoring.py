import threading
import time
from typing import List

import requests

from . import portfolio
from .constants import POWER_OUTAGE_REFRESH_SECONDS, STATE_FILE_PATH
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
    snapshot = portfolio.refresh_portfolio_snapshot(session, config, state)

    if isinstance(snapshot.pnl, str):
        send_telegram_message(session, config, settings, snapshot.pnl, state=state, force_send=True)
        return

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
    if time.time() - state.last_outage_check >= POWER_OUTAGE_REFRESH_SECONDS:
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
