from typing import Callable, Dict, Optional

import requests

from .command_handlers import (
    handle_aqi_command,
    handle_config_command_message,
    handle_futures_command,
    handle_help_command,
    handle_outage_command,
    handle_spot_command,
    handle_start_command,
    handle_status_command,
    handle_stop_command,
    handle_sysinfo_command,
    handle_unknown_command,
)
from .constants import TELEGRAM_API_URL
from .models import BotSettings, BotState, EnvConfig
from .telegram import is_telegram_polling_conflict, sanitize_telegram_error

CommandHandler = Callable[
    [requests.Session, EnvConfig, BotSettings, BotState, str, str],
    None,
]

EXACT_COMMAND_HANDLERS: Dict[str, CommandHandler] = {
    "/status": handle_status_command,
    "/stop": handle_stop_command,
    "/start": handle_start_command,
    "/outage": handle_outage_command,
    "/sysinfo": handle_sysinfo_command,
    "/aqi": handle_aqi_command,
    "/help": handle_help_command,
}

PREFIX_COMMAND_HANDLERS: Dict[str, CommandHandler] = {
    "/config": handle_config_command_message,
    "/futures": handle_futures_command,
    "/spot": handle_spot_command,
}


def _normalize_command_text(original_text: str) -> Optional[str]:
    original_text = original_text.strip()
    if not original_text.startswith("/"):
        return None

    parts = original_text.split()
    if not parts:
        return None

    cmd_part = parts[0].lower()
    if "@" in cmd_part:
        cmd_part = cmd_part.split("@")[0]

    if len(parts) == 1:
        return cmd_part
    return f"{cmd_part} {' '.join(parts[1:])}"


def _handler_for(text: str) -> CommandHandler:
    base_command = text.split()[0]
    if base_command in PREFIX_COMMAND_HANDLERS:
        return PREFIX_COMMAND_HANDLERS[base_command]
    return EXACT_COMMAND_HANDLERS.get(text, handle_unknown_command)


def _dispatch_command(
    session: requests.Session,
    config: EnvConfig,
    settings: BotSettings,
    state: BotState,
    chat_id: str,
    text: str,
) -> None:
    handler = _handler_for(text)
    handler(session, config, settings, state, chat_id, text)


def check_telegram_commands(
    session: requests.Session,
    config: EnvConfig,
    settings: BotSettings,
    state: BotState,
    last_update_id: Optional[int],
    poll_timeout: int,
) -> Optional[int]:
    url = f"{TELEGRAM_API_URL}/bot{config.telegram_token}/getUpdates"
    params = {"timeout": poll_timeout}
    if last_update_id is not None:
        params["offset"] = last_update_id + 1

    try:
        response = session.get(url, params=params, timeout=poll_timeout + 5)
        response.raise_for_status()
        updates = response.json()
    except Exception as exc:
        if is_telegram_polling_conflict(exc):
            state.telegram_command_polling_enabled = False
            print(
                "Telegram command polling disabled: another getUpdates consumer is already active "
                f"({sanitize_telegram_error(exc, config)})",
                flush=True,
            )
        else:
            print(f"Telegram command error: {sanitize_telegram_error(exc, config)}", flush=True)
        return last_update_id

    results = updates.get("result") or []
    if not results:
        return last_update_id

    latest_id = last_update_id
    for update in results:
        uid = update.get("update_id")
        if uid is None:
            continue

        if latest_id is None or uid > latest_id:
            latest_id = uid

        message = update.get("message", {})
        text = _normalize_command_text(message.get("text") or "")
        chat_id = str(message.get("chat", {}).get("id", ""))

        if text is None or chat_id != str(config.telegram_chat_id):
            continue

        _dispatch_command(session, config, settings, state, chat_id, text)

    return latest_id
