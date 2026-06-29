import datetime
from typing import List, Optional

import pytz
import requests

from .constants import TELEGRAM_API_URL, TELEGRAM_MAX_MESSAGE
from .models import BotSettings, BotState, EnvConfig


def split_telegram_message(message: str, max_length: int = TELEGRAM_MAX_MESSAGE) -> List[str]:
    if len(message) <= max_length:
        return [message]

    chunks = []
    current = ""
    for line in message.splitlines(keepends=True):
        if len(line) > max_length:
            if current:
                chunks.append(current)
                current = ""
            for idx in range(0, len(line), max_length):
                chunks.append(line[idx : idx + max_length])
            continue

        if current and len(current) + len(line) > max_length:
            chunks.append(current)
            current = line
        else:
            current += line

    if current:
        chunks.append(current)

    return chunks


def telegram_message_payload(base_payload: dict, text: str) -> dict:
    payload = {**base_payload, "text": text}
    if text.count("`") % 2 == 1:
        payload.pop("parse_mode", None)
    return payload


def send_telegram_message(
    session: requests.Session,
    config: EnvConfig,
    settings: BotSettings,
    message: str,
    *,
    chat_id: Optional[str] = None,
    state: Optional[BotState] = None,
    force_send: bool = False,
) -> Optional[dict]:
    tz = pytz.timezone(config.timezone)

    now_hour = datetime.datetime.now(tz).hour
    if state and state.night_mode_enabled and not force_send:
        start_hour, end_hour = state.night_mode_window
        if start_hour <= end_hour:
            if start_hour <= now_hour < end_hour:
                return None
        else:
            if now_hour >= start_hour or now_hour < end_hour:
                return None

    url = f"{TELEGRAM_API_URL}/bot{config.telegram_token}/sendMessage"
    payload = {
        "chat_id": chat_id or config.telegram_chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }
    try:
        if len(message) > TELEGRAM_MAX_MESSAGE:
            first_res = None
            for chunk in split_telegram_message(message, TELEGRAM_MAX_MESSAGE):
                res = session.post(
                    url,
                    data=telegram_message_payload(payload, chunk),
                    timeout=10,
                )
                res.raise_for_status()
                if first_res is None:
                    first_res = res.json()
            return first_res
        else:
            res = session.post(url, data=telegram_message_payload(payload, message), timeout=10)
            res.raise_for_status()
            return res.json()
    except Exception as exc:
        print(f"Telegram send error: {sanitize_telegram_error(exc, config)}", flush=True)
        return None


def is_telegram_polling_conflict(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 409


def sanitize_telegram_error(exc: Exception, config: EnvConfig) -> str:
    return str(exc).replace(f"/bot{config.telegram_token}/", "/bot<redacted>/")


def pin_telegram_message(session: requests.Session, config: EnvConfig, message_id: int) -> None:
    url = f"{TELEGRAM_API_URL}/bot{config.telegram_token}/pinChatMessage"
    payload = {
        "chat_id": config.telegram_chat_id,
        "message_id": message_id,
        "disable_notification": True,
    }
    try:
        res = session.post(url, data=payload, timeout=10)
        res.raise_for_status()
    except Exception as exc:
        print(f"Telegram pin error: {sanitize_telegram_error(exc, config)}", flush=True)


def unpin_telegram_message(session: requests.Session, config: EnvConfig, message_id: Optional[int] = None) -> None:
    url = f"{TELEGRAM_API_URL}/bot{config.telegram_token}/unpinChatMessage"
    payload = {"chat_id": config.telegram_chat_id}
    if message_id:
        payload["message_id"] = message_id
    try:
        res = session.post(url, data=payload, timeout=10)
        res.raise_for_status()
    except Exception as exc:
        print(f"Telegram unpin error: {sanitize_telegram_error(exc, config)}", flush=True)
