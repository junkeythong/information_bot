from dataclasses import dataclass
from typing import List, Optional

import requests

from .models import EnvConfig


@dataclass
class FreqtradeHealthResult:
    port: int
    healthy: bool
    summary: str


_TOKEN_CACHE = {}


def clear_freqtrade_token_cache() -> None:
    _TOKEN_CACHE.clear()


def _base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/api/v1"


def _login_for_token(session: requests.Session, config: EnvConfig, port: int, timeout: int) -> Optional[str]:
    if not (config.freqtrade_api_username and config.freqtrade_api_password):
        return None

    response = session.post(
        f"{_base_url(port)}/token/login",
        auth=(config.freqtrade_api_username, config.freqtrade_api_password),
        timeout=timeout,
    )
    response.raise_for_status()
    token = response.json().get("access_token")
    if isinstance(token, str) and token.strip():
        _TOKEN_CACHE[port] = token.strip()
        return token.strip()
    return None


def _get_token(session: requests.Session, config: EnvConfig, port: int, timeout: int, *, force_login: bool = False) -> Optional[str]:
    if config.freqtrade_api_token and not force_login:
        return config.freqtrade_api_token
    if not (config.freqtrade_api_username and config.freqtrade_api_password):
        return None
    if not force_login and port in _TOKEN_CACHE:
        return _TOKEN_CACHE[port]
    return _login_for_token(session, config, port, timeout)


def _authenticated_get(
    session: requests.Session,
    config: EnvConfig,
    port: int,
    endpoint: str,
    *,
    timeout: int,
    params: Optional[dict] = None,
) -> requests.Response:
    token = _get_token(session, config, port, timeout)
    if not token:
        raise RuntimeError("auth not configured")

    url = f"{_base_url(port)}{endpoint}"
    response = session.get(url, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=timeout)
    if response.status_code in {401, 403} and not config.freqtrade_api_token:
        token = _get_token(session, config, port, timeout, force_login=True)
        if token:
            response = session.get(url, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=timeout)
    response.raise_for_status()
    return response


def parse_freqtrade_ports(raw: Optional[object]) -> List[int]:
    if raw is None:
        return []

    if isinstance(raw, (list, tuple)):
        parts = raw
    else:
        value = str(raw).strip()
        if not value or value.lower() in {"none", "null", "clear", "-"}:
            return []
        parts = value.split(",")

    ports = []
    for part in parts:
        token = str(part).strip()
        if not token:
            continue
        try:
            port = int(token)
        except ValueError as exc:
            raise ValueError("Freqtrade port values must be integers") from exc
        if not 1 <= port <= 65535:
            raise ValueError("Freqtrade port values must be between 1 and 65535")
        if port not in ports:
            ports.append(port)
    return ports


def _summary_from_health_payload(payload: object) -> str:
    if not isinstance(payload, dict):
        return "invalid health response"

    for key in ("status", "state", "bot_state", "trading_mode"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            summary = value.strip().lower()
            if summary in {"ok", "running", "healthy"}:
                return "healthy"
            return summary

    return "healthy"


def _is_healthy_summary(summary: str) -> bool:
    unhealthy_words = ("stop", "stopped", "stopping", "error", "fail", "unhealthy")
    normalized = summary.lower()
    return not any(word in normalized for word in unhealthy_words)


def _friendly_error_summary(exc: Exception) -> str:
    message = str(exc).lower()
    if "auth not configured" in message:
        return "auth not configured"
    if "401" in message or "403" in message or "unauthorized" in message or "forbidden" in message:
        return "auth failed"
    if "timed out" in message or "timeout" in message:
        return "timeout"
    if "connection" in message or "refused" in message:
        return "connection failed"
    return "health check failed"


def check_freqtrade_bots(
    session: requests.Session,
    config: EnvConfig,
    ports: List[int],
    *,
    timeout: int = 5,
) -> List[FreqtradeHealthResult]:
    results = []
    for port in ports:
        try:
            response = _authenticated_get(session, config, port, "/health", timeout=timeout)
            summary = _summary_from_health_payload(response.json())
            results.append(
                FreqtradeHealthResult(
                    port=port,
                    healthy=_is_healthy_summary(summary),
                    summary=summary,
                )
            )
        except Exception as exc:
            results.append(FreqtradeHealthResult(port=port, healthy=False, summary=_friendly_error_summary(exc)))
    return results


def format_freqtrade_status_section(results: List[FreqtradeHealthResult]) -> str:
    if not results:
        return ""

    lines = ["", "", "🤖 *Bots:*"]
    for result in results:
        icon = "✅" if result.healthy else "🔴"
        lines.append(f"• `{result.port}`: {icon} {result.summary}")
    return "\n".join(lines)


def format_freqtrade_alert(results: List[FreqtradeHealthResult]) -> str:
    unhealthy = [result for result in results if not result.healthy]
    if not unhealthy:
        return ""

    lines = ["🤖 *Freqtrade Health Alert:*"]
    for result in unhealthy:
        lines.append(f"• `{result.port}`: 🔴 {result.summary}")
    return "\n".join(lines)


def _normalize_freqtrade_pair(pair: object) -> str:
    value = str(pair or "").upper().strip()
    if ":" in value:
        value = value.split(":", 1)[0]
    return value.replace("/", "").replace("-", "").replace("_", "")


def fetch_freqtrade_exit_reasons(
    session: requests.Session,
    config: EnvConfig,
    ports: List[int],
    *,
    timeout: int = 5,
    limit: int = 500,
) -> dict:
    reasons = {}
    for port in ports:
        try:
            response = _authenticated_get(
                session,
                config,
                port,
                "/trades",
                timeout=timeout,
                params={"limit": limit, "offset": 0, "order_by_id": "false"},
            )
            payload = response.json()
            trades = payload.get("trades", payload if isinstance(payload, list) else [])
            for trade in trades:
                if not isinstance(trade, dict) or trade.get("is_open") is True:
                    continue
                symbol = _normalize_freqtrade_pair(trade.get("pair"))
                reason = trade.get("exit_reason") or trade.get("sell_reason")
                if symbol and reason and symbol not in reasons:
                    reasons[symbol] = str(reason)
        except Exception:
            continue
    return reasons


def apply_exit_reasons_to_closed_trades(pnl: object, exit_reasons: dict) -> object:
    if not isinstance(pnl, dict) or not exit_reasons:
        return pnl

    closed_trades = pnl.get("closed_trades")
    if not isinstance(closed_trades, list):
        return pnl

    updated = dict(pnl)
    updated_trades = []
    for trade in closed_trades:
        if not isinstance(trade, dict):
            updated_trades.append(trade)
            continue
        updated_trade = dict(trade)
        symbol = _normalize_freqtrade_pair(updated_trade.get("symbol"))
        reason = exit_reasons.get(symbol)
        if reason:
            updated_trade["exit_reason"] = reason
        updated_trades.append(updated_trade)
    updated["closed_trades"] = updated_trades
    return updated
