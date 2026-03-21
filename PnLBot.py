import atexit
import datetime
import hashlib
import hmac
import json
import os
import threading
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple, Union
import re
import html
import sys
import unicodedata

import psutil
import pytz
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from lunar_vn import solar_to_lunar


OPENAI_COSTS_URL = "https://api.openai.com/v1/organization/costs"
OPENAI_USAGE_URL = "https://api.openai.com/v1/organization/usage/completions"
IQAIR_API_URL = "https://api.airvisual.com/v2/nearest_city"
ASCII_STARTUP_BANNER = (
    "```\n"
    "ʕ•ᴥ•ʔ\n"
    " short it!\n"
    "```"
)
STATE_FILE_PATH = "pnl-bot-state.json"
TODO_FILE_PATH = "pnl-bot-todo-db.txt"
TELEGRAM_API_URL = "https://api.telegram.org"
TELEGRAM_MAX_MESSAGE = 4096
OPENAI_REFRESH_SECONDS = 300
EVN_SPC_OUTAGE_URL = "https://www.cskh.evnspc.vn/TraCuu/GetThongTinLichNgungGiamCungCapDien"
# Cache outages for some time to avoid frequent calls
POWER_OUTAGE_REFRESH_SECONDS = 3600
TELEGRAM_POLL_TIMEOUT = 30


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer") from exc


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be a float") from exc


@dataclass
class BotSettings:
    default_interval_seconds: int
    default_pnl_alert_low: int
    default_pnl_alert_high: int
    default_night_mode_enabled: bool
    night_mode_window: Tuple[int, int]
    init_capital: Optional[float] = None


def load_bot_settings() -> BotSettings:
    default_interval = env_int("PNL_BOT_DEFAULT_INTERVAL_SECONDS", 3600)
    default_low = env_int("PNL_BOT_DEFAULT_PNL_ALERT_LOW", -20)
    default_high = env_int("PNL_BOT_DEFAULT_PNL_ALERT_HIGH", 20)
    night_mode_start = env_int("PNL_BOT_NIGHT_MODE_START_HOUR", 0)
    night_mode_end = env_int("PNL_BOT_NIGHT_MODE_END_HOUR", 5)
    default_night_mode = env_bool("PNL_BOT_DEFAULT_NIGHT_MODE_ENABLED", True)
    init_capital = env_float("PNL_BOT_INIT_CAPITAL", 0.0)
    if not (0 <= night_mode_start <= 23 and 0 <= night_mode_end <= 24):
        raise RuntimeError("Night mode hours must be within 0-24 range")
    if night_mode_start == night_mode_end:
        raise RuntimeError("Night mode start and end hours must differ")
    night_mode_window = (night_mode_start, night_mode_end)

    return BotSettings(
        default_interval_seconds=default_interval,
        default_pnl_alert_low=default_low,
        default_pnl_alert_high=default_high,
        default_night_mode_enabled=default_night_mode,
        night_mode_window=night_mode_window,
        init_capital=(init_capital if init_capital > 0 else None),
    )


@dataclass
class EnvConfig:
    api_key: str
    api_secret: str
    telegram_token: str
    telegram_chat_id: str
    openai_admin_key: Optional[str] = None
    iqair_api_key: Optional[str] = None
    iqair_latitude: float = 10.8231
    iqair_longitude: float = 106.6297
    outage_street_filter: Optional[str] = None
    evn_madvi: str = "PB0100"
    evn_area_name: str = "Ho Chi Minh"
    timezone: str = "Asia/Ho_Chi_Minh"
    cpu_alert_threshold: int = 80
    mem_alert_threshold: int = 80
    disk_alert_threshold: int = 80


@dataclass
class BotState:
    interval_seconds: int
    night_mode_enabled: bool
    pnl_alert_low: int
    pnl_alert_high: int
    night_mode_window: Tuple[int, int]
    is_running: bool = True
    last_update_id: Optional[int] = None
    max_pnl: float = 0.0
    min_pnl: float = 0.0
    max_spot_balance: float = 0.0
    min_spot_balance: float = 0.0
    init_capital: Optional[float] = None
    night_mode_active: bool = False
    start_time: float = field(default_factory=time.time)
    openai_usage: Optional[dict] = None
    openai_usage_error: Optional[str] = None
    openai_usage_lock: Lock = field(default_factory=Lock, repr=False)
    power_outages: List[dict] = field(default_factory=list)
    last_outage_check: float = 0.0


@dataclass
class ConfigDefinition:
    description: str
    parser: Callable[[str, BotState, BotSettings], object]
    getter: Callable[[BotState, BotSettings], object]
    applier: Callable[[object, BotState, BotSettings], Optional[str]]


def parse_bool_value(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("Value must be true/false or on/off")


def parse_float_value(raw: str, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Value must be a number") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"Value must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"Value must be <= {maximum}")
    return value


def parse_int_value(raw: str, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Value must be an integer") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"Value must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"Value must be <= {maximum}")
    return value


def format_config_value(value: object) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
        return f"{value[0]} -> {value[1]}"
    return str(value)

def create_retry_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def epoch_utc(dt_obj: datetime.datetime) -> int:
    return int(dt_obj.astimezone(datetime.timezone.utc).timestamp())


def sum_openai_costs(session: requests.Session, key: str, start_epoch: int, end_epoch: int) -> float:
    headers = {"Authorization": f"Bearer {key}"}
    params = {"start_time": start_epoch, "end_time": end_epoch, "limit": 31}
    response = session.get(OPENAI_COSTS_URL, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    payload = response.json()
    total = 0.0
    for bucket in payload.get("data", []):
        rows = bucket.get("results") or bucket.get("result") or []
        for row in rows:
            amount = (row.get("amount") or {}).get("value", 0)
            try:
                total += float(amount or 0)
            except (TypeError, ValueError):
                continue
    return total


def sum_openai_usage(session: requests.Session, key: str, start_epoch: int, end_epoch: int) -> Tuple[int, int]:
    headers = {"Authorization": f"Bearer {key}"}
    params = {
        "start_time": start_epoch,
        "end_time": end_epoch,
        "bucket_width": "1d",
        "limit": 31,
    }
    response = session.get(OPENAI_USAGE_URL, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    payload = response.json()

    total_requests = 0
    total_tokens = 0
    for bucket in payload.get("data", []):
        rows = bucket.get("results") or bucket.get("result") or []
        for row in rows:
            requests_count = int(row.get("num_model_requests", 0) or 0)
            input_tokens = int(row.get("input_tokens", 0) or 0)
            output_tokens = int(row.get("output_tokens", 0) or 0)
            total_requests += requests_count
            total_tokens += input_tokens + output_tokens

            input_details = row.get("input_tokens_details") or {}
            cached_tokens = int(input_details.get("cached_tokens", 0) or 0)
            total_tokens += cached_tokens
    return total_requests, total_tokens


def retrieve_openai_usage(session: requests.Session, config: EnvConfig, settings: BotSettings, key: str, tzinfo: datetime.tzinfo):
    tz = pytz.timezone(config.timezone)
    now_local = datetime.datetime.now(tz)
    month_start_local = now_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    month_start_epoch = epoch_utc(month_start_local)
    current_epoch = int(datetime.datetime.now(datetime.timezone.utc).timestamp()) + 1
    end_time_utc = datetime.datetime.fromtimestamp(current_epoch - 1, tz=datetime.timezone.utc)
    end_time_local = end_time_utc.astimezone(tzinfo)

    previous_month_end_local = month_start_local - datetime.timedelta(seconds=1)
    previous_month_start_local = previous_month_end_local.replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    previous_month_end_epoch = epoch_utc(
        previous_month_end_local.replace(hour=23, minute=59, second=59, microsecond=0)
    ) + 1
    previous_month_start_epoch = epoch_utc(previous_month_start_local)

    mtd_cost = sum_openai_costs(session, key, month_start_epoch, current_epoch)
    last_month_cost = sum_openai_costs(session, key, previous_month_start_epoch, previous_month_end_epoch)
    mtd_requests, mtd_tokens = sum_openai_usage(session, key, month_start_epoch, current_epoch)

    return {
        "month_start_local": month_start_local,
        "previous_month_start_local": previous_month_start_local,
        "previous_month_end_local": previous_month_end_local,
        "mtd_cost": mtd_cost,
        "last_month_cost": last_month_cost,
        "mtd_requests": mtd_requests,
        "mtd_tokens": mtd_tokens,
        "refreshed_at": now_local,
        "end_time_local": end_time_local,
        "end_time_utc": end_time_utc,
    }


def format_openai_usage_report(usage: dict) -> str:
    mtd_cost_str = f"${usage['mtd_cost']:,.4f}"
    last_month_cost_str = f"${usage['last_month_cost']:,.4f}"
    return (
        "*📊 OpenAI Costs*\n"
        f"• MTD Cost: `{mtd_cost_str}`\n"
        f"• Last Month: `{last_month_cost_str}`"
    )


def refresh_openai_usage(session: requests.Session, config: EnvConfig, settings: BotSettings, state: BotState) -> Optional[dict]:
    if not config.openai_admin_key:
        return None

    with state.openai_usage_lock:
        try:
            # Pass None or dummy for tzinfo as retrieve_openai_usage now uses global constant
            usage = retrieve_openai_usage(session, config, settings, config.openai_admin_key, None)
        except requests.RequestException as exc:
            state.openai_usage = None
            state.openai_usage_error = f"OpenAI usage unavailable: {exc}"
            return None
        except Exception as exc:
            state.openai_usage = None
            state.openai_usage_error = f"OpenAI usage error: {exc}"
            return None

        state.openai_usage = usage
        state.openai_usage_error = None
        return usage


def start_openai_usage_worker(
    session: requests.Session, config: EnvConfig, settings: BotSettings, state: BotState, interval_seconds: int
) -> Optional[threading.Thread]:
    if not config.openai_admin_key:
        return None

    def worker():
        sleep_interval = max(60, interval_seconds)
        while True:
            refresh_openai_usage(session, config, settings, state)
            sleep_interval = max(60, OPENAI_REFRESH_SECONDS)
            time.sleep(sleep_interval)

    thread = threading.Thread(target=worker, name="openai-usage-worker", daemon=True)
    thread.start()
    return thread


def compose_status_message(
    state: BotState,
    config: EnvConfig,
    status_info: Optional[str],
    current_pnl: Union[float, str],
    *,
    openai_line: Optional[str] = None,
    spot_balance: Optional[Union[float, str]] = None,
) -> str:
    lines = [
        "🧭 Status:",
        f"• Running: `{state.is_running}`",
        f"• Interval: `{state.interval_seconds / 60:.1f}m`",
        f"• Night mode: `{state.night_mode_enabled}` (active: `{state.night_mode_active}`)",
        f"• Alert limit: `{state.pnl_alert_low} USDT ~ {state.pnl_alert_high} USDT`",
        f"• Uptime: `{get_uptime(state)}`",
        f"• Lunar Date: `{get_lunar_date_string(config.timezone)}`",
        f"• TODO Left: `{get_todo_count()}`",
    ]
    if state.init_capital:
        lines.append(f"• Init Capital: `{state.init_capital:,.2f} USDT`")

    if openai_line:
        lines.append(openai_line)

    lines.extend([
        "",
        "💰 *Spot Balance:*",
    ])
    if spot_balance is not None:
        if isinstance(spot_balance, dict):
            total = spot_balance.get("total", 0.0)
            pnl_perc_line = ""
            if state.init_capital:
                pnl_perc = (total - state.init_capital) / state.init_capital * 100
                pnl_perc_line = f" ({pnl_perc:+.2f}%)"

            lines.append(f"• Total: `{total:,.2f} USDT`{pnl_perc_line}")
            lines.append(f"• Max: `{state.max_spot_balance:,.2f} USDT`, Min: `{state.min_spot_balance:,.2f} USDT`")
            breakdown = spot_balance.get("breakdown", [])
            for item in breakdown[:5]:  # Show top 5 assets
                price_str = f" @ {item['price']:,.4f}" if item['asset'] != "USDT" else ""
                lines.append(f"  ▫️ `{item['asset']}`: `{item['usdt_value']:,.2f} USDT`{price_str}")
            if len(breakdown) > 5:
                lines.append(f"  ▫️ ... and {len(breakdown)-5} more assets")
        elif isinstance(spot_balance, (int, float)):
            total = float(spot_balance)
            pnl_perc_line = ""
            if state.init_capital:
                pnl_perc = (total - state.init_capital) / state.init_capital * 100
                pnl_perc_line = f" ({pnl_perc:+.2f}%)"

            lines.append(f"• Total: `{total:,.2f} USDT`{pnl_perc_line}")
            lines.append(f"• Max: `{state.max_spot_balance:,.2f} USDT`, Min: `{state.min_spot_balance:,.2f} USDT`")
        else:
            lines.append(f"• {spot_balance}")

    lines.extend([
        "",
        "📊 *Futures PnL:*",
    ])
    if isinstance(current_pnl, (int, float)):
        lines.append(f"• Current PnL: `{current_pnl:,.2f} USDT`")
    else:
        lines.append(f"• Current PnL: `{current_pnl}`")
    lines.extend([
        f"• Max PnL: `{state.max_pnl} USDT`, Min: `{state.min_pnl} USDT`",
    ])

    return "\n".join(lines)


def build_openai_status_line(state: BotState) -> Optional[str]:
    with state.openai_usage_lock:
        usage = state.openai_usage
        error = state.openai_usage_error

    if usage:
        return (
            f"• OpenAI cost (MTD): `${usage['mtd_cost']:,.4f}` "
            f"(last month `${usage['last_month_cost']:,.4f}`)"
        )
    if error:
        return f"• {error}"
    return "• OpenAI usage: fetching..."


def load_env_config() -> EnvConfig:
    def require_env(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return value


    config = EnvConfig(
        api_key=require_env("API_KEY"),
        api_secret=require_env("API_SECRET"),
        telegram_token=require_env("TELEGRAM_TOKEN"),
        telegram_chat_id=require_env("TELEGRAM_CHAT_ID"),
        openai_admin_key=os.getenv("OPENAI_ADMIN_KEY"),
        iqair_api_key=os.getenv("IQAIR_API_KEY"),
        iqair_latitude=env_float("IQAIR_LATITUDE", 10.8231),
        iqair_longitude=env_float("IQAIR_LONGITUDE", 106.6297),
        outage_street_filter=os.getenv("PNL_BOT_OUTAGE_STREET_FILTER"),
        evn_madvi=env_str("PNL_BOT_EVN_MADVI", "PB0100"),
        evn_area_name=env_str("PNL_BOT_EVN_AREA_NAME", "Ho Chi Minh"),
        timezone=env_str("PNL_BOT_TIMEZONE", "Asia/Ho_Chi_Minh"),
        cpu_alert_threshold=env_int("PNL_BOT_CPU_ALERT_THRESHOLD", 80),
        mem_alert_threshold=env_int("PNL_BOT_MEM_ALERT_THRESHOLD", 80),
        disk_alert_threshold=env_int("PNL_BOT_DISK_ALERT_THRESHOLD", 80),
    )
    if config.outage_street_filter:
        print(f"DEBUG: Outage Filter loaded: '{config.outage_street_filter}'", flush=True)
    else:
        print("DEBUG: No Outage Filter loaded.", flush=True)
    return config


def load_persisted_state(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not load persisted state: {exc}")
        return None


def apply_persisted_configuration(persisted: dict, state: BotState, settings: BotSettings) -> None:
    if not persisted:
        return

    state_data = persisted.get("state", persisted)

    def _safe_int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _safe_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    state.interval_seconds = _safe_int(state_data.get("interval_seconds"), state.interval_seconds)
    state.night_mode_enabled = bool(state_data.get("night_mode_enabled", state.night_mode_enabled))
    state.is_running = bool(state_data.get("is_running", state.is_running))
    state.pnl_alert_low = _safe_int(state_data.get("pnl_alert_low"), state.pnl_alert_low)
    state.pnl_alert_high = _safe_int(state_data.get("pnl_alert_high"), state.pnl_alert_high)
    state.max_pnl = _safe_float(state_data.get("max_pnl"), state.max_pnl)
    state.min_pnl = _safe_float(state_data.get("min_pnl"), state.min_pnl)
    state.max_spot_balance = _safe_float(state_data.get("max_spot_balance"), state.max_spot_balance)
    state.min_spot_balance = _safe_float(state_data.get("min_spot_balance"), state.min_spot_balance)
    state.init_capital = _safe_float(state_data.get("init_capital"), state.init_capital)

    night_window = state_data.get("night_mode_window")
    if isinstance(night_window, (list, tuple)) and len(night_window) == 2:
        try:
            start_hour = int(night_window[0])
            end_hour = int(night_window[1])
            state.night_mode_window = (start_hour, end_hour)
            settings.night_mode_window = state.night_mode_window
        except (TypeError, ValueError):
            pass

    state.power_outages = state_data.get("power_outages", state.power_outages)
    state.last_outage_check = _safe_float(state_data.get("last_outage_check"), state.last_outage_check)


def persist_runtime_state(path: str, state: BotState, settings: BotSettings) -> None:
    state_data = {
        "interval_seconds": state.interval_seconds,
        "night_mode_enabled": state.night_mode_enabled,
        "is_running": state.is_running,
        "pnl_alert_low": state.pnl_alert_low,
        "pnl_alert_high": state.pnl_alert_high,
        "max_pnl": state.max_pnl,
        "min_pnl": state.min_pnl,
        "max_spot_balance": state.max_spot_balance,
        "min_spot_balance": state.min_spot_balance,
        "init_capital": state.init_capital,
        "night_mode_window": list(state.night_mode_window),
        "power_outages": state.power_outages,
        "last_outage_check": state.last_outage_check,
    }
    payload = {
        "state": state_data,
    }

    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp_path, path)
    except OSError as exc:
        print(f"Could not persist state: {exc}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def notify_exit(session: requests.Session, config: EnvConfig, settings: BotSettings) -> None:
    try:
        send_telegram_message(session, config, settings, "❌ Bot has been stopped.", force_send=True)
    except Exception as exc:
        print(f"Error while sending shutdown notification: {exc}")


def get_futures_pnl(session: requests.Session, config: EnvConfig) -> Union[float, str]:
    base_url = "https://fapi.binance.com"
    endpoint = "/fapi/v2/account"
    timestamp = int(time.time() * 1000)
    query_string = f"timestamp={timestamp}"
    signature = hmac.new(config.api_secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()
    url = f"{base_url}{endpoint}?{query_string}&signature={signature}"
    headers = {"X-MBX-APIKEY": config.api_key}

    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        positions = data.get("positions", [])
        total_unrealized_pnl = sum(float(position.get("unrealizedProfit", 0.0)) for position in positions)
        return round(total_unrealized_pnl, 2)
    except Exception as exc:
        return f"PnL fetch error: {exc}"


def get_spot_balance(session: requests.Session, config: EnvConfig) -> Union[dict, str]:
    base_url = "https://api.binance.com"
    endpoint = "/api/v3/account"
    timestamp = int(time.time() * 1000)
    query_string = f"timestamp={timestamp}"
    signature = hmac.new(config.api_secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()
    url = f"{base_url}{endpoint}?{query_string}&signature={signature}"
    headers = {"X-MBX-APIKEY": config.api_key}

    try:
        # 1. Get Account Balances
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        balances = data.get("balances", [])

        # 2. Get Current Prices
        price_url = "https://api.binance.com/api/v3/ticker/price"
        price_response = session.get(price_url, timeout=10)
        price_response.raise_for_status()
        prices = {item["symbol"]: float(item["price"]) for item in price_response.json()}

        total_usdt = 0.0
        breakdown = []

        for balance in balances:
            asset = balance.get("asset")
            free = float(balance.get("free", 0.0))
            locked = float(balance.get("locked", 0.0))
            amount = free + locked

            if amount <= 0:
                continue

            asset_usdt_value = 0.0
            if asset == "USDT":
                asset_usdt_value = amount
            else:
                symbol = f"{asset}USDT"
                if symbol in prices:
                    asset_usdt_value = amount * prices[symbol]
                else:
                    # Try getting price from other pairs if needed, but USDT is usually the base
                    continue

            if asset_usdt_value < 0.01:  # Filter out dust
                continue

            total_usdt += asset_usdt_value
            breakdown.append({
                "asset": asset,
                "amount": amount,
                "usdt_value": asset_usdt_value,
                "price": prices.get(symbol, 1.0) if asset != "USDT" else 1.0
            })

        # Sort breakdown by USDT value descending
        breakdown.sort(key=lambda x: x["usdt_value"], reverse=True)

        return {
            "total": round(total_usdt, 2),
            "breakdown": breakdown,
            "btc_price": prices.get("BTCUSDT", 0.0)
        }
    except Exception as exc:
        return f"Spot balance fetch error: {exc}"


def get_air_quality(session: requests.Session, config: EnvConfig) -> Union[dict, str]:
    """Fetch air quality data from IQAir API using GPS coordinates."""
    if not config.iqair_api_key:
        return "IQAir API key not configured"

    try:
        params = {
            "lat": config.iqair_latitude,
            "lon": config.iqair_longitude,
            "key": config.iqair_api_key,
        }
        response = session.get(IQAIR_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "success":
            return f"IQAir API error: {data.get('message', 'Unknown error')}"

        result_data = data.get("data", {})
        current = result_data.get("current", {})
        pollution = current.get("pollution", {})
        weather = current.get("weather", {})

        return {
            "city": result_data.get("city", "Unknown"),
            "country": result_data.get("country", "Unknown"),
            "aqi_us": pollution.get("aqius", 0),
            "temperature": weather.get("tp", 0),
            "humidity": weather.get("hu", 0),
        }
    except Exception as exc:
        return f"Air quality fetch error: {exc}"


def get_top_processes(n: int = 5) -> Tuple[str, str]:
    processes = []
    for proc in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_percent"]):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    top_cpu = sorted(processes, key=lambda x: x["cpu_percent"], reverse=True)[:n]
    top_mem = sorted(processes, key=lambda x: x["memory_percent"], reverse=True)[:n]

    cpu_info = "\n".join(
        [f"• `{proc['name']}` (PID `{proc['pid']}`): `{proc['cpu_percent']}%` CPU" for proc in top_cpu]
    )
    mem_info = "\n".join(
        [f"• `{proc['name']}` (PID `{proc['pid']}`): `{round(proc['memory_percent'], 1)}%` RAM" for proc in top_mem]
    )
    return cpu_info, mem_info


def get_system_info_text(config: EnvConfig, settings: BotSettings, show_all: bool = False) -> Optional[str]:
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent

    is_alert = (
        cpu > config.cpu_alert_threshold
        or mem > config.mem_alert_threshold
        or disk > config.disk_alert_threshold
    )
    if not show_all and not is_alert:
        return None

    info_lines = []
    title = "*🖥 System Alert:*" if is_alert else "*📊 Current system metrics:*"
    info_lines.append(title)

    def format_line(label, val, thresh):
        exceeded = val > thresh
        if exceeded:
            return f"🔴 *{label}: `{val}%` (alert when > `{thresh}%`)*"
        return f"• {label}: `{val}%` (alert when > `{thresh}%`)"

    # Add lines: always if show_all, or only if exceeded if is_alert
    for label, val, thresh in [
        ("CPU", cpu, config.cpu_alert_threshold),
        ("RAM", mem, config.mem_alert_threshold),
        ("Disk", disk, config.disk_alert_threshold),
    ]:
        if show_all or val > thresh:
            info_lines.append(format_line(label, val, thresh))

    top_cpu, top_mem = get_top_processes()
    info_lines.append("\n*⚙️ Top CPU processes:*\n" + top_cpu)
    info_lines.append("\n*💾 Top RAM processes:*\n" + top_mem)

    return "\n".join(info_lines)


def remove_accents(input_str: str) -> str:
    """Removes accents from Vietnamese characters, including the letter 'đ'."""
    if not input_str:
        return ""
    # Normalize unicode to decompose accents
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    # Filter out non-spacing marks (accents)
    s = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    # Specifically handle Vietnamese 'đ' and 'Đ'
    s = s.replace('đ', 'd').replace('Đ', 'D')
    return s


def send_telegram_message(
    session: requests.Session,

    config: EnvConfig,
    settings: BotSettings,
    message: str,
    *,
    state: Optional[BotState] = None,
    force_send: bool = False,
) -> None:
    tz = pytz.timezone(config.timezone)

    now_hour = datetime.datetime.now(tz).hour
    if state and state.night_mode_enabled and not force_send:
        start_hour, end_hour = state.night_mode_window
        if start_hour <= end_hour:
            if start_hour <= now_hour < end_hour:
                return
        else:
            if now_hour >= start_hour or now_hour < end_hour:
                return

    url = f"{TELEGRAM_API_URL}/bot{config.telegram_token}/sendMessage"
    payload = {
        "chat_id": config.telegram_chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }
    try:
        if len(message) > TELEGRAM_MAX_MESSAGE:
            for idx in range(0, len(message), TELEGRAM_MAX_MESSAGE):
                res = session.post(
                    url,
                    data={**payload, "text": message[idx : idx + TELEGRAM_MAX_MESSAGE]},
                    timeout=10,
                )
                print(f"DEBUG: Telegram response chunk: {res.text}", flush=True)
                res.raise_for_status()
        else:
            res = session.post(url, data=payload, timeout=10)
            print(f"DEBUG: Telegram response: {res.text}", flush=True)
            res.raise_for_status()
    except Exception as exc:
        print(f"Telegram send error: {exc}", flush=True)


def get_uptime(state: BotState) -> str:
    uptime_seconds = int(time.time() - state.start_time)

    months, remainder = divmod(uptime_seconds, 2592000) # 30 days
    days, remainder = divmod(remainder, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if months > 0:
        parts.append(f"{months}mo")
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or (not parts):
        parts.append(f"{hours}h")
    if minutes > 0 or (not parts and hours == 0):
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return ",".join(parts)


def get_lunar_date_string(timezone_name: str) -> str:
    try:
        now = datetime.datetime.now(pytz.timezone(timezone_name))
        lunar = solar_to_lunar(now)
        leap_str = " (Nhuận)" if lunar.leap else ""
        return f"{lunar.day}/{lunar.month}/{lunar.year}{leap_str}"
    except Exception as exc:
        return f"Error: {exc}"


def get_power_outages(session: requests.Session, config: EnvConfig) -> List[dict]:
    """Fetches upcoming power outages from EVN SPC API, based on configured Ma Don Vi."""
    tz = pytz.timezone(config.timezone)
    now = datetime.datetime.now(tz)
    tomorrow = now + datetime.timedelta(days=1)
    end_date = now + datetime.timedelta(days=7)

    params = {
        "madvi": config.evn_madvi,
        "tuNgay": now.strftime("%d-%m-%Y"),
        "denNgay": end_date.strftime("%d-%m-%Y"),
        "ChucNang": "MaDonVi",
    }

    try:
        response = session.get(EVN_SPC_OUTAGE_URL, params=params, timeout=10)
        response.raise_for_status()
        html_content = response.text

        # Robust parsing for the specific structure found in live responses
        blocks = re.findall(r'<div class="entry">(.*?)</div>\s*<br />', html_content, re.DOTALL)

        outages = []
        for block in blocks:
            # Area extraction
            area_match = re.search(r'class="where"><b>KHU VỰC:</b>\s*(.*?)</span>', block, re.DOTALL)
            # Time extraction (handling multi-line and whitespace)
            time_match = re.search(r'class="time">.*?<span style="white-space:nowrap;">\s*(.*?)\s*</span>\s*</span>', block, re.DOTALL)
            # Reason extraction
            reason_match = re.search(r'class="cause">.*?<span>(.*?)</span>\s*</span>', block, re.DOTALL)

            if area_match and time_match:
                area = html.unescape(area_match.group(1).strip())
                # Clean up time string (remove extra whitespace/newlines)
                time_info = html.unescape(time_match.group(1).strip())
                time_info = re.sub(r'\s+', ' ', time_info)

                # Skip if filter is set and does not match area (accent-insensitive)
                filter_str = config.outage_street_filter
                if filter_str:
                    normalized_area = remove_accents(area).lower()
                    normalized_filter = remove_accents(filter_str).lower()
                    print(f"DEBUG: Checking normalized area '{normalized_area}' against filter '{normalized_filter}'", flush=True)
                    if normalized_filter not in normalized_area:
                        print(f"DEBUG: Filter mismatch, skipping.", flush=True)
                        continue
                    print(f"DEBUG: Filter match!", flush=True)

                reason = html.unescape(reason_match.group(1).strip()) if reason_match else "N/A"
                reason = re.sub(r'\s+', ' ', reason)

                # Use area + time as a unique key for deduplication
                outage_id = hashlib.md5(f"{area}{time_info}".encode("utf-8")).hexdigest()

                outages.append({
                    "id": outage_id,
                    "area": area,
                    "time": time_info,
                    "reason": reason
                })
        print(f"DEBUG: Found {len(outages)} outages for {config.evn_area_name}.", flush=True)
        return outages
    except Exception as exc:
        print(f"Error fetching power outages: {exc}")
        return []


def refresh_power_outages(session: requests.Session, config: EnvConfig, state: BotState) -> List[dict]:
    """Updates the state with new power outages and returns only the NEW ones."""
    current_outages = get_power_outages(session, config)
    if not current_outages:
        return []

    seen_ids = {o["id"] for o in state.power_outages}
    new_outages = [o for o in current_outages if o["id"] not in seen_ids]

    # Keep only current and future outages in state
    # Since we don't have perfect date parsing here, we'll just keep the latest results
    # and filter by ID to find truly new ones.
    state.power_outages = current_outages
    state.last_outage_check = time.time()

    return new_outages


def get_todo_count() -> int:
    if not os.path.exists(TODO_FILE_PATH):
        return 0
    try:
        with open(TODO_FILE_PATH, "r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except Exception:
        return 0


def ensure_todo_file_exists(todo_file: str) -> None:
    if not os.path.exists(todo_file):
        with open(todo_file, "w", encoding="utf-8") as handle:
            handle.write("")


def append_todo(todo_file: str, todo_item: str) -> None:
    with open(todo_file, "a", encoding="utf-8") as handle:
        handle.write(f"- {todo_item}\n")


def read_todos(todo_file: str) -> Optional[str]:
    if not os.path.exists(todo_file):
        return None
    with open(todo_file, "r", encoding="utf-8") as handle:
        lines = [line for line in handle.readlines() if line.strip()]
    return "".join(lines) if lines else None


def _parse_pnl_low(raw: str, state: BotState, _: BotSettings) -> int:
    value = parse_int_value(raw)
    if value >= state.pnl_alert_high:
        raise ValueError("Lower bound must be less than current upper bound")
    return value


def _parse_pnl_high(raw: str, state: BotState, _: BotSettings) -> int:
    value = parse_int_value(raw)
    if value <= state.pnl_alert_low:
        raise ValueError("Upper bound must be greater than current lower bound")
    return value


def _parse_night_mode_start(raw: str, state: BotState, _: BotSettings) -> int:
    value = parse_int_value(raw, minimum=0, maximum=23)
    if value == state.night_mode_window[1]:
        raise ValueError("Start hour must differ from end hour")
    return value


def _parse_night_mode_end(raw: str, state: BotState, _: BotSettings) -> int:
    value = parse_int_value(raw, minimum=0, maximum=24)
    if value == state.night_mode_window[0]:
        raise ValueError("End hour must differ from start hour")
    return value


def _apply_night_mode_start(value: int, state: BotState, settings: BotSettings) -> None:
    state.night_mode_window = (value, state.night_mode_window[1])
    settings.night_mode_window = state.night_mode_window
    state.night_mode_active = False


def _apply_night_mode_end(value: int, state: BotState, settings: BotSettings) -> None:
    state.night_mode_window = (state.night_mode_window[0], value)
    settings.night_mode_window = state.night_mode_window
    state.night_mode_active = False


def _apply_bool(attribute: str) -> Callable[[bool, BotState, BotSettings], None]:
    def _inner(val: bool, state: BotState, settings: BotSettings) -> None:
        setattr(state, attribute, val)
        if attribute == "night_mode_enabled" and not val:
            state.night_mode_active = False
    return _inner

CONFIG_ORDER = [
    "bot_running",
    "interval_seconds",
    "pnl_alert_low",
    "pnl_alert_high",
    "night_mode_enabled",
    "night_mode_start_hour",
    "night_mode_end_hour",
    "init_capital",
    "max_pnl",
    "min_pnl",
    "max_spot_balance",
    "min_spot_balance",
    "outage_filter",
]


CONFIG_DEFINITIONS: Dict[str, ConfigDefinition] = {
    "bot_running": ConfigDefinition(
        description="Toggle bot activity",
        parser=lambda raw, state, settings: parse_bool_value(raw),
        getter=lambda state, settings: state.is_running,
        applier=lambda value, state, settings: setattr(state, "is_running", value),
    ),
    "interval_seconds": ConfigDefinition(
        description="PnL polling interval in seconds",
        parser=lambda raw, state, settings: parse_int_value(raw, minimum=10, maximum=86400),
        getter=lambda state, settings: state.interval_seconds,
        applier=lambda value, state, settings: setattr(state, "interval_seconds", value),
    ),
    "pnl_alert_low": ConfigDefinition(
        description="Lower unrealized PnL alert threshold",
        parser=_parse_pnl_low,
        getter=lambda state, settings: state.pnl_alert_low,
        applier=lambda value, state, settings: setattr(state, "pnl_alert_low", value),
    ),
    "pnl_alert_high": ConfigDefinition(
        description="Upper unrealized PnL alert threshold",
        parser=_parse_pnl_high,
        getter=lambda state, settings: state.pnl_alert_high,
        applier=lambda value, state, settings: setattr(state, "pnl_alert_high", value),
    ),
    "night_mode_enabled": ConfigDefinition(
        description="Toggle quiet hours",
        parser=lambda raw, state, settings: parse_bool_value(raw),
        getter=lambda state, settings: state.night_mode_enabled,
        applier=_apply_bool("night_mode_enabled"),
    ),
    "night_mode_start_hour": ConfigDefinition(
        description="Quiet hours start hour (0-23)",
        parser=_parse_night_mode_start,
        getter=lambda state, settings: state.night_mode_window[0],
        applier=_apply_night_mode_start,
    ),
    "night_mode_end_hour": ConfigDefinition(
        description="Quiet hours end hour (0-24)",
        parser=_parse_night_mode_end,
        getter=lambda state, settings: state.night_mode_window[1],
        applier=_apply_night_mode_end,
    ),
    "init_capital": ConfigDefinition(
        description="Initial capital for PnL % calculation (0 to disable)",
        parser=lambda raw, state, settings: parse_float_value(raw, minimum=0.0),
        getter=lambda state, settings: state.init_capital or 0.0,
        applier=lambda value, state, settings: setattr(state, "init_capital", value if value > 0 else None),
    ),
    "max_pnl": ConfigDefinition(
        description="Historical maximum unrealized futures PnL",
        parser=lambda raw, state, settings: parse_float_value(raw),
        getter=lambda state, settings: state.max_pnl,
        applier=lambda value, state, settings: setattr(state, "max_pnl", value),
    ),
    "min_pnl": ConfigDefinition(
        description="Historical minimum unrealized futures PnL",
        parser=lambda raw, state, settings: parse_float_value(raw),
        getter=lambda state, settings: state.min_pnl,
        applier=lambda value, state, settings: setattr(state, "min_pnl", value),
    ),
    "max_spot_balance": ConfigDefinition(
        description="Historical maximum total spot USDT balance",
        parser=lambda raw, state, settings: parse_float_value(raw, minimum=0.0),
        getter=lambda state, settings: state.max_spot_balance,
        applier=lambda value, state, settings: setattr(state, "max_spot_balance", value),
    ),
    "min_spot_balance": ConfigDefinition(
        description="Historical minimum total spot USDT balance",
        parser=lambda raw, state, settings: parse_float_value(raw, minimum=0.0),
        getter=lambda state, settings: state.min_spot_balance,
        applier=lambda value, state, settings: setattr(state, "min_spot_balance", value),
    ),
    "outage_filter": ConfigDefinition(
        description="Power outage street filter (env only)",
        parser=lambda raw, state, settings: raw,
        getter=lambda state, settings: "N/A",  # Overridden in listing
        applier=lambda value, state, settings: None,
    ),
}


def format_config_listing(state: BotState, settings: BotSettings, config: EnvConfig) -> str:
    lines = ["*⚙️ Runtime configuration:*"]
    for key in CONFIG_ORDER:
        definition = CONFIG_DEFINITIONS[key]
        if key == "outage_filter":
            value = config.outage_street_filter or "None"
        else:
            value = definition.getter(state, settings)
        lines.append(
            f"• `{key}`: `{format_config_value(value)}` – {definition.description}"
        )
    return "\n".join(lines)


def handle_config_command(text: str, state: BotState, settings: BotSettings, config: EnvConfig) -> str:
    tokens = text.split(maxsplit=3)

    if len(tokens) == 1 or (len(tokens) >= 2 and tokens[1].lower() == "show"):
        return format_config_listing(state, settings, config)

    if len(tokens) < 2:
        return "❌ Usage: `/config show|get|set`"

    action = tokens[1].lower()

    if action == "get":
        if len(tokens) < 3:
            return "❌ Usage: `/config get <name>`"
        key = tokens[2].lower()
        definition = CONFIG_DEFINITIONS.get(key)
        if not definition:
            return "❌ Unknown configuration key. Use `/config show` to list options."
        value = definition.getter(state, settings)
        return f"`{key}` = `{format_config_value(value)}` – {definition.description}"

    if action == "set":
        if len(tokens) < 4:
            return "❌ Usage: `/config set <name> <value>`"
        key = tokens[2].lower()
        value_text = tokens[3]
        definition = CONFIG_DEFINITIONS.get(key)
        if not definition:
            return "❌ Unknown configuration key. Use `/config show` to list options."
        try:
            parsed_value = definition.parser(value_text, state, settings)
            result = definition.applier(parsed_value, state, settings)
            persist_runtime_state(STATE_FILE_PATH, state, settings)
            if isinstance(result, str) and result:
                extra = f"\n{result}"
            else:
                extra = ""
            updated = definition.getter(state, settings)
            return f"✅ `{key}` updated to `{format_config_value(updated)}`{extra}"
        except ValueError as exc:
            return f"❌ {exc}"

    return "❌ Unsupported config command. Use `/config show|get|set`."



def check_telegram_commands(
    session: requests.Session,
    openai_session: Optional[requests.Session],
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
        print(f"Telegram command error: {exc}")
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
        original_text = (message.get("text") or "").strip()
        chat_id = str(message.get("chat", {}).get("id", ""))

        if not original_text.startswith("/"):
            continue

        # Extract base command and handle @botname suffix
        parts = original_text.split()
        cmd_part = parts[0].lower()
        if "@" in cmd_part:
            cmd_part = cmd_part.split("@")[0]

        text = cmd_part
        if len(parts) > 1:
            text += " " + " ".join(parts[1:])

        if text.startswith("/"):
            print(f"DEBUG: Received command '{text}' (original: '{original_text}') from chat_id '{chat_id}'", flush=True)

        if chat_id != str(config.telegram_chat_id):
            continue

        if text == "/config" or text.startswith("/config "):
            response = handle_config_command(text, state, settings, config)
            send_telegram_message(
                session,
                config,
                settings,
                response,
                state=state,
                force_send=True,
            )
        elif text == "/status":
            pnl = get_futures_pnl(session, config)
            spot_balance = get_spot_balance(session, config)
            state_changed = False
            if isinstance(pnl, (int, float)):
                prev_max_p = state.max_pnl
                prev_min_p = state.min_pnl
                if pnl >= 0:
                    state.max_pnl = max(state.max_pnl, pnl)
                if pnl <= 0:
                    state.min_pnl = min(state.min_pnl, pnl)
                state_changed = state_changed or prev_max_p != state.max_pnl or prev_min_p != state.min_pnl

            if isinstance(spot_balance, dict):
                total_s = spot_balance.get("total", 0.0)
                if total_s > 0:
                    prev_max_s = state.max_spot_balance
                    prev_min_s = state.min_spot_balance
                    if state.max_spot_balance == 0: state.max_spot_balance = total_s
                    if state.min_spot_balance == 0: state.min_spot_balance = total_s
                    state.max_spot_balance = max(state.max_spot_balance, total_s)
                    state.min_spot_balance = min(state.min_spot_balance, total_s)
                    state_changed = state_changed or prev_max_s != state.max_spot_balance or prev_min_s != state.min_spot_balance
            # OpenAI retrieval is now only done via /openai command to save resources
            openai_line = None
            send_telegram_message(
                session,
                config,
                settings,
                compose_status_message(state, config, None, pnl, openai_line=openai_line, spot_balance=spot_balance),
                state=state,
                force_send=True,
            )
            if state_changed:
                persist_runtime_state(STATE_FILE_PATH, state, settings)
        elif text == "/stop":
            handle_config_command("/config set bot_running off", state, settings, config)
            send_telegram_message(
                session,
                config,
                settings,
                "⛔ Bot paused. No alerts will be sent until `/start` is issued.",
                state=state,
                force_send=True,
            )
        elif text == "/start":
            handle_config_command("/config set bot_running on", state, settings, config)
            send_telegram_message(
                session,
                config,
                settings,
                "▶️ Bot resumed. Alerts are active again.",
                state=state,
                force_send=True,
            )
        elif text == "/pnl":
            pnl = get_futures_pnl(session, config)
            state_changed = False
            if isinstance(pnl, (int, float)):
                prev_max = state.max_pnl
                prev_min = state.min_pnl
                state.max_pnl = max(state.max_pnl, pnl)
                state.min_pnl = min(state.min_pnl, pnl)
                state_changed = prev_max != state.max_pnl or prev_min != state.min_pnl
                send_telegram_message(
                    session,
                    config,
                    settings,
                    f"📊 PnL: `{pnl}` USDT, `[{state.min_pnl},{state.max_pnl}]`",
                    state=state,
                    force_send=True,
                )
            else:
                send_telegram_message(session, config, settings, f"`{pnl}`", state=state, force_send=True)
            if state_changed:
                persist_runtime_state(STATE_FILE_PATH, state, settings)
        elif text in {"/openai", "/openaiusage"}:
            if not config.openai_admin_key:
                send_telegram_message(
                    session,
                    config,
                    settings,
                    "❌ OpenAI admin key is not configured. Set OPENAI_ADMIN_KEY to enable this command.",
                    state=state,
                    force_send=True,
                )
                continue
            send_telegram_message(
                session,
                config,
                settings,
                "Retrieving OpenAI usage, please wait ... it might take a while.",
                state=state,
                force_send=True,
            )
            usage_session = openai_session or session
            refresh_openai_usage(usage_session, config, settings, state)
            with state.openai_usage_lock:
                usage = state.openai_usage
                error = state.openai_usage_error
            if usage:
                message = format_openai_usage_report(usage)
            elif error:
                message = f"❌ {error}"
            else:
                message = "ℹ️ OpenAI usage update is still in progress. Try again in a moment."
            send_telegram_message(
                session,
                config,
                settings,
                message,
                state=state,
                force_send=True,
            )
        elif text == "/showtodo":
            print("DEBUG: Processing /showtodo command", flush=True)
            todos = read_todos(TODO_FILE_PATH)
            if todos:
                # Use a code block to prevent Markdown parsing errors with special characters
                send_telegram_message(
                    session,
                    config,
                    settings,
                    f"*📋 TODO list:*\n```text\n{todos}\n```",
                    state=state,
                    force_send=True,
                )
            else:
                send_telegram_message(
                    session,
                    config,
                    settings,
                    "📭 The TODO list is currently empty.",
                    state=state,
                    force_send=True,
                )
        elif text == "/spot" or text.startswith("/spot "):
            # specific logic to allow "/spot reset"
            parts = text.split()
            reset_mode = len(parts) > 1 and parts[1].strip().lower() == "reset"

            spot_balance = get_spot_balance(session, config)
            if isinstance(spot_balance, dict):
                total = spot_balance.get("total", 0.0)
                if total > 0:
                    prev_max_s = state.max_spot_balance
                    prev_min_s = state.min_spot_balance

                    # If reset is requested, snap min/max to current total
                    if reset_mode:
                        state.max_spot_balance = total
                        state.min_spot_balance = total

                    if state.max_spot_balance == 0: state.max_spot_balance = total
                    if state.min_spot_balance == 0: state.min_spot_balance = total
                    state.max_spot_balance = max(state.max_spot_balance, total)
                    state.min_spot_balance = min(state.min_spot_balance, total)
                    if reset_mode or prev_max_s != state.max_spot_balance or prev_min_s != state.min_spot_balance:
                        persist_runtime_state(STATE_FILE_PATH, state, settings)

                breakdown = spot_balance.get("breakdown", [])
                pnl_perc_info = ""
                if state.init_capital:
                    pnl_perc = (total - state.init_capital) / state.init_capital * 100
                    pnl_perc_info = f" ({pnl_perc:+.2f}%)"

                msg_lines = [
                    f"💰 *Spot Balance:* `{total:,.2f} USDT`{pnl_perc_info}",
                    f"📊 *Range:* `[{state.min_spot_balance:,.2f}, {state.max_spot_balance:,.2f}]`"
                ]
                if breakdown:
                    msg_lines.append("\n*Asset Breakdown:*")
                    for item in breakdown:
                        price_str = f" @ {item['price']:,.4f}" if item['asset'] != "USDT" else ""
                        msg_lines.append(f"• `{item['asset']}`: `{item['usdt_value']:,.2f} USDT`{price_str}")

                send_telegram_message(
                    session,
                    config,
                    settings,
                    "\n".join(msg_lines),
                    state=state,
                    force_send=True,
                )
            elif isinstance(spot_balance, (int, float)):
                send_telegram_message(
                    session,
                    config,
                    settings,
                    f"💰 Spot USDT Balance: {spot_balance:,.2f} USDT",
                    state=state,
                    force_send=True,
                )
            else:
                send_telegram_message(session, config, settings, spot_balance, state=state, force_send=True)
        elif text == "/outage":
            outages = get_power_outages(session, config)
            if outages:
                filter_info = f" (Street: `{config.outage_street_filter}`)" if config.outage_street_filter else ""
                lines = [f"📅 *Upcoming Power Outages ({config.evn_area_name}){filter_info}:*"]
                for o in outages:
                    lines.append(f"• `{o['time']}`")
                    lines.append(f"  ▫️ Area: `{o['area']}`")
                    lines.append(f"  ▫️ Reason: `{o['reason']}`")
                message = "\n".join(lines)
            else:
                message = f"✅ No power outages scheduled for the next 7 days in {config.evn_area_name}."

            send_telegram_message(session, config, settings, message, state=state, force_send=True)
        elif text == "/sysinfo":
            sysinfo = get_system_info_text(config, settings, show_all=True)
            send_telegram_message(
                session,
                config,
                settings,
                sysinfo or "No system information available.",
                state=state,
                force_send=True,
            )
        elif text in {"/aqi"}:
            aqi_data = get_air_quality(session, config)
            if isinstance(aqi_data, dict):
                aqi_us = aqi_data.get("aqi_us", 0)
                # Determine AQI level and emoji
                if aqi_us <= 50:
                    aqi_emoji = "🟢"  # Green heart - Good
                    aqi_level = "Good"
                elif aqi_us <= 100:
                    aqi_emoji = "🟡"  # Yellow heart - Moderate
                    aqi_level = "Moderate"
                elif aqi_us <= 150:
                    aqi_emoji = "🟠"  # Orange heart - Unhealthy for Sensitive Groups
                    aqi_level = "Unhealthy for Sensitive Groups"
                elif aqi_us <= 200:
                    aqi_emoji = "🔴"  # Red heart - Unhealthy
                    aqi_level = "Unhealthy"
                elif aqi_us <= 300:
                    aqi_emoji = "🟣"  # Purple heart - Very Unhealthy
                    aqi_level = "Very Unhealthy"
                else:
                    aqi_emoji = "🟤"  # Broken heart - Hazardous
                    aqi_level = "Hazardous"

                message = (
                    f"{aqi_emoji} *Air Quality - {aqi_data.get('city', 'N/A')}*\n"
                    f"• AQI (US): `{aqi_us}` - `{aqi_level}`\n"
                    f"• Temperature: `{aqi_data.get('temperature', 0)}°C`\n"
                    f"• Humidity: `{aqi_data.get('humidity', 0)}%`"
                )
            else:
                message = f"⚠️ {aqi_data}"

            send_telegram_message(
                session,
                config,
                settings,
                message,
                state=state,
                force_send=True,
            )
        elif text == "/todo" or text.startswith("/todo "):
            todo_item = text[len("/todo") :].strip()
            if todo_item:
                try:
                    append_todo(TODO_FILE_PATH, todo_item)
                    send_telegram_message(
                        session,
                        config,
                        settings,
                        f"📝 Added todo:\n`{todo_item}`",
                        state=state,
                        force_send=True,
                    )
                except Exception as exc:
                    send_telegram_message(
                        session,
                        config,
                        settings,
                        f"⚠️ Could not save todo: `{exc}`",
                        state=state,
                        force_send=True,
                    )
            else:
                send_telegram_message(
                    session,
                    config,
                    settings,
                    "⚠️ Empty todo. Usage: `/todo <description>`",
                    state=state,
                    force_send=True,
                )
        elif text == "/lunar":
            lunar_str = get_lunar_date_string(config.timezone)
            send_telegram_message(
                session,
                config,
                settings,
                f"📅 *Lunar Calendar (VN):* `{lunar_str}`",
                state=state,
                force_send=True,
            )
        elif text == "/help":
            send_telegram_message(
                session,
                config,
                settings,
                "*ℹ️ Info commands:*\n"
                "• `/status` – Comprehensive snapshot (PnL, Spot, Config)\n"
                "• `/pnl` – Quick unrealized PnL check\n"
                "• `/spot` – Quick spot balance check\n"
                "• `/aqi` – Air quality index (IQAir)\n"
                "• `/sysinfo` – System information\n"
                "• `/openai` – Usage and cost stats\n"
                "• `/showtodo` – View the TODO list\n"
                "• `/lunar` – View current lunar calendar (VN)\n"
                "• `/outage` – View power outage schedule\n"
                "• `/help` – This reference\n"
                "\n*🛠 Configuration:*\n"
                "• `/config show` – View all runtime parameters\n"
                "• `/config set <key> <value>` – Update a parameter\n"
                "• `/start` / `/stop` – Resume or pause alerts\n"
                "• `/todo <text>` – Add an item to TODO\n"
                "• `/spot reset` – reset clear min/max history",
                state=state,
                force_send=True,
            )
        elif text.startswith("/"):
            # Ensure we don't catch just "/" or very short noise
            if len(text.split()[0]) > 1:
                send_telegram_message(
                    session,
                    config,
                    settings,
                    "⚠️ Unsupported command. Type `/help` for the command list.",
                    state=state,
                    force_send=True,
                )

    return latest_id


def monitor_loop(session: requests.Session, config: EnvConfig, settings: BotSettings, state: BotState) -> None:
    pnl = get_futures_pnl(session, config)
    spot_balance = get_spot_balance(session, config)

    if isinstance(pnl, str):
        send_telegram_message(session, config, settings, pnl, state=state, force_send=True)
        return

    state_changed = False

    # Update Futures Stats
    if isinstance(pnl, (int, float)):
        prev_max_p = state.max_pnl
        prev_min_p = state.min_pnl
        if pnl >= 0:
            state.max_pnl = max(state.max_pnl, pnl)
        if pnl <= 0:
            state.min_pnl = min(state.min_pnl, pnl)
        if state.max_pnl != prev_max_p or state.min_pnl != prev_min_p:
            state_changed = True

    # Update Spot Stats
    total_spot = 0.0
    if isinstance(spot_balance, dict):
        total_spot = spot_balance.get("total", 0.0)
    elif isinstance(spot_balance, (int, float)):
        total_spot = float(spot_balance)

    if total_spot > 0:
        prev_max_s = state.max_spot_balance
        prev_min_s = state.min_spot_balance
        if state.max_spot_balance == 0:
            state.max_spot_balance = total_spot
        if state.min_spot_balance == 0:
            state.min_spot_balance = total_spot

        state.max_spot_balance = max(state.max_spot_balance, total_spot)
        state.min_spot_balance = min(state.min_spot_balance, total_spot)
        if state.max_spot_balance != prev_max_s or state.min_spot_balance != prev_min_s:
            state_changed = True

    # Format spot balance message
    spot_msg = ""
    if total_spot > 0:
        pnl_perc_info = ""
        if state.init_capital:
            pnl_perc = (total_spot - state.init_capital) / state.init_capital * 100
            pnl_perc_info = f" ({pnl_perc:+.2f}%)"

        spot_msg = f"💰 *Spot:* `{total_spot:,.2f} USDT`{pnl_perc_info}, `[{state.min_spot_balance:,.2f}, {state.max_spot_balance:,.2f}]`\n"
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
            futures_msg = f"Heavy loss: 🔻 `{pnl}` USDT, `[ {state.min_pnl}, {state.max_pnl} ]`"
        elif pnl >= state.pnl_alert_high:
            futures_msg = f"High profit: 🟢 `{pnl}` USDT, `[ {state.min_pnl}, {state.max_pnl} ]`"
        else:
            futures_msg = f"📊 Futures PnL: `{pnl}` USDT, `[{state.min_pnl},{state.max_pnl}]`"

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

    system_info = get_system_info_text(config, settings)
    if system_info:
        send_telegram_message(session, config, settings, system_info, state=state)


def init_last_update_id(session: requests.Session, config: EnvConfig, settings: BotSettings) -> Optional[int]:
    try:
        url = f"{TELEGRAM_API_URL}/bot{config.telegram_token}/getUpdates"
        response = session.get(url, params={"timeout": 1}, timeout=6)
        response.raise_for_status()
        updates = response.json().get("result") or []
        if updates:
            return max(update["update_id"] for update in updates if "update_id" in update)
    except Exception as exc:
        print(f"Startup could not fetch update_id: {exc}")
    return None


def main() -> None:
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
    openai_session = create_retry_session() if config.openai_admin_key else None
    state = BotState(
        interval_seconds=settings.default_interval_seconds,
        night_mode_enabled=settings.default_night_mode_enabled,
        pnl_alert_low=settings.default_pnl_alert_low,
        pnl_alert_high=settings.default_pnl_alert_high,
        night_mode_window=settings.night_mode_window,
        init_capital=settings.init_capital,
    )

    persisted = load_persisted_state(STATE_FILE_PATH)
    if persisted:
        apply_persisted_configuration(persisted, state, settings)

    ensure_todo_file_exists(TODO_FILE_PATH)
    atexit.register(lambda: notify_exit(session, config, settings))

    try:
        state.last_update_id = init_last_update_id(session, config, settings)
        pnl = get_futures_pnl(session, config)
        if isinstance(pnl, (int, float)):
            state.max_pnl = pnl if pnl > 0 else state.max_pnl
            state.min_pnl = pnl if pnl < 0 else state.min_pnl
        else:
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

        # Fetch spot balance at startup
        spot_balance = get_spot_balance(session, config)
        if isinstance(spot_balance, dict):
            total_s = spot_balance.get("total", 0.0)
            if total_s > 0:
                if state.max_spot_balance == 0: state.max_spot_balance = total_s
                if state.min_spot_balance == 0: state.min_spot_balance = total_s
                state.max_spot_balance = max(state.max_spot_balance, total_s)
                state.min_spot_balance = min(state.min_spot_balance, total_s)
                persist_runtime_state(STATE_FILE_PATH, state, settings)



        # OpenAI retrieval is done on demand
        openai_line = None
        send_telegram_message(
            session,
            config,
            settings,
            compose_status_message(state, config, None, pnl, openai_line=openai_line, spot_balance=spot_balance),
            state=state,
            force_send=True,
        )

        last_run = time.time()
        while True:
            poll_timeout = min(TELEGRAM_POLL_TIMEOUT, max(1, state.interval_seconds // 2))
            state.last_update_id = check_telegram_commands(
                session,
                openai_session,
                config,
                settings,
                state,
                state.last_update_id,
                poll_timeout,
            )

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


if __name__ == "__main__":
    main()
