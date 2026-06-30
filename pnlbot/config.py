import os
from importlib import metadata
from typing import Optional

from .constants import INTERVAL_MAX_SECONDS, INTERVAL_MIN_SECONDS
from .models import BotSettings, EnvConfig


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


def validate_interval_seconds(value: int, source: str) -> int:
    if not (INTERVAL_MIN_SECONDS <= value <= INTERVAL_MAX_SECONDS):
        raise RuntimeError(
            f"{source} must be between {INTERVAL_MIN_SECONDS} and {INTERVAL_MAX_SECONDS} seconds"
        )
    return value


def is_valid_night_mode_window(start_hour: int, end_hour: int) -> bool:
    return 0 <= start_hour <= 23 and 0 <= end_hour <= 24 and start_hour != end_hour


def load_bot_settings() -> BotSettings:
    default_interval = env_int("PNL_BOT_DEFAULT_INTERVAL_SECONDS", 3600)
    night_mode_start = env_int("PNL_BOT_NIGHT_MODE_START_HOUR", 0)
    night_mode_end = env_int("PNL_BOT_NIGHT_MODE_END_HOUR", 5)
    default_night_mode = env_bool("PNL_BOT_DEFAULT_NIGHT_MODE_ENABLED", True)
    init_capital = env_float("PNL_BOT_INIT_CAPITAL", 0.0)
    validate_interval_seconds(default_interval, "PNL_BOT_DEFAULT_INTERVAL_SECONDS")
    if not (0 <= night_mode_start <= 23 and 0 <= night_mode_end <= 24):
        raise RuntimeError("Night mode hours must be within 0-24 range")
    if night_mode_start == night_mode_end:
        raise RuntimeError("Night mode start and end hours must differ")
    night_mode_window = (night_mode_start, night_mode_end)

    return BotSettings(
        default_interval_seconds=default_interval,
        default_night_mode_enabled=default_night_mode,
        night_mode_window=night_mode_window,
        init_capital=(init_capital if init_capital > 0 else None),
    )


def parse_bool_value(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("Value must be true/false or on/off")


def parse_outage_filter_value(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value or value.lower() in {"none", "null", "clear", "-"}:
        return None
    return value


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


def log_lunar_vn_version() -> str:
    try:
        version = metadata.version("lunar-vn")
    except metadata.PackageNotFoundError:
        version = "not installed"
    print(f"lunar-vn version: {version}", flush=True)
    return version


def load_env_config() -> EnvConfig:
    def require_env(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return value

    return EnvConfig(
        api_key=require_env("API_KEY"),
        api_secret=require_env("API_SECRET"),
        telegram_token=require_env("TELEGRAM_TOKEN"),
        telegram_chat_id=require_env("TELEGRAM_CHAT_ID"),
        iqair_api_key=os.getenv("IQAIR_API_KEY"),
        iqair_latitude=env_float("IQAIR_LATITUDE", 10.8231),
        iqair_longitude=env_float("IQAIR_LONGITUDE", 106.6297),
        outage_street_filter=os.getenv("PNL_BOT_OUTAGE_STREET_FILTER"),
        evn_madvi=env_str("PNL_BOT_EVN_MADVI", "PB0100"),
        evn_area_name=env_str("PNL_BOT_EVN_AREA_NAME", "Ho Chi Minh"),
        timezone=env_str("PNL_BOT_TIMEZONE", "Asia/Ho_Chi_Minh"),
        cpu_alert_threshold=env_int("PNL_BOT_CPU_ALERT_THRESHOLD", 80),
        mem_alert_threshold=env_int("PNL_BOT_MEM_ALERT_THRESHOLD", 80),
        disk_alert_threshold=env_int("PNL_BOT_DISK_ALERT_THRESHOLD", 90),
        freqtrade_api_token=os.getenv("FREQTRADE_API_TOKEN"),
        freqtrade_api_username=os.getenv("FREQTRADE_API_USERNAME"),
        freqtrade_api_password=os.getenv("FREQTRADE_API_PASSWORD"),
    )
