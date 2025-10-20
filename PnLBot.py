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
from typing import Optional, Tuple, Union

import psutil
import pytz
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OPENAI_COSTS_URL = "https://api.openai.com/v1/organization/costs"
OPENAI_USAGE_URL = "https://api.openai.com/v1/organization/usage/completions"
ASCII_STARTUP_BANNER = (
    "```\n"
    "     /\\\n"
    "    /  \\\n"
    "   / #### \\\n"
    "  <  ####  >\n"
    "   \\ #### /\n"
    "    \\    /\n"
    "     \\/\n"
    "```"
)
OPENAI_REFRESH_DEFAULT_SECONDS = 300


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


@dataclass
class BotSettings:
    default_interval_seconds: int
    default_pnl_alert_low: int
    default_pnl_alert_high: int
    default_night_mode_enabled: bool
    night_mode_window: Tuple[int, int]
    timezone_name: str
    todo_db_file: str
    telegram_max_message: int
    telegram_api_url: str
    system_cpu_alert_threshold: int
    system_memory_alert_threshold: int
    system_disk_alert_threshold: int
    telegram_poll_timeout: int
    state_store_file: str
    openai_refresh_seconds: int


def load_bot_settings() -> BotSettings:
    default_interval = env_int("PNL_BOT_DEFAULT_INTERVAL_SECONDS", 900)
    default_low = env_int("PNL_BOT_DEFAULT_PNL_ALERT_LOW", -20)
    default_high = env_int("PNL_BOT_DEFAULT_PNL_ALERT_HIGH", 20)
    night_mode_start = env_int("PNL_BOT_NIGHT_MODE_START_HOUR", 0)
    night_mode_end = env_int("PNL_BOT_NIGHT_MODE_END_HOUR", 5)
    if not (0 <= night_mode_start <= 23 and 0 <= night_mode_end <= 24):
        raise RuntimeError("Night mode hours must be within 0-24 range")
    if night_mode_start == night_mode_end:
        raise RuntimeError("Night mode start and end hours must differ")
    night_mode_window = (night_mode_start, night_mode_end)

    timezone_name = env_str("PNL_BOT_TIMEZONE", "Asia/Ho_Chi_Minh")
    todo_db_file = env_str("PNL_BOT_TODO_FILE", "pnl-bot-todo-db.txt")
    telegram_max_message = env_int("PNL_BOT_TELEGRAM_MAX_MESSAGE", 4096)
    telegram_api_url = env_str("PNL_BOT_TELEGRAM_API_URL", "https://api.telegram.org")
    cpu_threshold = env_int("PNL_BOT_CPU_ALERT_THRESHOLD", 80)
    memory_threshold = env_int("PNL_BOT_MEMORY_ALERT_THRESHOLD", 80)
    disk_threshold = env_int("PNL_BOT_DISK_ALERT_THRESHOLD", 90)
    poll_timeout = env_int("PNL_BOT_TELEGRAM_POLL_TIMEOUT", 25)
    state_store_file = env_str("PNL_BOT_STATE_FILE", "pnl-bot-state.json")
    default_night_mode = env_bool("PNL_BOT_DEFAULT_NIGHT_MODE_ENABLED", True)
    openai_refresh_seconds = env_int("PNL_BOT_OPENAI_REFRESH_SECONDS", OPENAI_REFRESH_DEFAULT_SECONDS)

    return BotSettings(
        default_interval_seconds=default_interval,
        default_pnl_alert_low=default_low,
        default_pnl_alert_high=default_high,
        default_night_mode_enabled=default_night_mode,
        night_mode_window=night_mode_window,
        timezone_name=timezone_name,
        todo_db_file=todo_db_file,
        telegram_max_message=telegram_max_message,
        telegram_api_url=telegram_api_url,
        system_cpu_alert_threshold=cpu_threshold,
        system_memory_alert_threshold=memory_threshold,
        system_disk_alert_threshold=disk_threshold,
        telegram_poll_timeout=poll_timeout,
        state_store_file=state_store_file,
        openai_refresh_seconds=openai_refresh_seconds,
    )


@dataclass
class EnvConfig:
    api_key: str
    api_secret: str
    telegram_token: str
    telegram_chat_id: str
    openai_admin_key: Optional[str] = None


@dataclass
class BotState:
    interval_seconds: int
    night_mode_enabled: bool
    pnl_alert_low: int
    pnl_alert_high: int
    timezone: datetime.tzinfo
    night_mode_window: Tuple[int, int]
    is_running: bool = True
    last_update_id: Optional[int] = None
    max_pnl: float = 0.0
    min_pnl: float = 0.0
    night_mode_active: bool = False
    start_time: float = field(default_factory=time.time)
    openai_usage: Optional[dict] = None
    openai_usage_error: Optional[str] = None
    openai_usage_lock: Lock = field(default_factory=Lock, repr=False)


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
    now_local = datetime.datetime.now(tzinfo)
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
    refreshed_at = usage.get("refreshed_at")
    end_time_local = usage.get("end_time_local")
    refreshed_str = (
        f"`{refreshed_at:%Y-%m-%d %H:%M %Z}`" if isinstance(refreshed_at, datetime.datetime) else "`unknown`"
    )
    end_time_str = (
        f"`{end_time_local:%Y-%m-%d %H:%M %Z}`" if isinstance(end_time_local, datetime.datetime) else "`unknown`"
    )
    return (
        "*📊 OpenAI Month-to-Date*\n"
        f"• Period: `{usage['month_start_local']:%Y-%m-%d}` → now\n"
        f"• Cost: `{mtd_cost_str}`\n"
        f"• Requests: `{usage['mtd_requests']:,}`\n"
        f"• Tokens: `{usage['mtd_tokens']:,}`\n"
        f"• Last month: `{last_month_cost_str}` (`{usage['previous_month_start_local']:%Y-%m-%d}` → "
        f"`{usage['previous_month_end_local']:%Y-%m-%d}`)\n"
        f"• End time: {end_time_str}\n"
        f"• Updated: {refreshed_str}"
    )


def refresh_openai_usage(session: requests.Session, config: EnvConfig, settings: BotSettings, state: BotState) -> Optional[dict]:
    if not config.openai_admin_key:
        return None

    with state.openai_usage_lock:
        try:
            usage = retrieve_openai_usage(session, config, settings, config.openai_admin_key, state.timezone)
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

    interval = max(60, interval_seconds)

    def worker():
        while True:
            refresh_openai_usage(session, config, settings, state)
            time.sleep(interval)

    thread = threading.Thread(target=worker, name="openai-usage-worker", daemon=True)
    thread.start()
    return thread


def compose_status_message(
    state: BotState,
    status_info: Optional[str],
    current_pnl: Union[float, str],
    *,
    local_time: Optional[datetime.datetime] = None,
    openai_line: Optional[str] = None,
) -> str:
    info = status_info or "No system information available."
    lines = [
        "🧭 Status:",
        f"• Running: `{state.is_running}`",
        f"• Interval: `{state.interval_seconds}s`",
        f"• Night mode: `{state.night_mode_enabled}` (active: `{state.night_mode_active}`)",
        f"• Alert limit: `{state.pnl_alert_low} USDT ~ {state.pnl_alert_high} USDT`",
    ]
    if isinstance(current_pnl, (int, float)):
        lines.append(f"• Current PnL: `{current_pnl:,.2f} USDT`")
    else:
        lines.append(f"• Current PnL: `{current_pnl}`")
    lines.extend([
        f"• Max PnL: `{state.max_pnl} USDT`, Min: `{state.min_pnl} USDT`",
        f"• Uptime: `{get_uptime(state)}`",
    ])
    if local_time is not None:
        lines.insert(2, f"• Local time: `{local_time:%H:%M}`")
    if openai_line:
        lines.append(openai_line)
    lines.append("")
    lines.append(info)
    return "\n".join(lines)


def build_openai_status_line(state: BotState) -> Optional[str]:
    with state.openai_usage_lock:
        usage = state.openai_usage
        error = state.openai_usage_error

    if usage:
        refreshed_at = usage.get("refreshed_at")
        end_time_local = usage.get("end_time_local")
        refreshed_str = (
            f"{refreshed_at:%Y-%m-%d %H:%M %Z}" if isinstance(refreshed_at, datetime.datetime) else "unknown"
        )
        end_time_str = (
            f"{end_time_local:%Y-%m-%d %H:%M %Z}" if isinstance(end_time_local, datetime.datetime) else "unknown"
        )
        return (
            f"• OpenAI cost (MTD): `${usage['mtd_cost']:,.4f}` "
            f"(last month `${usage['last_month_cost']:,.4f}`) "
            f"[end `{end_time_str}` | updated `{refreshed_str}`]"
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

    return EnvConfig(
        api_key=require_env("API_KEY"),
        api_secret=require_env("API_SECRET"),
        telegram_token=require_env("TELEGRAM_TOKEN"),
        telegram_chat_id=require_env("TELEGRAM_CHAT_ID"),
        openai_admin_key=os.getenv("OPENAI_ADMIN_KEY"),
    )


def load_persisted_state(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not load persisted state: {exc}")
        return None


def persist_runtime_state(path: str, state: BotState) -> None:
    data = {
        "interval_seconds": state.interval_seconds,
        "night_mode_enabled": state.night_mode_enabled,
        "is_running": state.is_running,
        "pnl_alert_low": state.pnl_alert_low,
        "pnl_alert_high": state.pnl_alert_high,
        "max_pnl": state.max_pnl,
        "min_pnl": state.min_pnl,
    }
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
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
        [f"`{proc['name']}` (PID {proc['pid']}): {proc['cpu_percent']}% CPU" for proc in top_cpu]
    )
    mem_info = "\n".join(
        [f"`{proc['name']}` (PID {proc['pid']}): {round(proc['memory_percent'], 1)}% RAM" for proc in top_mem]
    )
    return cpu_info, mem_info


def get_system_info_text(settings: BotSettings, show_all: bool = False) -> Optional[str]:
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent

    is_alert = (
        cpu > settings.system_cpu_alert_threshold
        or mem > settings.system_memory_alert_threshold
        or disk > settings.system_disk_alert_threshold
    )
    if not show_all and not is_alert:
        return None

    info_lines = []
    title = "*🖥 System Alert:*" if is_alert else "*📊 Current system metrics:*"
    info_lines.append(title)

    info_lines.append(f"• CPU: `{cpu}%` (alert when > {settings.system_cpu_alert_threshold}%)")
    info_lines.append(f"• RAM: `{mem}%` (alert when > {settings.system_memory_alert_threshold}%)")
    info_lines.append(f"• Disk: `{disk}%` (alert when > {settings.system_disk_alert_threshold}%)")

    top_cpu, top_mem = get_top_processes()
    info_lines.append("\n*⚙️ Top CPU processes:*\n" + top_cpu)
    info_lines.append("\n*💾 Top RAM processes:*\n" + top_mem)

    return "\n".join(info_lines)


def send_telegram_message(
    session: requests.Session,

    config: EnvConfig,
    settings: BotSettings,
    message: str,
    *,
    state: Optional[BotState] = None,
    force_send: bool = False,
) -> None:
    tz = None
    if state:
        tz = state.timezone
    else:
        tz = pytz.timezone(settings.timezone_name)

    now_hour = datetime.datetime.now(tz).hour
    if state and state.night_mode_enabled and not force_send:
        start_hour, end_hour = state.night_mode_window
        if start_hour <= end_hour:
            if start_hour <= now_hour < end_hour:
                return
        else:
            if now_hour >= start_hour or now_hour < end_hour:
                return

    url = f"{settings.telegram_api_url}/bot{config.telegram_token}/sendMessage"
    payload = {
        "chat_id": config.telegram_chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }
    try:
        if len(message) > settings.telegram_max_message:
            for idx in range(0, len(message), settings.telegram_max_message):
                session.post(
                    url,
                    data={**payload, "text": message[idx : idx + settings.telegram_max_message]},
                    timeout=10,
                )
        else:
            session.post(url, data=payload, timeout=10)
    except Exception as exc:
        print(f"Telegram send error: {exc}")


def get_uptime(state: BotState) -> str:
    uptime_seconds = int(time.time() - state.start_time)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h,{minutes}m,{seconds}s"


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


def check_telegram_commands(
    session: requests.Session,
    openai_session: Optional[requests.Session],
    config: EnvConfig,
    settings: BotSettings,
    state: BotState,
    last_update_id: Optional[int],
    poll_timeout: int,
) -> Optional[int]:
    url = f"{settings.telegram_api_url}/bot{config.telegram_token}/getUpdates"
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
        text = (message.get("text") or "").strip()
        chat_id = str(message.get("chat", {}).get("id", ""))

        if chat_id != str(config.telegram_chat_id):
            continue

        if text.startswith("/setinterval"):
            parts = text.split()
            if len(parts) == 2 and parts[1].isdigit():
                value = int(parts[1])
                if 10 <= value <= 3600:
                    state.interval_seconds = value
                    persist_runtime_state(settings.state_store_file, state)
                    send_telegram_message(
                        session,
                        config,
                        settings,
                        f"⏱ Interval updated: {state.interval_seconds} seconds",
                        state=state,
                        force_send=True,
                    )
                else:
                    send_telegram_message(
                        session,
                        config,
                        settings,
                        "❌ Interval must be between 10 and 3600 seconds.",
                        state=state,
                        force_send=True,
                    )
        elif text.startswith("/setlimit"):
            parts = text.split()
            if len(parts) == 3 and parts[1].lstrip("-").isdigit() and parts[2].lstrip("-").isdigit():
                low_limit = int(parts[1])
                high_limit = int(parts[2])
                if low_limit < high_limit:
                    state.pnl_alert_low = low_limit
                    state.pnl_alert_high = high_limit
                    persist_runtime_state(settings.state_store_file, state)
                    send_telegram_message(
                        session,
                        config,
                        settings,
                        f"📈 PnL alert range updated: `{state.pnl_alert_low} ~ {state.pnl_alert_high} USDT`",
                        state=state,
                        force_send=True,
                    )
                else:
                    send_telegram_message(
                        session,
                        config,
                        settings,
                        "❌ The lower bound must be smaller than the upper bound.",
                        state=state,
                        force_send=True,
                    )
        elif text == "/status":
            pnl = get_futures_pnl(session, config)
            state_changed = False
            if isinstance(pnl, (int, float)):
                prev_max = state.max_pnl
                prev_min = state.min_pnl
                state.max_pnl = max(state.max_pnl, pnl)
                state.min_pnl = min(state.min_pnl, pnl)
                state_changed = prev_max != state.max_pnl or prev_min != state.min_pnl
            status_info = get_system_info_text(settings, show_all=True)
            if config.openai_admin_key:
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
                openai_line = build_openai_status_line(state)
            else:
                openai_line = None
            send_telegram_message(
                session,
                config,
                settings,
                compose_status_message(state, status_info, pnl, openai_line=openai_line),
                state=state,
                force_send=True,
            )
            if state_changed:
                persist_runtime_state(settings.state_store_file, state)
        elif text == "/stop":
            state.is_running = False
            persist_runtime_state(settings.state_store_file, state)
            send_telegram_message(
                session,
                config,
                settings,
                "⛔ Bot paused. No alerts will be sent until `/start` is issued.",
                state=state,
                force_send=True,
            )
        elif text == "/start":
            state.is_running = True
            persist_runtime_state(settings.state_store_file, state)
            send_telegram_message(
                session,
                config,
                settings,
                "▶️ Bot resumed. Alerts are active again.",
                state=state,
                force_send=True,
            )
        elif text == "/nightmode off":
            state.night_mode_enabled = False
            persist_runtime_state(settings.state_store_file, state)
            send_telegram_message(
                session,
                config,
                settings,
                "🌙 Night mode is now OFF.",
                state=state,
                force_send=True,
            )
        elif text == "/nightmode on":
            state.night_mode_enabled = True
            persist_runtime_state(settings.state_store_file, state)
            send_telegram_message(
                session,
                config,
                settings,
                "🌙 Night mode is now ON.",
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
                    f"📊 PnL: {pnl} USDT, `[ {state.min_pnl}, {state.max_pnl} ]`",
                    state=state,
                    force_send=True,
                )
            else:
                send_telegram_message(session, config, settings, pnl, state=state, force_send=True)
            if state_changed:
                persist_runtime_state(settings.state_store_file, state)
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
        elif text == "/uptime":
            send_telegram_message(
                session,
                config,
                settings,
                f"⏳ Bot uptime: `{get_uptime(state)}`",
                state=state,
                force_send=True,
            )
        elif text == "/sysinfo":
            sysinfo = get_system_info_text(settings, show_all=True)
            send_telegram_message(
                session,
                config,
                settings,
                sysinfo or "No system information available.",
                state=state,
                force_send=True,
            )
        elif text.startswith("/todo"):
            todo_item = text[len("/todo") :].strip()
            if todo_item:
                try:
                    append_todo(settings.todo_db_file, todo_item)
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
        elif text == "/showtodo":
            todos = read_todos(settings.todo_db_file)
            if todos:
                send_telegram_message(
                    session,
                    config,
                    settings,
                    f"*📋 TODO list:*\n{todos}",
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
        elif text == "/help":
            send_telegram_message(
                session,
                config,
                settings,
                "*📘 Available commands:*\n"
                "• `/setinterval <seconds>` – Update reporting interval\n"
                "• `/setlimit <min> <max>` – Update PnL alert thresholds\n"
                "• `/status` – Show current configuration and metrics\n"
                "• `/start`, `/stop` – Resume or pause alerts\n"
                "• `/pnl` – Show current unrealized PnL\n"
                "• `/nightmode on/off` – Toggle quiet hours\n"
                "• `/uptime` – Show bot uptime\n"
                "• `/sysinfo` – Show system metrics\n"
                "• `/todo <text>` – Append an item to the TODO list\n"
                "• `/showtodo` – Display the TODO list\n"
                "• `/openai` (`/openaiusage`) – Show OpenAI usage and cost\n"
                "• `/help` – Show this command reference",
                state=state,
                force_send=True,
            )
        elif text.startswith("/"):
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
    if isinstance(pnl, str):
        send_telegram_message(session, config, settings, pnl, state=state, force_send=True)
        return

    prev_max = state.max_pnl
    prev_min = state.min_pnl

    if pnl >= 0:
        state.max_pnl = max(state.max_pnl, pnl)
    if pnl <= 0:
        state.min_pnl = min(state.min_pnl, pnl)

    if pnl <= state.pnl_alert_low:
        send_telegram_message(
            session,
            config,
            settings,
            f"Heavy loss: 🔻 {pnl} USDT, `[ {state.min_pnl}, {state.max_pnl} ]`",
            state=state,
        )
    elif pnl >= state.pnl_alert_high:
        send_telegram_message(
            session,
            config,
            settings,
            f"High profit: 🟢 {pnl} USDT, `[ {state.min_pnl}, {state.max_pnl} ]`",
            state=state,
        )
    else:
        send_telegram_message(
            session,
            config,
            settings,
            f"{pnl} USDT, `[ {state.min_pnl}, {state.max_pnl} ]`",
            state=state,
        )

    if state.max_pnl != prev_max or state.min_pnl != prev_min:
        persist_runtime_state(settings.state_store_file, state)

    system_info = get_system_info_text(settings)
    if system_info:
        send_telegram_message(session, config, settings, system_info, state=state)


def init_last_update_id(session: requests.Session, config: EnvConfig, settings: BotSettings) -> Optional[int]:
    try:
        url = f"{settings.telegram_api_url}/bot{config.telegram_token}/getUpdates"
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
    timezone = pytz.timezone(settings.timezone_name)
    state = BotState(
        interval_seconds=settings.default_interval_seconds,
        night_mode_enabled=settings.default_night_mode_enabled,
        pnl_alert_low=settings.default_pnl_alert_low,
        pnl_alert_high=settings.default_pnl_alert_high,
        timezone=timezone,
        night_mode_window=settings.night_mode_window,
    )

    persisted = load_persisted_state(settings.state_store_file)
    if persisted:
        state.interval_seconds = int(persisted.get("interval_seconds", state.interval_seconds))
        state.night_mode_enabled = bool(persisted.get("night_mode_enabled", state.night_mode_enabled))
        state.is_running = bool(persisted.get("is_running", state.is_running))
        state.pnl_alert_low = int(persisted.get("pnl_alert_low", state.pnl_alert_low))
        state.pnl_alert_high = int(persisted.get("pnl_alert_high", state.pnl_alert_high))
        state.max_pnl = float(persisted.get("max_pnl", state.max_pnl))
        state.min_pnl = float(persisted.get("min_pnl", state.min_pnl))

    ensure_todo_file_exists(settings.todo_db_file)
    atexit.register(lambda: notify_exit(session, config, settings))

    try:
        state.last_update_id = init_last_update_id(session, config, settings)
        if openai_session:
            refresh_openai_usage(openai_session, config, settings, state)
            start_openai_usage_worker(openai_session, config, settings, state, settings.openai_refresh_seconds)
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

        persist_runtime_state(settings.state_store_file, state)

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

        now_dt = datetime.datetime.now(state.timezone)
        status_info = get_system_info_text(settings, show_all=True)
        openai_line = build_openai_status_line(state) if config.openai_admin_key else None
        send_telegram_message(
            session,
            config,
            settings,
            compose_status_message(state, status_info, pnl, local_time=now_dt, openai_line=openai_line),
            state=state,
            force_send=True,
        )

        last_run = time.time()
        while True:
            poll_timeout = min(settings.telegram_poll_timeout, max(1, state.interval_seconds // 2))
            state.last_update_id = check_telegram_commands(
                session,
                openai_session,
                config,
                settings,
                state,
                state.last_update_id,
                poll_timeout,
            )

            tz_now = datetime.datetime.now(state.timezone)
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
                persist_runtime_state(settings.state_store_file, state)
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
