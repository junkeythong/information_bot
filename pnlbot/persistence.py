import json
import os
import shutil
from typing import Optional

from .config import is_valid_night_mode_window, parse_bool_value, parse_outage_filter_value
from .constants import INTERVAL_MAX_SECONDS, INTERVAL_MIN_SECONDS, STATE_BACKUP_SUFFIX
from .models import BotSettings, BotState
from .freqtrade import parse_freqtrade_ports


RUNTIME_CONFIG_OVERRIDES_VERSION = 1


def state_backup_path(path: str) -> str:
    return f"{path}{STATE_BACKUP_SUFFIX}"


def _load_state_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_persisted_state(path: str) -> Optional[dict]:
    backup_path = state_backup_path(path)
    if not os.path.exists(path):
        if os.path.exists(backup_path):
            try:
                print(f"State file {path} not found; loading backup {backup_path}")
                return _load_state_file(backup_path)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"Could not load backup state: {exc}")
        return None
    try:
        return _load_state_file(path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not load persisted state: {exc}")
        if os.path.exists(backup_path):
            try:
                print(f"Loading backup state from {backup_path}")
                return _load_state_file(backup_path)
            except (OSError, json.JSONDecodeError) as backup_exc:
                print(f"Could not load backup state: {backup_exc}")
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

    def _safe_bool(value, default):
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        try:
            return parse_bool_value(str(value))
        except ValueError:
            return default

    def _append_override(target, key: str) -> None:
        if key not in target:
            target.append(key)

    def _parse_ports_value(raw_value) -> list:
        if isinstance(raw_value, (list, tuple)):
            parsed = []
            for raw_port in raw_value:
                try:
                    ports = parse_freqtrade_ports([raw_port])
                except ValueError:
                    continue
                for port in ports:
                    if port not in parsed:
                        parsed.append(port)
            return parsed
        try:
            return parse_freqtrade_ports(raw_value)
        except ValueError:
            return []

    def _infer_legacy_runtime_overrides() -> list:
        inferred = []

        interval_seconds = _safe_int(state_data.get("interval_seconds"), state.interval_seconds)
        if (
            "interval_seconds" in state_data
            and INTERVAL_MIN_SECONDS <= interval_seconds <= INTERVAL_MAX_SECONDS
            and interval_seconds != state.interval_seconds
        ):
            _append_override(inferred, "interval_seconds")

        if "night_mode_enabled" in state_data and _safe_bool(
            state_data.get("night_mode_enabled"), state.night_mode_enabled
        ) != state.night_mode_enabled:
            _append_override(inferred, "night_mode_enabled")

        if "is_running" in state_data and _safe_bool(state_data.get("is_running"), state.is_running) != state.is_running:
            _append_override(inferred, "bot_running")

        pnl_alert_low = _safe_int(state_data.get("pnl_alert_low"), state.pnl_alert_low)
        pnl_alert_high = _safe_int(state_data.get("pnl_alert_high"), state.pnl_alert_high)
        if "pnl_alert_low" in state_data and pnl_alert_low != state.pnl_alert_low:
            _append_override(inferred, "pnl_alert_low")
        if "pnl_alert_high" in state_data and pnl_alert_high != state.pnl_alert_high:
            _append_override(inferred, "pnl_alert_high")

        if "init_capital" in state_data:
            init_capital = _safe_float(state_data.get("init_capital"), state.init_capital or 0.0)
            normalized_init_capital = init_capital if init_capital and init_capital > 0 else None
            if normalized_init_capital != state.init_capital:
                _append_override(inferred, "init_capital")

        if "outage_filter" in state_data:
            outage_filter = parse_outage_filter_value(state_data.get("outage_filter"))
            if outage_filter != state.outage_street_filter:
                _append_override(inferred, "outage_filter")

        night_window = state_data.get("night_mode_window")
        if isinstance(night_window, (list, tuple)) and len(night_window) == 2:
            try:
                start_hour = int(night_window[0])
                end_hour = int(night_window[1])
                if is_valid_night_mode_window(start_hour, end_hour):
                    if start_hour != state.night_mode_window[0]:
                        _append_override(inferred, "night_mode_start_hour")
                    if end_hour != state.night_mode_window[1]:
                        _append_override(inferred, "night_mode_end_hour")
            except (TypeError, ValueError):
                pass

        if "freqtrade_ports" in state_data:
            freqtrade_ports = _parse_ports_value(state_data.get("freqtrade_ports"))
            if freqtrade_ports != state.freqtrade_ports:
                _append_override(inferred, "freqtrade_ports")

        if "freqtrade_alert_cooldown_seconds" in state_data:
            cooldown = _safe_int(
                state_data.get("freqtrade_alert_cooldown_seconds"),
                state.freqtrade_alert_cooldown_seconds,
            )
            if 60 <= cooldown <= 86400 and cooldown != state.freqtrade_alert_cooldown_seconds:
                _append_override(inferred, "freqtrade_alert_cooldown_seconds")

        return inferred

    def _has_versioned_legacy_recovery_signal() -> bool:
        return bool(_parse_ports_value(state_data.get("freqtrade_ports")))

    raw_overrides = state_data.get("runtime_config_overrides")
    if isinstance(raw_overrides, list):
        state.runtime_config_overrides = [str(item) for item in raw_overrides if isinstance(item, str)]
    needs_legacy_migration = state_data.get("runtime_config_overrides_version") != RUNTIME_CONFIG_OVERRIDES_VERSION
    if not state.runtime_config_overrides and (needs_legacy_migration or _has_versioned_legacy_recovery_signal()):
        state.runtime_config_overrides = _infer_legacy_runtime_overrides()
    overrides = set(state.runtime_config_overrides)

    def _has_override(key: str) -> bool:
        return key in overrides

    if _has_override("interval_seconds"):
        interval_seconds = _safe_int(state_data.get("interval_seconds"), state.interval_seconds)
        if INTERVAL_MIN_SECONDS <= interval_seconds <= INTERVAL_MAX_SECONDS:
            state.interval_seconds = interval_seconds

    pre_open_position_interval = _safe_int(
        state_data.get("pre_open_position_interval_seconds"),
        0,
    )
    if INTERVAL_MIN_SECONDS <= pre_open_position_interval <= INTERVAL_MAX_SECONDS:
        state.pre_open_position_interval_seconds = pre_open_position_interval

    if _has_override("night_mode_enabled"):
        state.night_mode_enabled = _safe_bool(state_data.get("night_mode_enabled"), state.night_mode_enabled)
    if _has_override("bot_running"):
        state.is_running = _safe_bool(state_data.get("is_running"), state.is_running)

    pnl_alert_low = _safe_int(state_data.get("pnl_alert_low"), state.pnl_alert_low)
    pnl_alert_high = _safe_int(state_data.get("pnl_alert_high"), state.pnl_alert_high)
    if _has_override("pnl_alert_low") and _has_override("pnl_alert_high") and pnl_alert_low < pnl_alert_high:
        state.pnl_alert_low = pnl_alert_low
        state.pnl_alert_high = pnl_alert_high
    elif _has_override("pnl_alert_low") and pnl_alert_low < state.pnl_alert_high:
        state.pnl_alert_low = pnl_alert_low
    elif _has_override("pnl_alert_high") and pnl_alert_high > state.pnl_alert_low:
        state.pnl_alert_high = pnl_alert_high

    state.max_spot_balance = _safe_float(state_data.get("max_spot_balance"), state.max_spot_balance)
    state.min_spot_balance = _safe_float(state_data.get("min_spot_balance"), state.min_spot_balance)
    if _has_override("init_capital"):
        init_capital = _safe_float(state_data.get("init_capital"), state.init_capital)
        state.init_capital = init_capital if init_capital and init_capital > 0 else None
    if _has_override("outage_filter") and "outage_filter" in state_data:
        outage_filter = parse_outage_filter_value(state_data.get("outage_filter"))
        if outage_filter is not None:
            state.outage_street_filter = outage_filter
    state.last_lunar_alert_date = state_data.get("last_lunar_alert_date", state.last_lunar_alert_date)
    state.last_spot_report_date = state_data.get("last_spot_report_date", state.last_spot_report_date)
    state.pinned_daily_message_id = state_data.get("pinned_daily_message_id", state.pinned_daily_message_id)

    night_window = state_data.get("night_mode_window")
    if (
        (_has_override("night_mode_start_hour") or _has_override("night_mode_end_hour"))
        and isinstance(night_window, (list, tuple))
        and len(night_window) == 2
    ):
        try:
            start_hour = int(night_window[0])
            end_hour = int(night_window[1])
            if is_valid_night_mode_window(start_hour, end_hour):
                state.night_mode_window = (start_hour, end_hour)
                settings.night_mode_window = state.night_mode_window
        except (TypeError, ValueError):
            pass

    state.power_outages = state_data.get("power_outages", state.power_outages)
    futures_position_ranges = state_data.get("futures_position_ranges")
    if isinstance(futures_position_ranges, dict):
        state.futures_position_ranges = futures_position_ranges
    closed_position_ranges = state_data.get("closed_position_ranges")
    if isinstance(closed_position_ranges, list):
        state.closed_position_ranges = [item for item in closed_position_ranges if isinstance(item, dict)]
    state.last_outage_check = _safe_float(state_data.get("last_outage_check"), state.last_outage_check)
    if _has_override("freqtrade_ports"):
        raw_freqtrade_ports = state_data.get("freqtrade_ports")
        if isinstance(raw_freqtrade_ports, (list, tuple)):
            restored_ports = []
            for raw_port in raw_freqtrade_ports:
                try:
                    parsed_ports = parse_freqtrade_ports([raw_port])
                except ValueError:
                    continue
                for port in parsed_ports:
                    if port not in restored_ports:
                        restored_ports.append(port)
            state.freqtrade_ports = restored_ports
        else:
            try:
                state.freqtrade_ports = parse_freqtrade_ports(raw_freqtrade_ports)
            except ValueError:
                state.freqtrade_ports = []
    if _has_override("freqtrade_alert_cooldown_seconds"):
        cooldown = _safe_int(
            state_data.get("freqtrade_alert_cooldown_seconds"),
            state.freqtrade_alert_cooldown_seconds,
        )
        if 60 <= cooldown <= 86400:
            state.freqtrade_alert_cooldown_seconds = cooldown


def _safe_persisted_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _preserve_existing_spot_range(path: str, state_data: dict) -> None:
    if not os.path.exists(path):
        return

    try:
        existing_state = _load_state_file(path).get("state", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return

    existing_max = _safe_persisted_float(existing_state.get("max_spot_balance"))
    existing_min = _safe_persisted_float(existing_state.get("min_spot_balance"))
    current_max = _safe_persisted_float(state_data.get("max_spot_balance"))
    current_min = _safe_persisted_float(state_data.get("min_spot_balance"))

    if existing_max > current_max:
        state_data["max_spot_balance"] = existing_max
    if existing_min > 0 and (current_min <= 0 or existing_min < current_min):
        state_data["min_spot_balance"] = existing_min


def _preserve_existing_runtime_overrides(path: str, state_data: dict) -> None:
    if not os.path.exists(path):
        return

    try:
        existing_state = _load_state_file(path).get("state", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return

    existing_overrides = existing_state.get("runtime_config_overrides")
    if not isinstance(existing_overrides, list):
        return

    current_overrides = state_data.setdefault("runtime_config_overrides", [])
    if not isinstance(current_overrides, list):
        current_overrides = []
        state_data["runtime_config_overrides"] = current_overrides
    current_override_set = set(current_overrides)

    override_fields = {
        "bot_running": ("is_running",),
        "interval_seconds": ("interval_seconds",),
        "pnl_alert_low": ("pnl_alert_low",),
        "pnl_alert_high": ("pnl_alert_high",),
        "night_mode_enabled": ("night_mode_enabled",),
        "night_mode_start_hour": ("night_mode_window",),
        "night_mode_end_hour": ("night_mode_window",),
        "init_capital": ("init_capital",),
        "outage_filter": ("outage_filter",),
        "freqtrade_ports": ("freqtrade_ports",),
        "freqtrade_alert_cooldown_seconds": ("freqtrade_alert_cooldown_seconds",),
    }

    for key in existing_overrides:
        if not isinstance(key, str) or key in current_override_set:
            continue
        fields = override_fields.get(key)
        if not fields:
            continue
        copied = False
        for field in fields:
            if field in existing_state:
                state_data[field] = existing_state[field]
                copied = True
        if copied:
            current_overrides.append(key)
            current_override_set.add(key)


def persist_runtime_state(
    path: str,
    state: BotState,
    settings: BotSettings,
    *,
    preserve_spot_range: bool = True,
) -> None:
    state_data = {
        "interval_seconds": state.interval_seconds,
        "night_mode_enabled": state.night_mode_enabled,
        "is_running": state.is_running,
        "pnl_alert_low": state.pnl_alert_low,
        "pnl_alert_high": state.pnl_alert_high,
        "max_spot_balance": state.max_spot_balance,
        "min_spot_balance": state.min_spot_balance,
        "init_capital": state.init_capital,
        "outage_filter": state.outage_street_filter,
        "night_mode_window": list(state.night_mode_window),
        "power_outages": state.power_outages,
        "last_outage_check": state.last_outage_check,
        "last_lunar_alert_date": state.last_lunar_alert_date,
        "last_spot_report_date": state.last_spot_report_date,
        "pinned_daily_message_id": state.pinned_daily_message_id,
        "freqtrade_ports": state.freqtrade_ports,
        "freqtrade_alert_cooldown_seconds": state.freqtrade_alert_cooldown_seconds,
        "pre_open_position_interval_seconds": state.pre_open_position_interval_seconds,
        "futures_position_ranges": state.futures_position_ranges,
        "closed_position_ranges": state.closed_position_ranges,
        "runtime_config_overrides": state.runtime_config_overrides,
        "runtime_config_overrides_version": RUNTIME_CONFIG_OVERRIDES_VERSION,
    }
    if preserve_spot_range:
        _preserve_existing_spot_range(path, state_data)
        state.max_spot_balance = _safe_persisted_float(state_data.get("max_spot_balance"), state.max_spot_balance)
        state.min_spot_balance = _safe_persisted_float(state_data.get("min_spot_balance"), state.min_spot_balance)
    _preserve_existing_runtime_overrides(path, state_data)
    payload = {
        "state": state_data,
    }

    tmp_path = f"{path}.tmp"
    backup_path = state_backup_path(path)
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        if os.path.exists(path):
            shutil.copy2(path, backup_path)
        os.replace(tmp_path, path)
    except OSError as exc:
        print(f"Could not persist state: {exc}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
