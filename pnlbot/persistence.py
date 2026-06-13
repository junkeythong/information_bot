import json
import os
import shutil
from typing import Optional

from .config import is_valid_night_mode_window, parse_bool_value, parse_outage_filter_value
from .constants import INTERVAL_MAX_SECONDS, INTERVAL_MIN_SECONDS, STATE_BACKUP_SUFFIX
from .models import BotSettings, BotState


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

    interval_seconds = _safe_int(state_data.get("interval_seconds"), state.interval_seconds)
    if INTERVAL_MIN_SECONDS <= interval_seconds <= INTERVAL_MAX_SECONDS:
        state.interval_seconds = interval_seconds

    state.night_mode_enabled = _safe_bool(state_data.get("night_mode_enabled"), state.night_mode_enabled)
    state.is_running = _safe_bool(state_data.get("is_running"), state.is_running)

    pnl_alert_low = _safe_int(state_data.get("pnl_alert_low"), state.pnl_alert_low)
    pnl_alert_high = _safe_int(state_data.get("pnl_alert_high"), state.pnl_alert_high)
    if pnl_alert_low < pnl_alert_high:
        state.pnl_alert_low = pnl_alert_low
        state.pnl_alert_high = pnl_alert_high

    state.max_pnl = _safe_float(state_data.get("max_pnl"), state.max_pnl)
    state.min_pnl = _safe_float(state_data.get("min_pnl"), state.min_pnl)
    state.max_spot_balance = _safe_float(state_data.get("max_spot_balance"), state.max_spot_balance)
    state.min_spot_balance = _safe_float(state_data.get("min_spot_balance"), state.min_spot_balance)
    init_capital = _safe_float(state_data.get("init_capital"), state.init_capital)
    state.init_capital = init_capital if init_capital and init_capital > 0 else None
    if "outage_filter" in state_data:
        state.outage_street_filter = parse_outage_filter_value(state_data.get("outage_filter"))
    state.last_lunar_alert_date = state_data.get("last_lunar_alert_date", state.last_lunar_alert_date)
    state.pinned_daily_message_id = state_data.get("pinned_daily_message_id", state.pinned_daily_message_id)

    night_window = state_data.get("night_mode_window")
    if isinstance(night_window, (list, tuple)) and len(night_window) == 2:
        try:
            start_hour = int(night_window[0])
            end_hour = int(night_window[1])
            if is_valid_night_mode_window(start_hour, end_hour):
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
        "outage_filter": state.outage_street_filter,
        "night_mode_window": list(state.night_mode_window),
        "power_outages": state.power_outages,
        "last_outage_check": state.last_outage_check,
        "last_lunar_alert_date": state.last_lunar_alert_date,
        "pinned_daily_message_id": state.pinned_daily_message_id,
    }
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
