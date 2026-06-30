from typing import Callable, Dict

from .config import (
    format_config_value,
    parse_bool_value,
    parse_float_value,
    parse_int_value,
    parse_outage_filter_value,
)
from .constants import STATE_FILE_PATH
from .freqtrade import parse_freqtrade_ports
from .models import BotSettings, BotState, ConfigDefinition, EnvConfig
from .persistence import apply_persisted_configuration, load_persisted_state, persist_runtime_state


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
    "night_mode_enabled",
    "night_mode_start_hour",
    "night_mode_end_hour",
    "init_capital",
    "max_spot_balance",
    "min_spot_balance",
    "outage_filter",
    "freqtrade_ports",
    "freqtrade_alert_cooldown_seconds",
]


CONFIG_DEFINITIONS: Dict[str, ConfigDefinition] = {
    "bot_running": ConfigDefinition(
        description="Toggle bot activity",
        parser=lambda raw, state, settings: parse_bool_value(raw),
        getter=lambda state, settings: state.is_running,
        applier=lambda value, state, settings: setattr(state, "is_running", value),
    ),
    "interval_seconds": ConfigDefinition(
        description="Futures polling interval in seconds",
        parser=lambda raw, state, settings: parse_int_value(raw, minimum=10, maximum=86400),
        getter=lambda state, settings: state.interval_seconds,
        applier=lambda value, state, settings: setattr(state, "interval_seconds", value),
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
        description="Power outage street filter (use none to clear)",
        parser=lambda raw, state, settings: parse_outage_filter_value(raw),
        getter=lambda state, settings: state.outage_street_filter or "None",
        applier=lambda value, state, settings: setattr(state, "outage_street_filter", value),
    ),
    "freqtrade_ports": ConfigDefinition(
        description="Comma-separated localhost Freqtrade API ports",
        parser=lambda raw, state, settings: parse_freqtrade_ports(raw),
        getter=lambda state, settings: ",".join(str(port) for port in state.freqtrade_ports) or "None",
        applier=lambda value, state, settings: setattr(state, "freqtrade_ports", value),
    ),
    "freqtrade_alert_cooldown_seconds": ConfigDefinition(
        description="Freqtrade health alert cooldown in seconds",
        parser=lambda raw, state, settings: parse_int_value(raw, minimum=60, maximum=86400),
        getter=lambda state, settings: state.freqtrade_alert_cooldown_seconds,
        applier=lambda value, state, settings: setattr(state, "freqtrade_alert_cooldown_seconds", value),
    ),
}


def _refresh_runtime_state_from_disk(state: BotState, settings: BotSettings, config: EnvConfig) -> None:
    persisted = load_persisted_state(STATE_FILE_PATH)
    if not persisted:
        return
    apply_persisted_configuration(persisted, state, settings)
    config.outage_street_filter = state.outage_street_filter


def format_config_listing(state: BotState, settings: BotSettings, config: EnvConfig) -> str:
    lines = ["*⚙️ Runtime configuration:*"]
    for key in CONFIG_ORDER:
        definition = CONFIG_DEFINITIONS[key]
        if key == "outage_filter":
            value = config.outage_street_filter or "None"
        else:
            value = definition.getter(state, settings)
        lines.append(f"• `{key}`: `{format_config_value(value)}`")
    return "\n".join(lines)


def handle_config_command(text: str, state: BotState, settings: BotSettings, config: EnvConfig) -> str:
    tokens = text.split(maxsplit=3)

    if len(tokens) == 1 or (len(tokens) >= 2 and tokens[1].lower() == "show"):
        _refresh_runtime_state_from_disk(state, settings, config)
        return format_config_listing(state, settings, config)

    if len(tokens) < 2:
        return "❌ Usage: `/config show|set`"

    action = tokens[1].lower()

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
            if key not in state.runtime_config_overrides:
                state.runtime_config_overrides.append(key)
            if key == "outage_filter":
                config.outage_street_filter = state.outage_street_filter
            persist_runtime_state(
                STATE_FILE_PATH,
                state,
                settings,
                preserve_spot_range=key not in {"max_spot_balance", "min_spot_balance"},
            )
            if isinstance(result, str) and result:
                extra = f"\n{result}"
            else:
                extra = ""
            updated = definition.getter(state, settings)
            return f"✅ `{key}` updated to `{format_config_value(updated)}`{extra}"
        except ValueError as exc:
            return f"❌ {exc}"

    return "❌ Unsupported config command. Use `/config show|set`."
