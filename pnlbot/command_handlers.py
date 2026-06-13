import requests

from . import portfolio
from .config_commands import handle_config_command
from .constants import STATE_FILE_PATH
from .market_data import get_air_quality
from .messages import compose_status_message
from .models import BotSettings, BotState, EnvConfig
from .outages import get_power_outages
from .persistence import persist_runtime_state
from .system_info import get_system_info_text
from .telegram import send_telegram_message


def handle_config_command_message(
    session: requests.Session,
    config: EnvConfig,
    settings: BotSettings,
    state: BotState,
    chat_id: str,
    text: str,
) -> None:
    response = handle_config_command(text, state, settings, config)
    send_telegram_message(
        session,
        config,
        settings,
        response,
        chat_id=chat_id,
        state=state,
        force_send=True,
    )


def handle_status_command(
    session: requests.Session,
    config: EnvConfig,
    settings: BotSettings,
    state: BotState,
    chat_id: str,
    _: str,
) -> None:
    snapshot = portfolio.refresh_portfolio_snapshot(session, config, state)

    send_telegram_message(
        session,
        config,
        settings,
        compose_status_message(state, config, None, snapshot.pnl, spot_balance=snapshot.spot_balance),
        chat_id=chat_id,
        state=state,
        force_send=True,
    )
    if snapshot.state_changed:
        persist_runtime_state(STATE_FILE_PATH, state, settings)


def handle_stop_command(
    session: requests.Session,
    config: EnvConfig,
    settings: BotSettings,
    state: BotState,
    chat_id: str,
    _: str,
) -> None:
    handle_config_command("/config set bot_running off", state, settings, config)
    send_telegram_message(
        session,
        config,
        settings,
        "⛔ Bot paused. No alerts will be sent until `/start` is issued.",
        chat_id=chat_id,
        state=state,
        force_send=True,
    )


def handle_start_command(
    session: requests.Session,
    config: EnvConfig,
    settings: BotSettings,
    state: BotState,
    chat_id: str,
    _: str,
) -> None:
    handle_config_command("/config set bot_running on", state, settings, config)
    send_telegram_message(
        session,
        config,
        settings,
        "▶️ Bot resumed. Alerts are active again.",
        chat_id=chat_id,
        state=state,
        force_send=True,
    )


def handle_futures_command(
    session: requests.Session,
    config: EnvConfig,
    settings: BotSettings,
    state: BotState,
    chat_id: str,
    text: str,
) -> None:
    parts = text.split()
    reset_mode = len(parts) > 1 and parts[1].strip().lower() == "reset"

    pnl, state_changed = portfolio.refresh_futures_pnl(session, config, state, reset_range=reset_mode)
    send_telegram_message(
        session,
        config,
        settings,
        portfolio.format_futures_pnl_summary(state, pnl),
        chat_id=chat_id,
        state=state,
        force_send=True,
    )
    if state_changed:
        persist_runtime_state(STATE_FILE_PATH, state, settings)


def handle_spot_command(
    session: requests.Session,
    config: EnvConfig,
    settings: BotSettings,
    state: BotState,
    chat_id: str,
    text: str,
) -> None:
    parts = text.split()
    reset_mode = len(parts) > 1 and parts[1].strip().lower() == "reset"

    spot_balance, state_changed = portfolio.refresh_spot_balance(session, config, state, reset_range=reset_mode)
    if state_changed:
        persist_runtime_state(STATE_FILE_PATH, state, settings)

    send_telegram_message(
        session,
        config,
        settings,
        portfolio.format_spot_balance_summary(state, spot_balance, include_asset_heading=True),
        chat_id=chat_id,
        state=state,
        force_send=True,
    )


def handle_outage_command(
    session: requests.Session,
    config: EnvConfig,
    settings: BotSettings,
    state: BotState,
    chat_id: str,
    _: str,
) -> None:
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

    send_telegram_message(session, config, settings, message, chat_id=chat_id, state=state, force_send=True)


def handle_sysinfo_command(
    session: requests.Session,
    config: EnvConfig,
    settings: BotSettings,
    state: BotState,
    chat_id: str,
    _: str,
) -> None:
    sysinfo = get_system_info_text(config, settings, show_all=True)
    send_telegram_message(
        session,
        config,
        settings,
        sysinfo or "No system information available.",
        chat_id=chat_id,
        state=state,
        force_send=True,
    )


def handle_aqi_command(
    session: requests.Session,
    config: EnvConfig,
    settings: BotSettings,
    state: BotState,
    chat_id: str,
    _: str,
) -> None:
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
        chat_id=chat_id,
        state=state,
        force_send=True,
    )


def handle_help_command(
    session: requests.Session,
    config: EnvConfig,
    settings: BotSettings,
    state: BotState,
    chat_id: str,
    _: str,
) -> None:
    send_telegram_message(
        session,
        config,
        settings,
        "*ℹ️ Info commands:*\n"
        "• `/status` – Comprehensive snapshot (Futures, Spot, Config)\n"
        "• `/futures` – Quick unrealized PnL check\n"
        "• `/spot` – Quick spot balance check\n"
        "• `/aqi` – Air quality index (IQAir)\n"
        "• `/sysinfo` – System information\n"
        "• `/outage` – View power outage schedule\n"
        "• `/help` – This reference\n"
        "\n*🛠 Configuration:*\n"
        "• `/config show` – View all runtime parameters\n"
        "• `/config set <key> <value>` – Update a parameter\n"
        "• `/start` / `/stop` – Resume or pause alerts\n"
        "• `/spot reset` – reset clear min/max history for spot\n"
        "• `/futures reset` – reset clear min/max history for futures",
        chat_id=chat_id,
        state=state,
        force_send=True,
    )


def handle_unknown_command(
    session: requests.Session,
    config: EnvConfig,
    settings: BotSettings,
    state: BotState,
    chat_id: str,
    text: str,
) -> None:
    if len(text.split()[0]) > 1:
        send_telegram_message(
            session,
            config,
            settings,
            "⚠️ Unsupported command. Type `/help` for the command list.",
            chat_id=chat_id,
            state=state,
            force_send=True,
        )
