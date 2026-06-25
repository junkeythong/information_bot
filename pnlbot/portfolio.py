from dataclasses import dataclass
from typing import Optional, Union

import requests

from .market_data import get_air_quality, get_futures_pnl, get_spot_balance
from .messages import get_pnl_icon
from .models import BotState, EnvConfig
from .state import update_pnl_range, update_spot_balance_range

BalanceResult = Union[dict, float, str]
PnlResult = Union[dict, float, str]


@dataclass
class PortfolioSnapshot:
    pnl: PnlResult
    spot_balance: BalanceResult
    state_changed: bool = False


def get_spot_total(spot_balance: BalanceResult) -> float:
    if isinstance(spot_balance, dict):
        return float(spot_balance.get("total", 0.0))
    if isinstance(spot_balance, (int, float)):
        return float(spot_balance)
    return 0.0


def get_futures_total(pnl: PnlResult) -> float:
    if isinstance(pnl, dict):
        return float(pnl.get("total", 0.0))
    if isinstance(pnl, (int, float)):
        return float(pnl)
    return 0.0


def refresh_futures_pnl(
    session: requests.Session,
    config: EnvConfig,
    state: BotState,
    *,
    reset_range: bool = False,
) -> tuple[PnlResult, bool]:
    pnl = get_futures_pnl(session, config)
    state_changed = False

    pnl_total = get_futures_total(pnl)
    if isinstance(pnl, (dict, int, float)):
        if reset_range:
            state.max_pnl = pnl_total
            state.min_pnl = pnl_total
            state_changed = True

        state_changed = update_pnl_range(state, pnl_total) or state_changed

    return pnl, state_changed


def refresh_spot_balance(
    session: requests.Session,
    config: EnvConfig,
    state: BotState,
    *,
    reset_range: bool = False,
) -> tuple[BalanceResult, bool]:
    spot_balance = get_spot_balance(session, config)
    total = get_spot_total(spot_balance)
    state_changed = False

    if total > 0:
        if reset_range:
            state.max_spot_balance = total
            state.min_spot_balance = total
            state_changed = True

        state_changed = update_spot_balance_range(state, total) or state_changed

    return spot_balance, state_changed


def refresh_portfolio_snapshot(
    session: requests.Session,
    config: EnvConfig,
    state: BotState,
) -> PortfolioSnapshot:
    pnl, pnl_changed = refresh_futures_pnl(session, config, state)
    spot_balance, spot_changed = refresh_spot_balance(session, config, state)
    return PortfolioSnapshot(
        pnl=pnl,
        spot_balance=spot_balance,
        state_changed=pnl_changed or spot_changed,
    )


def format_spot_balance_summary(
    state: BotState,
    spot_balance: BalanceResult,
    *,
    max_breakdown_items: Optional[int] = None,
    include_asset_heading: bool = False,
    hide_empty: bool = False,
) -> str:
    total = get_spot_total(spot_balance)
    if hide_empty and total <= 0:
        if isinstance(spot_balance, str) and not ("0.0" in spot_balance or "0 USDT" in spot_balance):
            return f"⚠️ {spot_balance}"
        return ""

    if isinstance(spot_balance, dict):
        pnl_perc_info = ""
        if state.init_capital:
            pnl_perc = (total - state.init_capital) / state.init_capital * 100
            icon = get_pnl_icon(pnl_perc)
            pnl_perc_info = f" {icon} ({pnl_perc:+.2f}%)"

        lines = [
            f"💰 *Spot:* `{total:,.2f} USDT`{pnl_perc_info}",
            f"📊 *Range:* `[{state.min_spot_balance:,.2f}, {state.max_spot_balance:,.2f}]`",
        ]

        breakdown = spot_balance.get("breakdown", [])
        if max_breakdown_items is not None:
            breakdown = breakdown[:max_breakdown_items]

        if breakdown:
            if include_asset_heading:
                lines.append("\n*Asset Breakdown:*")
            for item in breakdown:
                price_str = f" @ {item['price']:,.4f}" if item["asset"] != "USDT" else ""
                lines.append(f"• `{item['asset']}`: `{item['usdt_value']:,.2f} USDT`{price_str}")

        return "\n".join(lines)

    if isinstance(spot_balance, (int, float)):
        return f"💰 Spot USDT Balance: {float(spot_balance):,.2f} USDT"

    return str(spot_balance)


def format_futures_pnl_summary(
    state: BotState,
    pnl: PnlResult,
    *,
    use_alert_labels: bool = False,
    hide_zero: bool = False,
) -> str:
    if not isinstance(pnl, (dict, int, float)):
        return f"`{pnl}`"

    pnl_total = get_futures_total(pnl)
    if hide_zero and pnl_total == 0.0:
        return ""

    if use_alert_labels and pnl_total <= state.pnl_alert_low:
        lines = [f"Heavy loss: 🔻 `{pnl_total:,.2f} USDT`"]
    elif use_alert_labels and pnl_total >= state.pnl_alert_high:
        lines = [f"High profit: 🟢 `{pnl_total:,.2f} USDT`"]
    else:
        icon = get_pnl_icon(pnl_total)
        lines = [f"💰 *Futures:* `{pnl_total:,.2f} USDT` {icon}"]

    lines.append(f"📊 *Range:* `[{state.min_pnl:,.2f}, {state.max_pnl:,.2f}]`")

    if isinstance(pnl, dict):
        open_positions = pnl.get("open_positions", [])
        closed_trades = pnl.get("closed_trades", [])

        lines.append("")
        lines.append("*Open Positions:*")
        if open_positions:
            for position in open_positions:
                unrealized_pnl = float(position.get("unrealized_pnl", 0.0))
                lines.append(
                    f"• `{position.get('symbol', 'UNKNOWN')}`: `{unrealized_pnl:,.2f} USDT` {get_pnl_icon(unrealized_pnl)}"
                )
        else:
            lines.append("• None")

        lines.append("")
        lines.append("*Latest Closed Positions:*")
        if closed_trades:
            for trade in closed_trades[:3]:
                trade_pnl = float(trade.get("pnl", 0.0))
                exit_reason = trade.get("exit_reason")
                reason_text = f" (exit: `{exit_reason}`)" if exit_reason else ""
                lines.append(
                    f"• `{trade.get('symbol', 'UNKNOWN')}`: `{trade_pnl:,.2f} USDT` {get_pnl_icon(trade_pnl)}{reason_text}"
                )
        else:
            lines.append("• None")

    return "\n".join(lines)


def _format_monitoring_aqi(session: requests.Session, config: EnvConfig) -> str:
    if not config.iqair_api_key:
        return ""

    aqi_data = get_air_quality(session, config)
    if not isinstance(aqi_data, dict):
        return ""

    aqi_us = aqi_data.get("aqi_us", 0)
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

    return f"\n{aqi_emoji} *AQI:* `{aqi_us}` ({aqi_data.get('city', 'N/A')}), Temp: `{aqi_data.get('temperature', 0)}°C`"


def format_monitoring_message(
    session: requests.Session,
    config: EnvConfig,
    state: BotState,
    snapshot: PortfolioSnapshot,
) -> str:
    parts = [
        format_spot_balance_summary(
            state,
            snapshot.spot_balance,
            max_breakdown_items=5,
            hide_empty=True,
        ),
        format_futures_pnl_summary(
            state,
            snapshot.pnl,
            use_alert_labels=True,
            hide_zero=True,
        ),
        _format_monitoring_aqi(session, config),
    ]
    return "\n\n".join(part for part in parts if part).strip()
