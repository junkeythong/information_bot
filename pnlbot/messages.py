from typing import Optional, Union

from .models import BotState, EnvConfig
from .time_utils import get_lunar_date_string, get_uptime


def get_pnl_icon(val: float) -> str:
    if val > 0: return "🟢"
    if val < 0: return "🔴"
    return "⚪"


def format_closed_trade_line(trade: dict, *, bullet: str = "•") -> str:
    return format_closed_trade_lines(trade, bullet=bullet)[0]


def format_closed_trade_lines(trade: dict, *, bullet: str = "•", detail_bullet: str = "  ▫️") -> list:
    trade_pnl = float(trade.get("pnl", 0.0))
    exit_reason = trade.get("exit_reason")
    reason_text = f" (exit: `{exit_reason}`)" if exit_reason else ""
    lines = [
        f"{bullet} `{trade.get('symbol', 'UNKNOWN')}`: "
        f"`{trade_pnl:,.2f} USDT` {get_pnl_icon(trade_pnl)}{reason_text}"
    ]

    pnl_range = trade.get("pnl_range")
    if isinstance(pnl_range, dict):
        min_pnl = float(pnl_range.get("min_pnl", 0.0))
        max_pnl = float(pnl_range.get("max_pnl", 0.0))
        lines.append(
            f"{detail_bullet} Observed Open Min: `{min_pnl:,.2f} USDT` {get_pnl_icon(min_pnl)} "
            f"@ `{_format_position_price(pnl_range.get('min_price'))}`"
        )
        lines.append(
            f"{detail_bullet} Observed Open Max: `{max_pnl:,.2f} USDT` {get_pnl_icon(max_pnl)} "
            f"@ `{_format_position_price(pnl_range.get('max_price'))}`"
        )

    return lines


def _format_position_price(price: object) -> str:
    if price is None:
        return "n/a"
    return f"{float(price):,.8f}".rstrip("0").rstrip(".")


def format_open_position_lines(position: dict, *, bullet: str = "•", detail_bullet: str = "  ▫️") -> list:
    unrealized_pnl = float(position.get("unrealized_pnl", 0.0))
    mark_price = position.get("mark_price")
    price_text = f" @ `{_format_position_price(mark_price)}`" if mark_price is not None else ""
    side = position.get("side") or position.get("position_side") or ""
    side_text = f" {side}" if side else ""
    lines = [
        f"{bullet} `{position.get('symbol', 'UNKNOWN')}`{side_text}: "
        f"`{unrealized_pnl:,.2f} USDT` {get_pnl_icon(unrealized_pnl)}{price_text}"
    ]

    pnl_range = position.get("pnl_range")
    if isinstance(pnl_range, dict):
        min_pnl = float(pnl_range.get("min_pnl", 0.0))
        max_pnl = float(pnl_range.get("max_pnl", 0.0))
        lines.append(
            f"{detail_bullet} Observed Min: `{min_pnl:,.2f} USDT` {get_pnl_icon(min_pnl)} "
            f"@ `{_format_position_price(pnl_range.get('min_price'))}`"
        )
        lines.append(
            f"{detail_bullet} Observed Max: `{max_pnl:,.2f} USDT` {get_pnl_icon(max_pnl)} "
            f"@ `{_format_position_price(pnl_range.get('max_price'))}`"
        )

    return lines


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_price(value: float) -> str:
    if value >= 1:
        return f"{value:,.4f}"
    if value >= 0.01:
        return f"{value:,.6f}"
    return f"{value:,.8f}"


def format_spot_asset_snapshot_lines(assets: list, *, indent: str = "  ▫️", max_items: int = 5) -> list:
    lines = []
    for item in (assets or [])[:max_items]:
        asset = item.get("asset", "UNKNOWN")
        amount = _safe_float(item.get("amount"))
        price = _safe_float(item.get("price"))
        value = _safe_float(item.get("usdt_value"))
        if asset == "USDT":
            lines.append(f"{indent} `{asset}`: `{value:,.2f} USDT`")
        else:
            lines.append(
                f"{indent} `{asset}`: `{amount:,.8f}` @ `{_format_price(price)}` = `{value:,.2f} USDT`"
            )
    if assets and len(assets) > max_items:
        lines.append(f"{indent} ... and {len(assets) - max_items} more assets")
    return lines


def format_spot_range_lines(
    label: str,
    total: float,
    assets: list,
    observed_at: Optional[str],
    *,
    asset_indent: str = "  ▫️",
) -> list:
    header = f"• {label}: `{total:,.2f} USDT`"
    if observed_at:
        header += f" ({observed_at})"
    return [header] + format_spot_asset_snapshot_lines(assets, indent=asset_indent)


def compose_status_message(
    state: BotState,
    config: EnvConfig,
    status_info: Optional[str],
    current_pnl: Union[float, str],
    *,
    spot_balance: Optional[Union[float, str]] = None,
) -> str:
    lines = [
        "🧭 Status:",
        f"• Running: `{state.is_running}`",
        f"• Interval: `{state.interval_seconds / 60:.1f}m`",
        f"• Night mode: `{state.night_mode_enabled}`",
        f"• Uptime: `{get_uptime(state)}`",
        f"• Lunar: `{get_lunar_date_string(config.timezone)}`",
    ]

    lines.extend([
        "",
        "💰 *Spot:*",
    ])
    if state.init_capital:
        lines.append(f"• Init Capital: `{state.init_capital:,.2f} USDT`")
    if spot_balance is not None:
        if isinstance(spot_balance, dict):
            total = spot_balance.get("total", 0.0)
            pnl_perc_line = ""
            if state.init_capital:
                pnl_perc = (total - state.init_capital) / state.init_capital * 100
                icon = get_pnl_icon(pnl_perc)
                pnl_perc_line = f" {icon} ({pnl_perc:+.2f}%)"
                max_perc = (state.max_spot_balance - state.init_capital) / state.init_capital * 100
                min_perc = (state.min_spot_balance - state.init_capital) / state.init_capital * 100
            lines.append(f"• Total: `{total:,.2f} USDT`{pnl_perc_line}")
            if state.min_spot_balance > 0 or state.max_spot_balance > 0:
                lines.append("• Range:")
                min_lines = format_spot_range_lines(
                    "Min",
                    state.min_spot_balance,
                    state.min_spot_assets,
                    state.min_spot_observed_at,
                    asset_indent="    ▫️",
                )
                max_lines = format_spot_range_lines(
                    "Max",
                    state.max_spot_balance,
                    state.max_spot_assets,
                    state.max_spot_observed_at,
                    asset_indent="    ▫️",
                )
                if state.init_capital:
                    min_lines[0] += f" ({min_perc:+.2f}%)"
                    max_lines[0] += f" ({max_perc:+.2f}%)"
                lines.extend(f"  {line}" for line in min_lines)
                lines.extend(f"  {line}" for line in max_lines)
            breakdown = spot_balance.get("breakdown", [])
            for item in breakdown[:5]:
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
                max_perc = (state.max_spot_balance - state.init_capital) / state.init_capital * 100
                min_perc = (state.min_spot_balance - state.init_capital) / state.init_capital * 100
            lines.append(f"• Total: `{total:,.2f} USDT`{pnl_perc_line}")
            if state.min_spot_balance > 0 or state.max_spot_balance > 0:
                lines.append("• Range:")
                min_lines = format_spot_range_lines(
                    "Min",
                    state.min_spot_balance,
                    state.min_spot_assets,
                    state.min_spot_observed_at,
                    asset_indent="    ▫️",
                )
                max_lines = format_spot_range_lines(
                    "Max",
                    state.max_spot_balance,
                    state.max_spot_assets,
                    state.max_spot_observed_at,
                    asset_indent="    ▫️",
                )
                if state.init_capital:
                    min_lines[0] += f" ({min_perc:+.2f}%)"
                    max_lines[0] += f" ({max_perc:+.2f}%)"
                lines.extend(f"  {line}" for line in min_lines)
                lines.extend(f"  {line}" for line in max_lines)
        else:
            lines.append(f"• {spot_balance}")

    lines.extend([
        "",
        "💰 *Futures:*",
    ])
    if isinstance(current_pnl, dict):
        total_pnl = float(current_pnl.get("total", 0.0))
        icon = get_pnl_icon(total_pnl)
        lines.append(f"• Current PnL: `{total_pnl:,.2f} USDT` {icon}")
    else:
        if isinstance(current_pnl, (int, float)):
            icon = get_pnl_icon(current_pnl)
            lines.append(f"• Current PnL: `{current_pnl:,.2f} USDT` {icon}")
        else:
            lines.append(f"• Current PnL: `{current_pnl}`")
    if isinstance(current_pnl, dict):
        open_positions = current_pnl.get("open_positions", [])
        closed_trades = current_pnl.get("closed_trades", [])

        lines.append("• Open Positions:")
        if open_positions:
            for position in open_positions:
                lines.extend(format_open_position_lines(position, bullet="  ▫️", detail_bullet="    ▪️"))
        else:
            lines.append("  ▫️ None")

        lines.append("• Latest Closed Positions:")
        if closed_trades:
            for trade in closed_trades[:3]:
                lines.extend(format_closed_trade_lines(trade, bullet="  ▫️", detail_bullet="    ▪️"))
        else:
            lines.append("  ▫️ None")

    return "\n".join(lines)
