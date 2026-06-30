from typing import Optional, Union

from .models import BotState, EnvConfig
from .time_utils import get_lunar_date_string, get_uptime


def get_pnl_icon(val: float) -> str:
    if val > 0: return "🟢"
    if val < 0: return "🔴"
    return "⚪"


def format_closed_trade_line(trade: dict, *, bullet: str = "•") -> str:
    return format_closed_trade_lines(trade, bullet=bullet)[0]


def format_futures_account_balance_lines(pnl: dict, *, bullet: str = "•") -> list:
    fields = [
        ("Wallet Balance", "wallet_balance"),
        ("Available Balance", "available_balance"),
        ("Margin Balance", "margin_balance"),
    ]
    lines = []
    for label, key in fields:
        if key not in pnl:
            continue
        try:
            value = float(pnl.get(key, 0.0))
        except (TypeError, ValueError):
            continue
        lines.append(f"{bullet} {label}: `{value:,.2f} USDT`")
    return lines


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


def _trim_decimal(text: str) -> str:
    if "." not in text:
        return text
    return text.rstrip("0").rstrip(".")


def _format_price(value: float) -> str:
    if value >= 1:
        return _trim_decimal(f"{value:,.4f}")
    if value >= 0.01:
        return _trim_decimal(f"{value:,.6f}")
    return _trim_decimal(f"{value:,.8f}")


def format_spot_current_asset_lines(assets: list, *, indent: str = "  ▫️", max_items: int = 5) -> list:
    lines = []
    for item in (assets or [])[:max_items]:
        asset = item.get("asset", "UNKNOWN")
        price = _safe_float(item.get("price"))
        value = _safe_float(item.get("usdt_value"))
        if asset == "USDT":
            lines.append(f"{indent} `{asset}`: `{value:,.2f} USDT`")
        else:
            lines.append(f"{indent} `{asset}`: `{value:,.2f} USDT` @ `{_format_price(price)}`")
    if assets and len(assets) > max_items:
        lines.append(f"{indent} ... and {len(assets) - max_items} more assets")
    return lines


def format_spot_snapshot_price_lines(assets: list, *, indent: str = "  ▫️", max_items: int = 5) -> list:
    lines = []
    for item in (assets or [])[:max_items]:
        asset = item.get("asset", "UNKNOWN")
        price = _safe_float(item.get("price"))
        value = _safe_float(item.get("usdt_value"))
        if asset == "USDT":
            lines.append(f"{indent} `{asset}`: `{value:,.2f} USDT`")
        else:
            lines.append(f"{indent} `{asset}` @ `{_format_price(price)}`")
    if assets and len(assets) > max_items:
        lines.append(f"{indent} ... and {len(assets) - max_items} more assets")
    return lines


def spot_percent_text(total: float, init_capital: Optional[float]) -> str:
    if not init_capital:
        return ""
    pct = (total - init_capital) / init_capital * 100
    return f" ({pct:+.2f}%)"


def format_spot_min_max_lines(
    label: str,
    total: float,
    assets: list,
    init_capital: Optional[float],
    *,
    asset_indent: str = "  ▫️",
) -> list:
    if total <= 0:
        return []
    lines = [f"• {label}: `{total:,.2f} USDT`{spot_percent_text(total, init_capital)}"]
    lines.extend(format_spot_snapshot_price_lines(assets, indent=asset_indent))
    return lines


def format_spot_sections(
    state: BotState,
    spot_balance: dict,
    *,
    max_current_items: Optional[int] = None,
    current_asset_indent: str = "  ▫️",
    range_asset_indent: str = "  ▫️",
) -> list:
    lines = []
    breakdown = spot_balance.get("breakdown", [])
    if max_current_items is not None:
        breakdown = breakdown[:max_current_items]

    if breakdown:
        lines.append("• Current:")
        lines.extend(format_spot_current_asset_lines(breakdown, indent=current_asset_indent))

    lines.extend(
        format_spot_min_max_lines(
            "Min",
            state.min_spot_balance,
            state.min_spot_assets,
            state.init_capital,
            asset_indent=range_asset_indent,
        )
    )
    lines.extend(
        format_spot_min_max_lines(
            "Max",
            state.max_spot_balance,
            state.max_spot_assets,
            state.init_capital,
            asset_indent=range_asset_indent,
        )
    )
    return lines


def compose_status_message(
    state: BotState,
    config: EnvConfig,
    status_info: Optional[str],
    current_pnl: Union[float, str],
    *,
    spot_balance: Optional[Union[float, str]] = None,
    host_public_ip: Optional[str] = None,
) -> str:
    lines = [
        "🧭 Status:",
        f"• Running: `{state.is_running}`",
        f"• Interval: `{state.interval_seconds / 60:.1f}m`",
        f"• Night mode: `{state.night_mode_enabled}`",
        f"• Uptime: `{get_uptime(state)}`",
        f"• Lunar: `{get_lunar_date_string(config.timezone)}`",
    ]
    effective_host_ip = host_public_ip or state.host_public_ip
    if effective_host_ip:
        lines.append(f"• Host IP: `{effective_host_ip}`")

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
            lines.append(f"• Total: `{total:,.2f} USDT`{pnl_perc_line}")
            lines.extend(
                format_spot_sections(
                    state,
                    spot_balance,
                    max_current_items=5,
                    current_asset_indent="  ▫️",
                    range_asset_indent="  ▫️",
                )
            )
        elif isinstance(spot_balance, (int, float)):
            total = float(spot_balance)
            pnl_perc_line = ""
            if state.init_capital:
                pnl_perc = (total - state.init_capital) / state.init_capital * 100
                pnl_perc_line = f" ({pnl_perc:+.2f}%)"
            lines.append(f"• Total: `{total:,.2f} USDT`{pnl_perc_line}")
            lines.extend(
                format_spot_sections(
                    state,
                    {"breakdown": []},
                    current_asset_indent="  ▫️",
                    range_asset_indent="  ▫️",
                )
            )
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
        lines.extend(format_futures_account_balance_lines(current_pnl))
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
