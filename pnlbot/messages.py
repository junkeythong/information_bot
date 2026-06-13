from typing import Optional, Union

from .models import BotState, EnvConfig
from .time_utils import get_lunar_date_string, get_uptime


def get_pnl_icon(val: float) -> str:
    if val > 0: return "🟢"
    if val < 0: return "🔴"
    return "⚪"


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
                max_min_line = f"• Max: `{state.max_spot_balance:,.2f} USDT` ({max_perc:+.2f}%), Min: `{state.min_spot_balance:,.2f} USDT` ({min_perc:+.2f}%)"
            else:
                max_min_line = f"• Max: `{state.max_spot_balance:,.2f} USDT`, Min: `{state.min_spot_balance:,.2f} USDT`"

            lines.append(f"• Total: `{total:,.2f} USDT`{pnl_perc_line}")
            lines.append(max_min_line)
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
                max_min_line = f"• Max: `{state.max_spot_balance:,.2f} USDT` ({max_perc:+.2f}%), Min: `{state.min_spot_balance:,.2f} USDT` ({min_perc:+.2f}%)"
            else:
                max_min_line = f"• Max: `{state.max_spot_balance:,.2f} USDT`, Min: `{state.min_spot_balance:,.2f} USDT`"

            lines.append(f"• Total: `{total:,.2f} USDT`{pnl_perc_line}")
            lines.append(max_min_line)
        else:
            lines.append(f"• {spot_balance}")

    lines.extend([
        "",
        "💰 *Futures:*",
    ])
    if isinstance(current_pnl, (int, float)):
        icon = get_pnl_icon(current_pnl)
        lines.append(f"• Current PnL: `{current_pnl:,.2f} USDT` {icon}")
    else:
        lines.append(f"• Current PnL: `{current_pnl}`")
    lines.extend([
        f"• Max PnL: `{state.max_pnl:,.2f} USDT`, Min: `{state.min_pnl:,.2f} USDT`",
    ])

    return "\n".join(lines)
