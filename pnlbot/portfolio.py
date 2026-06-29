import datetime
from dataclasses import dataclass
from typing import Optional, Union

import pytz
import requests

from .market_data import get_futures_pnl, get_spot_balance
from .messages import format_closed_trade_lines, format_open_position_lines, format_spot_range_lines, get_pnl_icon
from .models import BotState, EnvConfig
from .state import update_spot_balance_range

BalanceResult = Union[dict, float, str]
PnlResult = Union[dict, float, str]


@dataclass
class PortfolioSnapshot:
    pnl: PnlResult
    spot_balance: Optional[BalanceResult] = None
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


def _spot_observed_at(config: EnvConfig) -> str:
    now = datetime.datetime.now(pytz.timezone(config.timezone))
    return now.strftime("%Y-%m-%d %H:%M")


def _spot_asset_snapshot(spot_balance: BalanceResult) -> list:
    if not isinstance(spot_balance, dict):
        return []

    snapshot = []
    for item in spot_balance.get("breakdown", []):
        try:
            snapshot.append({
                "asset": item.get("asset", "UNKNOWN"),
                "amount": float(item.get("amount", 0.0)),
                "usdt_value": float(item.get("usdt_value", 0.0)),
                "price": float(item.get("price", 0.0)),
            })
        except (TypeError, ValueError):
            continue
    return snapshot


def _position_range_key(position: dict) -> str:
    symbol = position.get("symbol", "UNKNOWN")
    position_side = position.get("position_side", "BOTH")
    side = position.get("side", "UNKNOWN")
    return f"{symbol}:{position_side}:{side}"


def _position_price(position: dict) -> Optional[float]:
    price = position.get("mark_price") or position.get("entry_price")
    if price is None:
        return None
    return float(price)


def _archive_closed_position_range(state: BotState, key: str, position_range: dict) -> None:
    archived_range = dict(position_range)
    archived_range["key"] = key
    state.closed_position_ranges.insert(0, archived_range)
    state.closed_position_ranges = state.closed_position_ranges[:10]
    print(
        "Archived observed Futures range "
        f"symbol={archived_range.get('symbol')} side={archived_range.get('side')} "
        f"min={archived_range.get('min_pnl')}@{archived_range.get('min_price')} "
        f"max={archived_range.get('max_pnl')}@{archived_range.get('max_price')}",
        flush=True,
    )


def _closed_trade_matches_range(trade: dict, position_range: dict) -> bool:
    if trade.get("symbol") != position_range.get("symbol"):
        return False

    for field in ("position_side", "side"):
        trade_value = trade.get(field)
        range_value = position_range.get(field)
        if range_value and trade_value != range_value:
            return False
    return True


def _attach_closed_position_ranges(state: BotState, pnl: PnlResult) -> bool:
    if not isinstance(pnl, dict):
        return False

    closed_trades = pnl.get("closed_trades") or []
    consumed_indexes = set()
    attached = False
    for trade in closed_trades:
        for index, position_range in enumerate(state.closed_position_ranges):
            if index in consumed_indexes or not _closed_trade_matches_range(trade, position_range):
                continue
            trade["pnl_range"] = dict(position_range)
            consumed_indexes.add(index)
            attached = True
            print(
                "Attached observed Futures range "
                f"symbol={position_range.get('symbol')} side={position_range.get('side')} "
                f"closed_pnl={trade.get('pnl')}",
                flush=True,
            )
            break

    if consumed_indexes:
        state.closed_position_ranges = [
            item for index, item in enumerate(state.closed_position_ranges)
            if index not in consumed_indexes
        ]
    return attached


def apply_position_pnl_ranges(state: BotState, pnl: PnlResult) -> bool:
    if not isinstance(pnl, dict):
        return False

    open_positions = pnl.get("open_positions") or []

    active_keys = set()
    state_changed = False
    for position in open_positions:
        key = _position_range_key(position)
        active_keys.add(key)
        current_pnl = float(position.get("unrealized_pnl", 0.0))
        current_price = _position_price(position)
        position_range = state.futures_position_ranges.get(key)

        if position_range is None:
            position_range = {
                "symbol": position.get("symbol", "UNKNOWN"),
                "position_side": position.get("position_side", "BOTH"),
                "side": position.get("side", "UNKNOWN"),
                "entry_price": position.get("entry_price"),
                "min_pnl": current_pnl,
                "min_price": current_price,
                "max_pnl": current_pnl,
                "max_price": current_price,
            }
            state.futures_position_ranges[key] = position_range
            state_changed = True
        else:
            if current_pnl < float(position_range.get("min_pnl", current_pnl)):
                position_range["min_pnl"] = current_pnl
                position_range["min_price"] = current_price
                state_changed = True
            if current_pnl > float(position_range.get("max_pnl", current_pnl)):
                position_range["max_pnl"] = current_pnl
                position_range["max_price"] = current_price
                state_changed = True

        position["pnl_range"] = dict(position_range)

    stale_keys = set(state.futures_position_ranges) - active_keys
    for key in stale_keys:
        _archive_closed_position_range(state, key, state.futures_position_ranges[key])
        del state.futures_position_ranges[key]
        state_changed = True

    state_changed = _attach_closed_position_ranges(state, pnl) or state_changed
    return state_changed



def refresh_futures_pnl(
    session: requests.Session,
    config: EnvConfig,
    state: BotState,
) -> tuple[PnlResult, bool]:
    pnl = get_futures_pnl(session, config)
    state_changed = False

    state_changed = apply_position_pnl_ranges(state, pnl) or state_changed

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
        asset_snapshot = _spot_asset_snapshot(spot_balance)
        observed_at = _spot_observed_at(config)

        if reset_range:
            state.max_spot_balance = total
            state.min_spot_balance = total
            state.max_spot_assets = list(asset_snapshot)
            state.min_spot_assets = list(asset_snapshot)
            state.max_spot_observed_at = observed_at
            state.min_spot_observed_at = observed_at
            state_changed = True

        state_changed = update_spot_balance_range(
            state,
            total,
            asset_snapshot=asset_snapshot,
            observed_at=observed_at,
        ) or state_changed

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

        lines = [f"💰 *Spot:* `{total:,.2f} USDT`{pnl_perc_info}"]
        if state.min_spot_balance > 0 or state.max_spot_balance > 0:
            lines.extend([
                "📊 *Range:*",
                *format_spot_range_lines(
                    "Min",
                    state.min_spot_balance,
                    state.min_spot_assets,
                    state.min_spot_observed_at,
                ),
                *format_spot_range_lines(
                    "Max",
                    state.max_spot_balance,
                    state.max_spot_assets,
                    state.max_spot_observed_at,
                ),
            ])

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


    if isinstance(pnl, dict):
        open_positions = pnl.get("open_positions", [])
        closed_trades = pnl.get("closed_trades", [])

        lines.append("")
        lines.append("*Open Positions:*")
        if open_positions:
            for position in open_positions:
                lines.extend(format_open_position_lines(position))
        else:
            lines.append("• None")

        lines.append("")
        lines.append("*Latest Closed Positions:*")
        if closed_trades:
            for trade in closed_trades[:3]:
                lines.extend(format_closed_trade_lines(trade))
        else:
            lines.append("• None")

    return "\n".join(lines)



def format_monitoring_message(
    session: requests.Session,
    config: EnvConfig,
    state: BotState,
    snapshot: PortfolioSnapshot,
) -> str:
    parts = []
    if snapshot.spot_balance is not None:
        parts.append(
            format_spot_balance_summary(
                state,
                snapshot.spot_balance,
                max_breakdown_items=5,
                hide_empty=True,
            )
        )
    parts.extend([
        format_futures_pnl_summary(
            state,
            snapshot.pnl,
            use_alert_labels=True,
            hide_zero=True,
        ),
    ])
    return "\n\n".join(part for part in parts if part).strip()
