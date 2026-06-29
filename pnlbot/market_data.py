import hashlib
import hmac
import time
from typing import Union

import requests

from .constants import IQAIR_API_URL
from .models import EnvConfig


def _signed_futures_url(endpoint: str, config: EnvConfig, query_params: str) -> str:
    base_url = "https://fapi.binance.com"
    signature = hmac.new(
        config.api_secret.encode("utf-8"),
        query_params.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{base_url}{endpoint}?{query_params}&signature={signature}"


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _recent_income_symbols(income_items: list, max_symbols: int = 10) -> list:
    symbols = []
    for item in sorted(income_items, key=lambda item: _as_int(item.get("time")), reverse=True):
        symbol = item.get("symbol")
        if symbol and symbol not in symbols:
            symbols.append(symbol)
        if len(symbols) >= max_symbols:
            break
    return symbols


def _closed_position_direction(position_side: str, order_side: str) -> str:
    if position_side in {"LONG", "SHORT"}:
        return position_side
    if order_side == "SELL":
        return "LONG"
    if order_side == "BUY":
        return "SHORT"
    return "UNKNOWN"


def _aggregate_closed_positions_from_trades(
    trades: list,
    limit: int = 3,
    *,
    burst_window_ms: int = 5000,
) -> list:
    realized_trades = []
    for trade in trades:
        realized_pnl = _as_float(trade.get("realizedPnl"))
        if realized_pnl == 0.0:
            continue

        position_side = trade.get("positionSide", "BOTH")
        order_side = trade.get("side", "UNKNOWN")
        realized_trades.append(
            {
                "symbol": trade.get("symbol", "UNKNOWN"),
                "position_side": position_side,
                "side": _closed_position_direction(position_side, order_side),
                "order_side": order_side,
                "pnl": realized_pnl,
                "time": _as_int(trade.get("time")),
            }
        )

    realized_trades.sort(key=lambda item: item["time"], reverse=True)

    closed_positions = []
    active_by_key = {}
    for trade in realized_trades:
        trade_key = (trade["symbol"], trade["position_side"], trade["side"])
        active_group = active_by_key.get(trade_key)
        if active_group is not None and active_group["oldest_time"] - trade["time"] <= burst_window_ms:
            active_group["pnl"] += trade["pnl"]
            active_group["oldest_time"] = min(active_group["oldest_time"], trade["time"])
            continue

        active_group = {
            "key": trade_key,
            "symbol": trade["symbol"],
            "position_side": trade["position_side"],
            "side": trade["side"],
            "pnl": trade["pnl"],
            "time": trade["time"],
            "oldest_time": trade["time"],
        }
        active_by_key[trade_key] = active_group
        closed_positions.append(active_group)

    return [
        {
            "symbol": item["symbol"],
            "position_side": item["position_side"],
            "side": item["side"],
            "pnl": round(item["pnl"], 2),
            "time": item["time"],
        }
        for item in closed_positions[:limit]
    ]


def _get_recent_closed_positions(
    session: requests.Session,
    config: EnvConfig,
    headers: dict,
    symbols: list,
    timestamp: int,
    *,
    limit: int = 3,
) -> list:
    trades = []
    for symbol in symbols:
        trades_query = f"symbol={symbol}&limit=100&timestamp={timestamp}"
        trades_url = _signed_futures_url("/fapi/v1/userTrades", config, trades_query)
        trades_response = session.get(trades_url, headers=headers, timeout=10)
        trades_response.raise_for_status()
        trades.extend(trades_response.json())

    return _aggregate_closed_positions_from_trades(trades, limit=limit)


def _position_mark_price(position: dict, amount: float, unrealized_pnl: float) -> float:
    mark_price = _as_float(position.get("markPrice"))
    if mark_price > 0:
        return mark_price

    entry_price = _as_float(position.get("entryPrice"))
    if entry_price > 0 and amount != 0.0:
        return entry_price + unrealized_pnl / amount

    return 0.0


def _position_direction(position_side: str, amount: float) -> str:
    if position_side in {"LONG", "SHORT"}:
        return position_side
    return "LONG" if amount > 0 else "SHORT"


def get_futures_pnl(session: requests.Session, config: EnvConfig) -> Union[dict, float, str]:
    timestamp = int(time.time() * 1000)
    query_string = f"timestamp={timestamp}"
    url = _signed_futures_url("/fapi/v2/account", config, query_string)
    headers = {"X-MBX-APIKEY": config.api_key}

    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        positions = data.get("positions", [])
        total_unrealized_pnl = sum(_as_float(position.get("unrealizedProfit")) for position in positions)

        open_positions = []
        for position in positions:
            position_amount = _as_float(position.get("positionAmt"))
            if position_amount == 0.0:
                continue

            unrealized_pnl = _as_float(position.get("unrealizedProfit"))
            entry_price = _as_float(position.get("entryPrice"))
            mark_price = _position_mark_price(position, position_amount, unrealized_pnl)
            position_side = position.get("positionSide", "BOTH")
            open_positions.append(
                {
                    "symbol": position.get("symbol", "UNKNOWN"),
                    "position_side": position_side,
                    "side": _position_direction(position_side, position_amount),
                    "amount": round(position_amount, 8),
                    "entry_price": round(entry_price, 8) if entry_price > 0 else None,
                    "mark_price": round(mark_price, 8) if mark_price > 0 else None,
                    "unrealized_pnl": round(unrealized_pnl, 2),
                }
            )

        income_query = f"incomeType=REALIZED_PNL&limit=10&timestamp={timestamp}"
        income_url = _signed_futures_url("/fapi/v1/income", config, income_query)
        income_response = session.get(income_url, headers=headers, timeout=10)
        income_response.raise_for_status()
        recent_symbols = _recent_income_symbols(income_response.json())
        closed_trades = _get_recent_closed_positions(session, config, headers, recent_symbols, timestamp)

        return {
            "total": round(total_unrealized_pnl, 2),
            "open_positions": open_positions,
            "closed_trades": closed_trades,
        }
    except Exception as exc:
        return f"PnL fetch error: {exc}"


def get_spot_balance(session: requests.Session, config: EnvConfig) -> Union[dict, str]:
    base_url = "https://api.binance.com"
    endpoint = "/api/v3/account"
    timestamp = int(time.time() * 1000)
    query_string = f"timestamp={timestamp}"
    signature = hmac.new(config.api_secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()
    url = f"{base_url}{endpoint}?{query_string}&signature={signature}"
    headers = {"X-MBX-APIKEY": config.api_key}

    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        balances = data.get("balances", [])

        price_url = "https://api.binance.com/api/v3/ticker/price"
        price_response = session.get(price_url, timeout=10)
        price_response.raise_for_status()
        prices = {item["symbol"]: float(item["price"]) for item in price_response.json()}

        total_usdt = 0.0
        breakdown = []

        for balance in balances:
            asset = balance.get("asset")
            free = float(balance.get("free", 0.0))
            locked = float(balance.get("locked", 0.0))
            amount = free + locked

            if amount <= 0:
                continue

            asset_usdt_value = 0.0
            if asset == "USDT":
                asset_usdt_value = amount
            else:
                symbol = f"{asset}USDT"
                if symbol in prices:
                    asset_usdt_value = amount * prices[symbol]
                else:
                    continue

            if asset_usdt_value < 0.01:
                continue

            total_usdt += asset_usdt_value
            breakdown.append({
                "asset": asset,
                "amount": amount,
                "usdt_value": asset_usdt_value,
                "price": prices.get(symbol, 1.0) if asset != "USDT" else 1.0
            })

        breakdown.sort(key=lambda x: x["usdt_value"], reverse=True)

        return {
            "total": round(total_usdt, 2),
            "breakdown": breakdown,
            "btc_price": prices.get("BTCUSDT", 0.0)
        }
    except Exception as exc:
        return f"Spot balance fetch error: {exc}"


def get_air_quality(session: requests.Session, config: EnvConfig) -> Union[dict, str]:
    if not config.iqair_api_key:
        return "IQAir API key not configured"

    try:
        params = {
            "lat": config.iqair_latitude,
            "lon": config.iqair_longitude,
            "key": config.iqair_api_key,
        }
        response = session.get(IQAIR_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "success":
            return f"IQAir API error: {data.get('message', 'Unknown error')}"

        result_data = data.get("data", {})
        current = result_data.get("current", {})
        pollution = current.get("pollution", {})
        weather = current.get("weather", {})

        return {
            "city": result_data.get("city", "Unknown"),
            "country": result_data.get("country", "Unknown"),
            "aqi_us": pollution.get("aqius", 0),
            "temperature": weather.get("tp", 0),
            "humidity": weather.get("hu", 0),
        }
    except Exception as exc:
        return f"Air quality fetch error: {exc}"
