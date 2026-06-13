import hashlib
import hmac
import time
from typing import Union

import requests

from .constants import IQAIR_API_URL
from .models import EnvConfig


def get_futures_pnl(session: requests.Session, config: EnvConfig) -> Union[float, str]:
    base_url = "https://fapi.binance.com"
    endpoint = "/fapi/v2/account"
    timestamp = int(time.time() * 1000)
    query_string = f"timestamp={timestamp}"
    signature = hmac.new(config.api_secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()
    url = f"{base_url}{endpoint}?{query_string}&signature={signature}"
    headers = {"X-MBX-APIKEY": config.api_key}

    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        positions = data.get("positions", [])
        total_unrealized_pnl = sum(float(position.get("unrealizedProfit", 0.0)) for position in positions)
        return round(total_unrealized_pnl, 2)
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
