import unittest
from unittest.mock import patch

from pnlbot.market_data import get_futures_pnl
from pnlbot.models import EnvConfig


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.urls = []

    def get(self, url, headers=None, timeout=None):
        self.urls.append(url)
        return FakeResponse(self.responses.pop(0))


class FuturesMarketDataTests(unittest.TestCase):
    def test_get_futures_pnl_returns_closed_positions_from_account_trades(self):
        session = FakeSession(
            [
                {
                    "positions": [
                        {
                            "symbol": "BTCUSDT",
                            "positionAmt": "0.01",
                            "unrealizedProfit": "12.345",
                            "entryPrice": "62000",
                            "markPrice": "63234.5",
                            "positionSide": "BOTH",
                        },
                        {
                            "symbol": "ETHUSDT",
                            "positionAmt": "0",
                            "unrealizedProfit": "0.0",
                        },
                        {
                            "symbol": "SOLUSDT",
                            "positionAmt": "-2",
                            "unrealizedProfit": "-1.235",
                            "entryPrice": "150.0",
                            "positionSide": "BOTH",
                        },
                    ]
                },
                [
                    {"symbol": "AERGOUSDT", "income": "1.03", "time": 13000},
                    {"symbol": "AERGOUSDT", "income": "0.94", "time": 7000},
                    {"symbol": "AERGOUSDT", "income": "0.76", "time": 1000},
                ],
                [
                    {"symbol": "AERGOUSDT", "orderId": 103, "side": "SELL", "realizedPnl": "-0.12", "time": 13000, "positionSide": "BOTH"},
                    {"symbol": "AERGOUSDT", "orderId": 102, "side": "SELL", "realizedPnl": "1.86", "time": 7000, "positionSide": "BOTH"},
                    {"symbol": "AERGOUSDT", "orderId": 101, "side": "SELL", "realizedPnl": "0.77", "time": 1001, "positionSide": "BOTH"},
                    {"symbol": "AERGOUSDT", "orderId": 101, "side": "SELL", "realizedPnl": "1.00", "time": 1000, "positionSide": "BOTH"},
                    {"symbol": "AERGOUSDT", "orderId": 100, "side": "SELL", "realizedPnl": "0.00", "time": 900, "positionSide": "BOTH"},
                ],
            ]
        )
        config = EnvConfig("key", "secret", "token", "chat")

        with patch("pnlbot.market_data.time.time", return_value=1000):
            result = get_futures_pnl(session, config)

        self.assertEqual(result["total"], 11.11)
        self.assertEqual(
            result["open_positions"],
            [
                {
                    "symbol": "BTCUSDT",
                    "position_side": "BOTH",
                    "side": "LONG",
                    "amount": 0.01,
                    "entry_price": 62000.0,
                    "mark_price": 63234.5,
                    "unrealized_pnl": 12.35,
                },
                {
                    "symbol": "SOLUSDT",
                    "position_side": "BOTH",
                    "side": "SHORT",
                    "amount": -2.0,
                    "entry_price": 150.0,
                    "mark_price": 150.6175,
                    "unrealized_pnl": -1.24,
                },
            ],
        )
        self.assertEqual(
            result["closed_trades"],
            [
                {"symbol": "AERGOUSDT", "position_side": "BOTH", "side": "LONG", "pnl": -0.12, "time": 13000},
                {"symbol": "AERGOUSDT", "position_side": "BOTH", "side": "LONG", "pnl": 1.86, "time": 7000},
                {"symbol": "AERGOUSDT", "position_side": "BOTH", "side": "LONG", "pnl": 1.77, "time": 1001},
            ],
        )
        self.assertIn("/fapi/v2/account", session.urls[0])
        self.assertIn("/fapi/v1/income", session.urls[1])
        self.assertIn("incomeType=REALIZED_PNL", session.urls[1])
        self.assertIn("/fapi/v1/userTrades", session.urls[2])
        self.assertIn("symbol=AERGOUSDT", session.urls[2])

    def test_get_futures_pnl_groups_close_bursts_across_orders(self):
        session = FakeSession(
            [
                {"positions": []},
                [
                    {"symbol": "AERGOUSDT", "income": "0.70", "time": 10020},
                    {"symbol": "AERGOUSDT", "income": "0.60", "time": 10010},
                    {"symbol": "AERGOUSDT", "income": "0.50", "time": 10000},
                ],
                [
                    {"symbol": "AERGOUSDT", "orderId": 203, "side": "SELL", "positionSide": "BOTH", "realizedPnl": "0.70", "time": 10020},
                    {"symbol": "AERGOUSDT", "orderId": 202, "side": "SELL", "positionSide": "BOTH", "realizedPnl": "0.60", "time": 10010},
                    {"symbol": "AERGOUSDT", "orderId": 201, "side": "SELL", "positionSide": "BOTH", "realizedPnl": "0.50", "time": 10000},
                    {"symbol": "AERGOUSDT", "orderId": 200, "side": "SELL", "positionSide": "BOTH", "realizedPnl": "1.25", "time": 4000},
                ],
            ]
        )
        config = EnvConfig("key", "secret", "token", "chat")

        with patch("pnlbot.market_data.time.time", return_value=20):
            result = get_futures_pnl(session, config)

        self.assertEqual(
            result["closed_trades"],
            [
                {"symbol": "AERGOUSDT", "position_side": "BOTH", "side": "LONG", "pnl": 1.8, "time": 10020},
                {"symbol": "AERGOUSDT", "position_side": "BOTH", "side": "LONG", "pnl": 1.25, "time": 4000},
            ],
        )

    def test_get_futures_pnl_groups_interleaved_symbol_close_bursts(self):
        session = FakeSession(
            [
                {"positions": []},
                [
                    {"symbol": "AERGOUSDT", "income": "0.70", "time": 10020},
                    {"symbol": "BTCUSDT", "income": "2.00", "time": 10015},
                    {"symbol": "AERGOUSDT", "income": "0.60", "time": 10010},
                ],
                [
                    {"symbol": "AERGOUSDT", "orderId": 203, "side": "SELL", "positionSide": "BOTH", "realizedPnl": "0.70", "time": 10020},
                    {"symbol": "AERGOUSDT", "orderId": 202, "side": "SELL", "positionSide": "BOTH", "realizedPnl": "0.60", "time": 10010},
                ],
                [
                    {"symbol": "BTCUSDT", "orderId": 301, "side": "BUY", "positionSide": "BOTH", "realizedPnl": "2.00", "time": 10015},
                ],
            ]
        )
        config = EnvConfig("key", "secret", "token", "chat")

        with patch("pnlbot.market_data.time.time", return_value=20):
            result = get_futures_pnl(session, config)

        self.assertEqual(
            result["closed_trades"],
            [
                {"symbol": "AERGOUSDT", "position_side": "BOTH", "side": "LONG", "pnl": 1.3, "time": 10020},
                {"symbol": "BTCUSDT", "position_side": "BOTH", "side": "SHORT", "pnl": 2.0, "time": 10015},
            ],
        )


if __name__ == "__main__":
    unittest.main()
