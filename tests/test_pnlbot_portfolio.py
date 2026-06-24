import unittest
from unittest.mock import patch

from pnlbot.models import BotState, EnvConfig
from pnlbot import portfolio


def make_state(**overrides):
    values = {
        "interval_seconds": 3600,
        "night_mode_enabled": True,
        "pnl_alert_low": -20,
        "pnl_alert_high": 20,
        "night_mode_window": (0, 5),
    }
    values.update(overrides)
    return BotState(**values)


class PortfolioSnapshotTests(unittest.TestCase):
    def test_refresh_portfolio_snapshot_fetches_and_updates_ranges(self):
        state = make_state(
            max_pnl=100.0,
            min_pnl=100.0,
            max_spot_balance=120.0,
            min_spot_balance=120.0,
        )
        config = EnvConfig("key", "secret", "token", "chat")

        with patch.object(portfolio, "get_futures_pnl", return_value=50.0):
            with patch.object(
                portfolio,
                "get_spot_balance",
                return_value={"total": 90.0, "breakdown": []},
            ):
                snapshot = portfolio.refresh_portfolio_snapshot(None, config, state)

        self.assertEqual(snapshot.pnl, 50.0)
        self.assertEqual(snapshot.spot_balance["total"], 90.0)
        self.assertTrue(snapshot.state_changed)
        self.assertEqual(state.max_pnl, 100.0)
        self.assertEqual(state.min_pnl, 50.0)
        self.assertEqual(state.max_spot_balance, 120.0)
        self.assertEqual(state.min_spot_balance, 90.0)

    def test_format_spot_balance_summary_limits_breakdown_when_requested(self):
        state = make_state(
            init_capital=100.0,
            max_spot_balance=120.0,
            min_spot_balance=80.0,
        )
        spot_balance = {
            "total": 110.0,
            "breakdown": [
                {"asset": "BTC", "usdt_value": 70.0, "price": 100000.0},
                {"asset": "ETH", "usdt_value": 40.0, "price": 3000.0},
            ],
        }

        message = portfolio.format_spot_balance_summary(
            state,
            spot_balance,
            max_breakdown_items=1,
            include_asset_heading=True,
        )

        self.assertIn("Spot", message)
        self.assertIn("+10.00%", message)
        self.assertIn("BTC", message)
        self.assertNotIn("ETH", message)
        self.assertIn("Asset Breakdown", message)

    def test_format_monitoring_message_combines_spot_futures_and_aqi(self):
        state = make_state(max_pnl=50.0, min_pnl=-10.0)
        config = EnvConfig("key", "secret", "token", "chat", iqair_api_key="iqair")
        snapshot = portfolio.PortfolioSnapshot(
            pnl=25.0,
            spot_balance={"total": 100.0, "breakdown": []},
            state_changed=False,
        )

        with patch.object(
            portfolio,
            "get_air_quality",
            return_value={"aqi_us": 85, "city": "Ho Chi Minh", "temperature": 28},
        ):
            message = portfolio.format_monitoring_message(None, config, state, snapshot)

        self.assertIn("Spot", message)
        self.assertIn("High profit", message)
        self.assertIn("AQI", message)

    def test_format_futures_pnl_summary_includes_open_and_closed_trades(self):
        state = make_state(max_pnl=50.0, min_pnl=-10.0)
        pnl = {
            "total": 12.5,
            "open_positions": [
                {"symbol": "BTCUSDT", "unrealized_pnl": 10.0},
                {"symbol": "ETHUSDT", "unrealized_pnl": 2.5},
            ],
            "closed_trades": [
                {"symbol": "SOLUSDT", "pnl": 4.25},
                {"symbol": "BNBUSDT", "pnl": -1.5},
                {"symbol": "ADAUSDT", "pnl": 0.0},
            ],
        }

        message = portfolio.format_futures_pnl_summary(state, pnl)

        self.assertIn("Futures", message)
        self.assertIn("12.50 USDT", message)
        self.assertIn("Open Positions", message)
        self.assertIn("BTCUSDT", message)
        self.assertIn("10.00 USDT", message)
        self.assertIn("Latest Closed Positions", message)
        self.assertIn("SOLUSDT", message)
        self.assertIn("4.25 USDT", message)
        self.assertIn("BNBUSDT", message)
        self.assertIn("-1.50 USDT", message)

    def test_refresh_futures_pnl_tracks_range_from_structured_total(self):
        state = make_state(max_pnl=5.0, min_pnl=5.0)
        config = EnvConfig("key", "secret", "token", "chat")

        with patch.object(
            portfolio,
            "get_futures_pnl",
            return_value={"total": -7.5, "open_positions": [], "closed_trades": []},
        ):
            pnl, state_changed = portfolio.refresh_futures_pnl(None, config, state)

        self.assertEqual(pnl["total"], -7.5)
        self.assertTrue(state_changed)
        self.assertEqual(state.max_pnl, 5.0)
        self.assertEqual(state.min_pnl, -7.5)


if __name__ == "__main__":
    unittest.main()
