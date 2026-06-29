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
    def test_refresh_portfolio_snapshot_fetches_and_updates_spot_range(self):
        state = make_state(
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
        self.assertEqual(state.max_spot_balance, 120.0)
        self.assertEqual(state.min_spot_balance, 90.0)

    def test_refresh_portfolio_snapshot_sets_fifteen_minute_interval_for_open_positions(self):
        state = make_state(interval_seconds=3600)
        config = EnvConfig("key", "secret", "token", "chat")
        pnl = {
            "total": 12.5,
            "open_positions": [
                {
                    "symbol": "BTCUSDT",
                    "position_side": "BOTH",
                    "side": "LONG",
                    "entry_price": 62000.0,
                    "mark_price": 63250.0,
                    "unrealized_pnl": 12.5,
                }
            ],
            "closed_trades": [],
        }

        with patch.object(portfolio, "get_futures_pnl", return_value=pnl):
            with patch.object(
                portfolio,
                "get_spot_balance",
                return_value={"total": 0.0, "breakdown": []},
            ):
                snapshot = portfolio.refresh_portfolio_snapshot(None, config, state)

        self.assertEqual(state.interval_seconds, 15 * 60)
        self.assertEqual(state.pre_open_position_interval_seconds, 3600)
        self.assertTrue(snapshot.state_changed)

    def test_refresh_portfolio_snapshot_restores_previous_interval_when_positions_close(self):
        state = make_state(
            interval_seconds=15 * 60,
            pre_open_position_interval_seconds=1800,
        )
        config = EnvConfig("key", "secret", "token", "chat")
        pnl = {"total": 0.0, "open_positions": [], "closed_trades": []}

        with patch.object(portfolio, "get_futures_pnl", return_value=pnl):
            with patch.object(
                portfolio,
                "get_spot_balance",
                return_value={"total": 0.0, "breakdown": []},
            ):
                snapshot = portfolio.refresh_portfolio_snapshot(None, config, state)

        self.assertEqual(state.interval_seconds, 1800)
        self.assertIsNone(state.pre_open_position_interval_seconds)
        self.assertTrue(snapshot.state_changed)

    def test_refresh_portfolio_snapshot_keeps_interval_without_open_positions(self):
        state = make_state(interval_seconds=1800)
        config = EnvConfig("key", "secret", "token", "chat")
        pnl = {"total": 0.0, "open_positions": [], "closed_trades": []}

        with patch.object(portfolio, "get_futures_pnl", return_value=pnl):
            with patch.object(
                portfolio,
                "get_spot_balance",
                return_value={"total": 0.0, "breakdown": []},
            ):
                snapshot = portfolio.refresh_portfolio_snapshot(None, config, state)

        self.assertEqual(state.interval_seconds, 1800)
        self.assertFalse(snapshot.state_changed)

    def test_refresh_futures_pnl_tracks_min_max_per_open_position_with_prices(self):
        state = make_state()
        config = EnvConfig("key", "secret", "token", "chat")
        first_pnl = {
            "total": 5.0,
            "open_positions": [
                {
                    "symbol": "BTCUSDT",
                    "position_side": "BOTH",
                    "side": "LONG",
                    "entry_price": 62000.0,
                    "mark_price": 62500.0,
                    "unrealized_pnl": 5.0,
                }
            ],
            "closed_trades": [],
        }
        second_pnl = {
            "total": -2.5,
            "open_positions": [
                {
                    "symbol": "BTCUSDT",
                    "position_side": "BOTH",
                    "side": "LONG",
                    "entry_price": 62000.0,
                    "mark_price": 61750.0,
                    "unrealized_pnl": -2.5,
                }
            ],
            "closed_trades": [],
        }

        with patch.object(portfolio, "get_futures_pnl", side_effect=[first_pnl, second_pnl]):
            first, first_changed = portfolio.refresh_futures_pnl(None, config, state)
            second, second_changed = portfolio.refresh_futures_pnl(None, config, state)

        self.assertTrue(first_changed)
        self.assertTrue(second_changed)
        self.assertEqual(first["open_positions"][0]["pnl_range"]["max_pnl"], 5.0)
        self.assertEqual(second["open_positions"][0]["pnl_range"]["max_price"], 62500.0)
        self.assertEqual(second["open_positions"][0]["pnl_range"]["min_pnl"], -2.5)
        self.assertEqual(second["open_positions"][0]["pnl_range"]["min_price"], 61750.0)

    def test_refresh_futures_pnl_attaches_observed_range_to_latest_closed_trade(self):
        state = make_state()
        config = EnvConfig("key", "secret", "token", "chat")
        open_pnl = {
            "total": 5.0,
            "open_positions": [
                {
                    "symbol": "BTCUSDT",
                    "position_side": "BOTH",
                    "side": "LONG",
                    "entry_price": 62000.0,
                    "mark_price": 62500.0,
                    "unrealized_pnl": 5.0,
                }
            ],
            "closed_trades": [],
        }
        closed_pnl = {
            "total": 0.0,
            "open_positions": [],
            "closed_trades": [
                {"symbol": "BTCUSDT", "position_side": "BOTH", "side": "LONG", "pnl": 4.0}
            ],
        }

        with patch.object(portfolio, "get_futures_pnl", side_effect=[open_pnl, closed_pnl]):
            portfolio.refresh_futures_pnl(None, config, state)
            closed, state_changed = portfolio.refresh_futures_pnl(None, config, state)

        self.assertTrue(state_changed)
        self.assertEqual(state.futures_position_ranges, {})
        self.assertEqual(state.closed_position_ranges, [])
        self.assertEqual(closed["closed_trades"][0]["pnl_range"]["max_pnl"], 5.0)
        self.assertEqual(closed["closed_trades"][0]["pnl_range"]["max_price"], 62500.0)

    def test_refresh_futures_pnl_does_not_attach_range_to_symbol_only_closed_trade(self):
        state = make_state(
            closed_position_ranges=[
                {
                    "key": "BTCUSDT:BOTH:LONG",
                    "symbol": "BTCUSDT",
                    "position_side": "BOTH",
                    "side": "LONG",
                    "min_pnl": -2.5,
                    "min_price": 61750.0,
                    "max_pnl": 5.0,
                    "max_price": 62500.0,
                }
            ]
        )
        config = EnvConfig("key", "secret", "token", "chat")
        closed_pnl = {
            "total": 0.0,
            "open_positions": [],
            "closed_trades": [{"symbol": "BTCUSDT", "pnl": 4.0}],
        }

        with patch.object(portfolio, "get_futures_pnl", return_value=closed_pnl):
            closed, state_changed = portfolio.refresh_futures_pnl(None, config, state)

        self.assertFalse(state_changed)
        self.assertNotIn("pnl_range", closed["closed_trades"][0])
        self.assertEqual(len(state.closed_position_ranges), 1)

    def test_refresh_futures_pnl_does_not_attach_range_to_opposite_side_closed_trade(self):
        state = make_state(
            closed_position_ranges=[
                {
                    "key": "BTCUSDT:BOTH:LONG",
                    "symbol": "BTCUSDT",
                    "position_side": "BOTH",
                    "side": "LONG",
                    "min_pnl": -2.5,
                    "min_price": 61750.0,
                    "max_pnl": 5.0,
                    "max_price": 62500.0,
                }
            ]
        )
        config = EnvConfig("key", "secret", "token", "chat")
        closed_pnl = {
            "total": 0.0,
            "open_positions": [],
            "closed_trades": [
                {"symbol": "BTCUSDT", "position_side": "BOTH", "side": "SHORT", "pnl": 4.0}
            ],
        }

        with patch.object(portfolio, "get_futures_pnl", return_value=closed_pnl):
            closed, state_changed = portfolio.refresh_futures_pnl(None, config, state)

        self.assertFalse(state_changed)
        self.assertNotIn("pnl_range", closed["closed_trades"][0])
        self.assertEqual(len(state.closed_position_ranges), 1)

    def test_refresh_futures_pnl_keeps_range_when_entry_price_changes(self):
        state = make_state()
        config = EnvConfig("key", "secret", "token", "chat")
        first_pnl = {
            "total": 5.0,
            "open_positions": [
                {
                    "symbol": "BTCUSDT",
                    "position_side": "BOTH",
                    "side": "LONG",
                    "entry_price": 62000.0,
                    "mark_price": 62500.0,
                    "unrealized_pnl": 5.0,
                }
            ],
            "closed_trades": [],
        }
        second_pnl = {
            "total": 8.0,
            "open_positions": [
                {
                    "symbol": "BTCUSDT",
                    "position_side": "BOTH",
                    "side": "LONG",
                    "entry_price": 62100.0,
                    "mark_price": 62800.0,
                    "unrealized_pnl": 8.0,
                }
            ],
            "closed_trades": [],
        }

        with patch.object(portfolio, "get_futures_pnl", side_effect=[first_pnl, second_pnl]):
            portfolio.refresh_futures_pnl(None, config, state)
            second, state_changed = portfolio.refresh_futures_pnl(None, config, state)

        self.assertTrue(state_changed)
        self.assertEqual(list(state.futures_position_ranges), ["BTCUSDT:BOTH:LONG"])
        self.assertEqual(state.closed_position_ranges, [])
        self.assertEqual(second["open_positions"][0]["pnl_range"]["max_pnl"], 8.0)
        self.assertEqual(second["open_positions"][0]["pnl_range"]["min_pnl"], 5.0)

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
        state = make_state()
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
        state = make_state()
        pnl = {
            "total": 12.5,
            "open_positions": [
                {
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "mark_price": 63000.0,
                    "unrealized_pnl": 10.0,
                    "pnl_range": {
                        "min_pnl": -2.5,
                        "min_price": 61000.0,
                        "max_pnl": 10.0,
                        "max_price": 63000.0,
                    },
                },
                {"symbol": "ETHUSDT", "unrealized_pnl": 2.5},
            ],
            "closed_trades": [
                {
                    "symbol": "SOLUSDT",
                    "pnl": 4.25,
                    "pnl_range": {
                        "min_pnl": -1.0,
                        "min_price": 98.0,
                        "max_pnl": 6.5,
                        "max_price": 105.0,
                    },
                },
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
        self.assertIn("63,000", message)
        self.assertIn("Observed Min", message)
        self.assertIn("61,000", message)
        self.assertIn("Observed Max", message)
        self.assertIn("Latest Closed Positions", message)
        self.assertIn("SOLUSDT", message)
        self.assertIn("4.25 USDT", message)
        self.assertIn("Observed Open Min", message)
        self.assertIn("98", message)
        self.assertIn("Observed Open Max", message)
        self.assertIn("105", message)
        self.assertIn("BNBUSDT", message)
        self.assertIn("-1.50 USDT", message)


    def test_format_futures_summary_includes_closed_trade_exit_reason_when_available(self):
        state = make_state()
        pnl = {
            "total": 0.0,
            "open_positions": [],
            "closed_trades": [
                {"symbol": "BTCUSDT", "pnl": 12.5, "exit_reason": "roi"},
            ],
        }

        message = portfolio.format_futures_pnl_summary(state, pnl)

        self.assertIn("BTCUSDT", message)
        self.assertIn("exit: `roi`", message)


if __name__ == "__main__":
    unittest.main()
