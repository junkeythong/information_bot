import unittest

from pnlbot.models import BotState
from pnlbot.state import update_spot_balance_range


def make_state(**overrides):
    values = {
        "interval_seconds": 3600,
        "night_mode_enabled": True,
        "night_mode_window": (0, 5),
    }
    values.update(overrides)
    return BotState(**values)


class StateHelperTests(unittest.TestCase):
    def test_update_spot_balance_range_initializes_and_tracks_bounds(self):
        state = make_state()
        first_snapshot = [{"asset": "BTC", "amount": 0.01, "price": 50000.0, "usdt_value": 500.0}]
        min_snapshot = [{"asset": "ETH", "amount": 0.2, "price": 2500.0, "usdt_value": 500.0}]
        max_snapshot = [{"asset": "SOL", "amount": 10.0, "price": 120.0, "usdt_value": 1200.0}]

        self.assertTrue(update_spot_balance_range(state, 100.0, asset_snapshot=first_snapshot, observed_at="t1"))
        self.assertEqual(state.min_spot_balance, 100.0)
        self.assertEqual(state.max_spot_balance, 100.0)
        self.assertEqual(state.min_spot_assets, first_snapshot)
        self.assertEqual(state.max_spot_assets, first_snapshot)
        self.assertEqual(state.min_spot_observed_at, "t1")
        self.assertEqual(state.max_spot_observed_at, "t1")

        self.assertTrue(update_spot_balance_range(state, 80.0, asset_snapshot=min_snapshot, observed_at="t2"))
        self.assertEqual(state.min_spot_balance, 80.0)
        self.assertEqual(state.max_spot_balance, 100.0)
        self.assertEqual(state.min_spot_assets, min_snapshot)
        self.assertEqual(state.max_spot_assets, first_snapshot)
        self.assertEqual(state.min_spot_observed_at, "t2")
        self.assertEqual(state.max_spot_observed_at, "t1")

        self.assertTrue(update_spot_balance_range(state, 120.0, asset_snapshot=max_snapshot, observed_at="t3"))
        self.assertEqual(state.min_spot_balance, 80.0)
        self.assertEqual(state.max_spot_balance, 120.0)
        self.assertEqual(state.min_spot_assets, min_snapshot)
        self.assertEqual(state.max_spot_assets, max_snapshot)
        self.assertEqual(state.min_spot_observed_at, "t2")
        self.assertEqual(state.max_spot_observed_at, "t3")

        self.assertFalse(
            update_spot_balance_range(
                state,
                90.0,
                asset_snapshot=[{"asset": "BNB", "amount": 1.0, "price": 700.0, "usdt_value": 700.0}],
                observed_at="t4",
            )
        )
        self.assertEqual(state.min_spot_balance, 80.0)
        self.assertEqual(state.max_spot_balance, 120.0)
        self.assertEqual(state.min_spot_assets, min_snapshot)
        self.assertEqual(state.max_spot_assets, max_snapshot)

    def test_update_spot_balance_range_ignores_empty_balance(self):
        state = make_state(max_spot_balance=100.0, min_spot_balance=80.0)

        self.assertFalse(update_spot_balance_range(state, 0.0))
        self.assertEqual(state.min_spot_balance, 80.0)
        self.assertEqual(state.max_spot_balance, 100.0)


if __name__ == "__main__":
    unittest.main()
