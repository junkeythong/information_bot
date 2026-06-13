import unittest

from pnlbot.models import BotState
from pnlbot.state import update_spot_balance_range


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


class StateHelperTests(unittest.TestCase):
    def test_update_spot_balance_range_initializes_and_tracks_bounds(self):
        state = make_state()

        self.assertTrue(update_spot_balance_range(state, 100.0))
        self.assertEqual(state.min_spot_balance, 100.0)
        self.assertEqual(state.max_spot_balance, 100.0)

        self.assertTrue(update_spot_balance_range(state, 80.0))
        self.assertEqual(state.min_spot_balance, 80.0)
        self.assertEqual(state.max_spot_balance, 100.0)

        self.assertFalse(update_spot_balance_range(state, 90.0))
        self.assertEqual(state.min_spot_balance, 80.0)
        self.assertEqual(state.max_spot_balance, 100.0)

    def test_update_spot_balance_range_ignores_empty_balance(self):
        state = make_state(max_spot_balance=100.0, min_spot_balance=80.0)

        self.assertFalse(update_spot_balance_range(state, 0.0))
        self.assertEqual(state.min_spot_balance, 80.0)
        self.assertEqual(state.max_spot_balance, 100.0)


if __name__ == "__main__":
    unittest.main()
