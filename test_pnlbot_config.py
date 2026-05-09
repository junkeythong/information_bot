import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import PnLBot


def make_settings_and_state():
    settings = PnLBot.BotSettings(
        default_interval_seconds=3600,
        default_pnl_alert_low=-20,
        default_pnl_alert_high=20,
        default_night_mode_enabled=True,
        night_mode_window=(0, 5),
    )
    state = PnLBot.BotState(
        interval_seconds=settings.default_interval_seconds,
        night_mode_enabled=settings.default_night_mode_enabled,
        pnl_alert_low=settings.default_pnl_alert_low,
        pnl_alert_high=settings.default_pnl_alert_high,
        night_mode_window=settings.night_mode_window,
    )
    return settings, state


class BotSettingsValidationTests(unittest.TestCase):
    def test_rejects_default_interval_outside_runtime_bounds(self):
        with patch.dict(os.environ, {"PNL_BOT_DEFAULT_INTERVAL_SECONDS": "-1"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "PNL_BOT_DEFAULT_INTERVAL_SECONDS"):
                PnLBot.load_bot_settings()

    def test_rejects_default_pnl_thresholds_that_cross(self):
        env = {
            "PNL_BOT_DEFAULT_PNL_ALERT_LOW": "20",
            "PNL_BOT_DEFAULT_PNL_ALERT_HIGH": "20",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "lower than high"):
                PnLBot.load_bot_settings()


class PersistedStateValidationTests(unittest.TestCase):
    def test_ignores_invalid_persisted_bounds_and_parses_boolean_strings(self):
        settings, state = make_settings_and_state()

        PnLBot.apply_persisted_configuration(
            {
                "state": {
                    "interval_seconds": -1,
                    "night_mode_enabled": "false",
                    "is_running": "false",
                    "night_mode_window": [99, 99],
                }
            },
            state,
            settings,
        )

        self.assertEqual(state.interval_seconds, 3600)
        self.assertFalse(state.night_mode_enabled)
        self.assertFalse(state.is_running)
        self.assertEqual(state.night_mode_window, (0, 5))

    def test_persist_runtime_state_keeps_backup_of_previous_state(self):
        settings, state = make_settings_and_state()
        state.max_spot_balance = 1200.0
        state.min_spot_balance = 700.0

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            backup_path = Path(f"{state_path}.bak")
            state_path.write_text(
                json.dumps({"state": {"max_spot_balance": 999.0, "min_spot_balance": 888.0}}),
                encoding="utf-8",
            )

            PnLBot.persist_runtime_state(str(state_path), state, settings)

            current = json.loads(state_path.read_text(encoding="utf-8"))
            backup = json.loads(backup_path.read_text(encoding="utf-8"))

        self.assertEqual(current["state"]["max_spot_balance"], 1200.0)
        self.assertEqual(current["state"]["min_spot_balance"], 700.0)
        self.assertEqual(backup["state"]["max_spot_balance"], 999.0)
        self.assertEqual(backup["state"]["min_spot_balance"], 888.0)

    def test_load_persisted_state_falls_back_to_backup_when_primary_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            backup_path = Path(f"{state_path}.bak")
            backup_path.write_text(
                json.dumps({"state": {"max_spot_balance": 1200.0, "min_spot_balance": 700.0}}),
                encoding="utf-8",
            )

            output = StringIO()
            with redirect_stdout(output):
                persisted = PnLBot.load_persisted_state(str(state_path))

        self.assertEqual(persisted["state"]["max_spot_balance"], 1200.0)
        self.assertIn("backup", output.getvalue().lower())

    def test_load_persisted_state_falls_back_to_backup_when_primary_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            backup_path = Path(f"{state_path}.bak")
            state_path.write_text("{bad json", encoding="utf-8")
            backup_path.write_text(
                json.dumps({"state": {"max_spot_balance": 1200.0, "min_spot_balance": 700.0}}),
                encoding="utf-8",
            )

            output = StringIO()
            with redirect_stdout(output):
                persisted = PnLBot.load_persisted_state(str(state_path))

        self.assertEqual(persisted["state"]["min_spot_balance"], 700.0)
        self.assertIn("backup", output.getvalue().lower())


class OutageFilterConfigTests(unittest.TestCase):
    def test_config_set_outage_filter_updates_runtime_config_and_persists(self):
        settings, state = make_settings_and_state()
        config = PnLBot.EnvConfig("key", "secret", "token", "chat")

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            with patch.object(PnLBot, "STATE_FILE_PATH", str(state_path)):
                response = PnLBot.handle_config_command(
                    "/config set outage_filter Main Street", state, settings, config
                )

            persisted = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertIn("Main Street", response)
        self.assertEqual(config.outage_street_filter, "Main Street")
        self.assertEqual(state.outage_street_filter, "Main Street")
        self.assertEqual(persisted["state"]["outage_filter"], "Main Street")

    def test_config_set_outage_filter_none_clears_runtime_config_and_persists(self):
        settings, state = make_settings_and_state()
        state.outage_street_filter = "Main Street"
        config = PnLBot.EnvConfig("key", "secret", "token", "chat", outage_street_filter="Main Street")

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            with patch.object(PnLBot, "STATE_FILE_PATH", str(state_path)):
                response = PnLBot.handle_config_command(
                    "/config set outage_filter none", state, settings, config
                )

            persisted = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertIn("None", response)
        self.assertIsNone(config.outage_street_filter)
        self.assertIsNone(state.outage_street_filter)
        self.assertIsNone(persisted["state"]["outage_filter"])

    def test_persisted_outage_filter_is_restored_to_state(self):
        settings, state = make_settings_and_state()

        PnLBot.apply_persisted_configuration(
            {"state": {"outage_filter": "Main Street"}},
            state,
            settings,
        )

        self.assertEqual(state.outage_street_filter, "Main Street")


if __name__ == "__main__":
    unittest.main()
