import datetime
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from pnlbot import config_commands, http as http_client, monitoring, portfolio
from pnlbot.config import load_bot_settings
from pnlbot.logging import RotatingLogStream
from pnlbot.models import BotSettings, BotState, EnvConfig
from pnlbot.persistence import (
    apply_persisted_configuration,
    load_persisted_state,
    persist_runtime_state,
)
from pnlbot.time_utils import should_send_daily_status


def make_settings_and_state():
    settings = BotSettings(
        default_interval_seconds=3600,
        default_pnl_alert_low=-20,
        default_pnl_alert_high=20,
        default_night_mode_enabled=True,
        night_mode_window=(0, 5),
    )
    state = BotState(
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
                load_bot_settings()

    def test_rejects_default_pnl_thresholds_that_cross(self):
        env = {
            "PNL_BOT_DEFAULT_PNL_ALERT_LOW": "20",
            "PNL_BOT_DEFAULT_PNL_ALERT_HIGH": "20",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "lower than high"):
                load_bot_settings()


class RequestsSessionTests(unittest.TestCase):
    def test_retry_session_falls_back_to_system_ca_when_certifi_path_is_missing(self):
        with patch.object(http_client.requests.certs, "where", return_value="/missing/cacert.pem"):
            with patch.object(http_client.os.path, "exists", side_effect=lambda path: path == "/system/ca.pem"):
                with patch.object(http_client, "SYSTEM_CA_BUNDLE_PATHS", ("/system/ca.pem",)):
                    session = http_client.create_retry_session()

        self.assertEqual(session.verify, "/system/ca.pem")


class LogRotationTests(unittest.TestCase):
    def test_rotating_log_stream_keeps_current_log_and_one_backup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "pnlbot.log"
            stream = RotatingLogStream(str(log_path), max_bytes=10)
            try:
                stream.write("123456789")
                stream.write("abc")
            finally:
                stream.close()

            files = sorted(path.name for path in Path(tmpdir).iterdir())

        self.assertEqual(files, ["pnlbot.log", "pnlbot.log.1"])


class PersistedStateValidationTests(unittest.TestCase):
    def test_ignores_invalid_persisted_bounds_and_parses_boolean_strings(self):
        settings, state = make_settings_and_state()

        apply_persisted_configuration(
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

            persist_runtime_state(str(state_path), state, settings)

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
                persisted = load_persisted_state(str(state_path))

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
                persisted = load_persisted_state(str(state_path))

        self.assertEqual(persisted["state"]["min_spot_balance"], 700.0)
        self.assertIn("backup", output.getvalue().lower())


class OutageFilterConfigTests(unittest.TestCase):
    def test_config_set_outage_filter_updates_runtime_config_and_persists(self):
        settings, state = make_settings_and_state()
        config = EnvConfig("key", "secret", "token", "chat")

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            with patch.object(config_commands, "STATE_FILE_PATH", str(state_path)):
                response = config_commands.handle_config_command(
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
        config = EnvConfig("key", "secret", "token", "chat", outage_street_filter="Main Street")

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            with patch.object(config_commands, "STATE_FILE_PATH", str(state_path)):
                response = config_commands.handle_config_command(
                    "/config set outage_filter none", state, settings, config
                )

            persisted = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertIn("None", response)
        self.assertIsNone(config.outage_street_filter)
        self.assertIsNone(state.outage_street_filter)
        self.assertIsNone(persisted["state"]["outage_filter"])

    def test_persisted_outage_filter_is_restored_to_state(self):
        settings, state = make_settings_and_state()

        apply_persisted_configuration(
            {"state": {"outage_filter": "Main Street"}},
            state,
            settings,
        )

        self.assertEqual(state.outage_street_filter, "Main Street")


class PowerOutageRefreshTests(unittest.TestCase):
    def test_empty_outage_refresh_updates_last_check_time(self):
        settings, state = make_settings_and_state()
        config = EnvConfig("key", "secret", "token", "chat")
        state.last_outage_check = 10.0

        with patch.object(monitoring, "get_power_outages", return_value=[]):
            with patch.object(monitoring.time, "time", return_value=1234.0):
                new_outages = monitoring.refresh_power_outages(None, config, state)

        self.assertEqual(new_outages, [])
        self.assertEqual(state.last_outage_check, 1234.0)


class PnlRangeTrackingTests(unittest.TestCase):
    def test_monitor_loop_tracks_lower_positive_futures_pnl_after_reset(self):
        settings, state = make_settings_and_state()
        config = EnvConfig("key", "secret", "token", "chat")
        state.max_pnl = 100.0
        state.min_pnl = 100.0
        state.last_outage_check = monitoring.time.time()

        with patch.object(portfolio, "get_futures_pnl", return_value=50.0):
            with patch.object(portfolio, "get_spot_balance", return_value={"total": 0.0, "breakdown": []}):
                with patch.object(monitoring, "send_telegram_message"):
                    with patch.object(monitoring, "persist_runtime_state"):
                        monitoring.monitor_loop(None, config, settings, state)

        self.assertEqual(state.max_pnl, 100.0)
        self.assertEqual(state.min_pnl, 50.0)


class DailyStatusSchedulingTests(unittest.TestCase):
    def test_daily_status_is_not_due_when_bot_is_paused(self):
        settings, state = make_settings_and_state()
        state.is_running = False
        now = datetime.datetime(2026, 6, 13, 8, 0, 0)

        self.assertFalse(should_send_daily_status(state, now))


if __name__ == "__main__":
    unittest.main()
