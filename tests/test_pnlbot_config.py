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
from pnlbot.config import load_bot_settings, load_env_config
from pnlbot.logging import RotatingLogStream
from pnlbot.models import BotSettings, BotState, EnvConfig
from pnlbot.freqtrade import FreqtradeHealthResult
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
                    "runtime_config_overrides": ["interval_seconds", "night_mode_enabled", "bot_running", "night_mode_start_hour", "night_mode_end_hour"],
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


    def test_versioned_auto_state_without_runtime_overrides_keeps_env_defaults(self):
        settings = BotSettings(3600, -20, 20, True, (0, 5), init_capital=5000.0)
        state = BotState(
            3600,
            True,
            -20,
            20,
            (0, 5),
            init_capital=settings.init_capital,
        )

        apply_persisted_configuration(
            {
                "state": {
                    "runtime_config_overrides_version": 1,
                    "runtime_config_overrides": [],
                    "init_capital": 1000.0,
                    "freqtrade_ports": [],
                    "freqtrade_alert_cooldown_seconds": 600,
                }
            },
            state,
            settings,
        )

        self.assertEqual(state.init_capital, 5000.0)
        self.assertEqual(state.freqtrade_ports, [])
        self.assertEqual(state.freqtrade_alert_cooldown_seconds, 300)

    def test_legacy_unversioned_state_migrates_runtime_config_values(self):
        settings = BotSettings(3600, -20, 20, True, (0, 5), init_capital=5000.0)
        state = BotState(
            3600,
            True,
            -20,
            20,
            (0, 5),
            init_capital=settings.init_capital,
        )

        apply_persisted_configuration(
            {
                "state": {
                    "runtime_config_overrides": [],
                    "interval_seconds": 1800,
                    "init_capital": 1000.0,
                    "outage_filter": "Kim Dong",
                    "freqtrade_ports": [8123],
                    "freqtrade_alert_cooldown_seconds": 600,
                }
            },
            state,
            settings,
        )

        self.assertEqual(state.interval_seconds, 1800)
        self.assertEqual(state.init_capital, 1000.0)
        self.assertEqual(state.outage_street_filter, "Kim Dong")
        self.assertEqual(state.freqtrade_ports, [8123])
        self.assertEqual(state.freqtrade_alert_cooldown_seconds, 600)
        self.assertIn("interval_seconds", state.runtime_config_overrides)
        self.assertIn("init_capital", state.runtime_config_overrides)
        self.assertIn("outage_filter", state.runtime_config_overrides)
        self.assertIn("freqtrade_ports", state.runtime_config_overrides)
        self.assertIn("freqtrade_alert_cooldown_seconds", state.runtime_config_overrides)

    def test_versioned_empty_override_state_with_ports_recovers_legacy_runtime_values(self):
        settings = BotSettings(3600, -20, 20, True, (0, 5), init_capital=5000.0)
        state = BotState(
            3600,
            True,
            -20,
            20,
            (0, 5),
            init_capital=settings.init_capital,
        )

        apply_persisted_configuration(
            {
                "state": {
                    "runtime_config_overrides_version": 1,
                    "runtime_config_overrides": [],
                    "init_capital": 1000.0,
                    "freqtrade_ports": [8123, 8214],
                }
            },
            state,
            settings,
        )

        self.assertEqual(state.init_capital, 1000.0)
        self.assertEqual(state.freqtrade_ports, [8123, 8214])
        self.assertIn("init_capital", state.runtime_config_overrides)
        self.assertIn("freqtrade_ports", state.runtime_config_overrides)

    def test_persisted_runtime_overrides_replace_env_defaults(self):
        settings = BotSettings(3600, -20, 20, True, (0, 5), init_capital=5000.0)
        state = BotState(
            3600,
            True,
            -20,
            20,
            (0, 5),
            init_capital=settings.init_capital,
            freqtrade_ports=[9000],
        )

        apply_persisted_configuration(
            {
                "state": {
                    "runtime_config_overrides": [
                        "init_capital",
                        "freqtrade_ports",
                        "freqtrade_alert_cooldown_seconds",
                    ],
                    "init_capital": 1000.0,
                    "freqtrade_ports": [8123],
                    "freqtrade_alert_cooldown_seconds": 600,
                }
            },
            state,
            settings,
        )

        self.assertEqual(state.init_capital, 1000.0)
        self.assertEqual(state.freqtrade_ports, [8123])
        self.assertEqual(state.freqtrade_alert_cooldown_seconds, 600)

    def test_persist_runtime_state_preserves_existing_runtime_overrides_missing_from_current_state(self):
        settings, state = make_settings_and_state()
        state.freqtrade_ports = [8123]
        state.runtime_config_overrides = ["freqtrade_ports"]

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "state": {
                            "runtime_config_overrides": [
                                "interval_seconds",
                                "pnl_alert_low",
                                "pnl_alert_high",
                                "freqtrade_ports",
                            ],
                            "runtime_config_overrides_version": 1,
                            "interval_seconds": 1800,
                            "pnl_alert_low": -10,
                            "pnl_alert_high": 30,
                            "freqtrade_ports": [8136, 8138, 8139],
                        }
                    }
                ),
                encoding="utf-8",
            )

            persist_runtime_state(str(state_path), state, settings)

            persisted = json.loads(state_path.read_text(encoding="utf-8"))["state"]

        self.assertEqual(persisted["interval_seconds"], 1800)
        self.assertEqual(persisted["pnl_alert_low"], -10)
        self.assertEqual(persisted["pnl_alert_high"], 30)
        self.assertEqual(persisted["freqtrade_ports"], [8123])
        self.assertIn("interval_seconds", persisted["runtime_config_overrides"])
        self.assertIn("pnl_alert_low", persisted["runtime_config_overrides"])
        self.assertIn("pnl_alert_high", persisted["runtime_config_overrides"])
        self.assertIn("freqtrade_ports", persisted["runtime_config_overrides"])

    def test_persisted_open_position_interval_restore_state_round_trips(self):
        settings, state = make_settings_and_state()
        state.interval_seconds = 15 * 60
        state.pre_open_position_interval_seconds = 1800

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            persist_runtime_state(str(state_path), state, settings)
            persisted = load_persisted_state(str(state_path))

        restored_settings, restored_state = make_settings_and_state()
        apply_persisted_configuration(persisted, restored_state, restored_settings)

        self.assertEqual(restored_state.pre_open_position_interval_seconds, 1800)

    def test_persisted_position_ranges_round_trip(self):
        settings, state = make_settings_and_state()
        state.futures_position_ranges = {
            "BTCUSDT:BOTH:LONG": {
                "symbol": "BTCUSDT",
                "side": "LONG",
                "entry_price": 62000.0,
                "min_pnl": -2.5,
                "min_price": 61750.0,
                "max_pnl": 5.0,
                "max_price": 62500.0,
            }
        }
        state.closed_position_ranges = [
            {
                "key": "ETHUSDT:BOTH:SHORT",
                "symbol": "ETHUSDT",
                "side": "SHORT",
                "entry_price": 3500.0,
                "min_pnl": -1.0,
                "min_price": 3520.0,
                "max_pnl": 3.5,
                "max_price": 3450.0,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            persist_runtime_state(str(state_path), state, settings)
            persisted = load_persisted_state(str(state_path))

        restored_settings, restored_state = make_settings_and_state()
        apply_persisted_configuration(persisted, restored_state, restored_settings)

        restored_range = restored_state.futures_position_ranges["BTCUSDT:BOTH:LONG"]
        self.assertEqual(restored_range["min_price"], 61750.0)
        self.assertEqual(restored_range["max_pnl"], 5.0)
        self.assertEqual(restored_state.closed_position_ranges[0]["symbol"], "ETHUSDT")
        self.assertEqual(restored_state.closed_position_ranges[0]["max_price"], 3450.0)

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
            {"state": {"runtime_config_overrides": ["outage_filter"], "outage_filter": "Main Street"}},
            state,
            settings,
        )

        self.assertEqual(state.outage_street_filter, "Main Street")

    def test_empty_persisted_outage_filter_keeps_env_default(self):
        settings, state = make_settings_and_state()
        state.outage_street_filter = "Env Street"

        apply_persisted_configuration(
            {"state": {"runtime_config_overrides": ["outage_filter"], "outage_filter": None}},
            state,
            settings,
        )

        self.assertEqual(state.outage_street_filter, "Env Street")


class FreqtradeRuntimeConfigTests(unittest.TestCase):
    def test_config_set_freqtrade_ports_updates_runtime_state_and_persists_without_token(self):
        settings, state = make_settings_and_state()
        config = EnvConfig("key", "secret", "token", "chat", freqtrade_api_token="dummy-token")

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            with patch.object(config_commands, "STATE_FILE_PATH", str(state_path)):
                response = config_commands.handle_config_command(
                    "/config set freqtrade_ports 8123, 8214", state, settings, config
                )

            persisted = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertIn("8123,8214", response)
        self.assertEqual(state.freqtrade_ports, [8123, 8214])
        self.assertEqual(persisted["state"]["freqtrade_ports"], [8123, 8214])
        self.assertNotIn("dummy-token", json.dumps(persisted))

    def test_persisted_freqtrade_runtime_settings_are_restored(self):
        settings, state = make_settings_and_state()

        apply_persisted_configuration(
            {
                "state": {
                    "runtime_config_overrides": ["freqtrade_ports", "freqtrade_alert_cooldown_seconds"],
                    "freqtrade_ports": [8123, "8214", "bad", 0],
                    "freqtrade_alert_cooldown_seconds": 600,
                }
            },
            state,
            settings,
        )

        self.assertEqual(state.freqtrade_ports, [8123, 8214])
        self.assertEqual(state.freqtrade_alert_cooldown_seconds, 600)


    def test_config_get_is_removed(self):
        settings, state = make_settings_and_state()
        config = EnvConfig("key", "secret", "token", "chat")

        response = config_commands.handle_config_command("/config get init_capital", state, settings, config)

        self.assertIn("Unsupported", response)

    def test_config_show_reflects_runtime_init_capital_and_freqtrade_ports_updates(self):
        settings, state = make_settings_and_state()
        config = EnvConfig("key", "secret", "token", "chat")

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            with patch.object(config_commands, "STATE_FILE_PATH", str(state_path)):
                config_commands.handle_config_command(
                    "/config set init_capital 2222", state, settings, config
                )
                config_commands.handle_config_command(
                    "/config set freqtrade_ports 8123,8214", state, settings, config
                )
                response = config_commands.handle_config_command("/config show", state, settings, config)

        self.assertIn("`init_capital`: `2222.0`", response)
        self.assertIn("`freqtrade_ports`: `8123,8214`", response)
        self.assertNotIn(" – ", response)
        self.assertNotIn("Initial capital", response)
        self.assertNotIn("Comma-separated", response)

    def test_config_show_refreshes_persisted_runtime_values_before_listing(self):
        settings_a, state_a = make_settings_and_state()
        settings_b, state_b = make_settings_and_state()
        config = EnvConfig("key", "secret", "token", "chat")

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            with patch.object(config_commands, "STATE_FILE_PATH", str(state_path)):
                config_commands.handle_config_command(
                    "/config set init_capital 2222", state_a, settings_a, config
                )
                config_commands.handle_config_command(
                    "/config set freqtrade_ports 8123,8214", state_a, settings_a, config
                )
                response = config_commands.handle_config_command(
                    "/config show", state_b, settings_b, config
                )

        self.assertIn("`init_capital`: `2222.0`", response)
        self.assertIn("`freqtrade_ports`: `8123,8214`", response)

    def test_load_env_config_reads_freqtrade_api_token(self):
        env = {
            "API_KEY": "key",
            "API_SECRET": "secret",
            "TELEGRAM_TOKEN": "telegram-token",
            "TELEGRAM_CHAT_ID": "chat",
            "FREQTRADE_API_TOKEN": "freqtrade-token",
            "FREQTRADE_API_USERNAME": "freqtrade-user",
            "FREQTRADE_API_PASSWORD": "freqtrade-pass",
        }

        with patch.dict(os.environ, env, clear=True):
            config = load_env_config()

        self.assertEqual(config.freqtrade_api_token, "freqtrade-token")
        self.assertEqual(config.freqtrade_api_username, "freqtrade-user")
        self.assertEqual(config.freqtrade_api_password, "freqtrade-pass")


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



class FreqtradeMonitoringTests(unittest.TestCase):
    def test_monitor_loop_enriches_closed_positions_with_freqtrade_exit_reason(self):
        settings, state = make_settings_and_state()
        config = EnvConfig("key", "secret", "token", "chat", freqtrade_api_token="ft-token")
        state.freqtrade_ports = [8123]
        state.pnl_alert_low = -10
        state.pnl_alert_high = 10
        state.last_outage_check = monitoring.time.time()

        snapshot = portfolio.PortfolioSnapshot(
            pnl={
                "total": 12.5,
                "open_positions": [],
                "closed_trades": [{"symbol": "BTCUSDT", "pnl": 12.5}],
            },
            spot_balance={"total": 0.0, "breakdown": []},
            state_changed=False,
        )

        enriched_pnl = {
            "total": 12.5,
            "open_positions": [],
            "closed_trades": [{"symbol": "BTCUSDT", "pnl": 12.5, "exit_reason": "roi"}],
        }

        with patch.object(monitoring.portfolio, "refresh_portfolio_snapshot", return_value=snapshot):
            with patch.object(monitoring, "enrich_pnl_with_freqtrade_exit_reasons", return_value=enriched_pnl):
                with patch.object(monitoring, "send_telegram_message") as send_message:
                    monitoring.monitor_loop(None, config, settings, state)

        messages = [call.args[3] for call in send_message.call_args_list]
        self.assertTrue(any("exit: `roi`" in message for message in messages))

    def test_fast_health_alert_helper_sends_unhealthy_bot_alert_once_per_cooldown(self):
        settings, state = make_settings_and_state()
        config = EnvConfig("key", "secret", "token", "chat", freqtrade_api_token="ft-token")
        state.freqtrade_ports = [8123]
        state.freqtrade_alert_cooldown_seconds = 300

        with patch.object(
            monitoring,
            "check_freqtrade_bots",
            return_value=[FreqtradeHealthResult(8123, False, "connection failed")],
        ):
            with patch.object(monitoring, "send_telegram_message") as send_message:
                changed_first = monitoring.maybe_send_freqtrade_health_alert(
                    None, config, settings, state, now=1000.0
                )
                changed_second = monitoring.maybe_send_freqtrade_health_alert(
                    None, config, settings, state, now=1010.0
                )

        self.assertTrue(changed_first)
        self.assertFalse(changed_second)
        self.assertEqual(send_message.call_count, 1)
        self.assertIn("8123", send_message.call_args.args[3])
        self.assertIn("connection failed", send_message.call_args.args[3])

    def test_monitor_loop_does_not_send_freqtrade_health_alerts(self):
        settings, state = make_settings_and_state()
        config = EnvConfig("key", "secret", "token", "chat", freqtrade_api_token="ft-token")
        state.freqtrade_ports = [8123]
        state.pnl_alert_low = -10
        state.pnl_alert_high = 10
        state.last_outage_check = monitoring.time.time()

        snapshot = portfolio.PortfolioSnapshot(
            pnl=0.0,
            spot_balance={"total": 0.0, "breakdown": []},
            state_changed=False,
        )

        with patch.object(monitoring.portfolio, "refresh_portfolio_snapshot", return_value=snapshot):
            with patch.object(monitoring, "enrich_pnl_with_freqtrade_exit_reasons", return_value=0.0):
                with patch.object(monitoring, "check_freqtrade_bots") as check_bots:
                    with patch.object(monitoring, "send_telegram_message") as send_message:
                        monitoring.monitor_loop(None, config, settings, state)

        check_bots.assert_not_called()
        self.assertFalse(
            any("Freqtrade Health Alert" in call.args[3] for call in send_message.call_args_list)
        )


class DailyStatusSchedulingTests(unittest.TestCase):
    def test_daily_status_is_not_due_when_bot_is_paused(self):
        settings, state = make_settings_and_state()
        state.is_running = False
        now = datetime.datetime(2026, 6, 13, 8, 0, 0)

        self.assertFalse(should_send_daily_status(state, now))


if __name__ == "__main__":
    unittest.main()
