import datetime
import unittest
from unittest.mock import patch

from pnlbot import runtime
from pnlbot.freqtrade import FreqtradeHealthResult
from pnlbot.models import BotSettings, BotState, EnvConfig


BotStateSettings = BotSettings(3600, -20, 20, True, (0, 5))


class StartupStatusMessageTests(unittest.TestCase):
    def test_startup_status_includes_bot_health_and_exit_reasons(self):
        config = EnvConfig("key", "secret", "token", "chat", freqtrade_api_token="ft-token")
        state = BotState(
            3600,
            True,
            -20,
            20,
            (0, 5),
            freqtrade_ports=[8136],
        )
        pnl = {
            "total": 0.0,
            "open_positions": [],
            "closed_trades": [{"symbol": "TAOUSDT", "pnl": -0.06}],
        }

        enriched_pnl = {
            "total": 0.0,
            "open_positions": [],
            "closed_trades": [{"symbol": "TAOUSDT", "pnl": -0.06, "exit_reason": "roi"}],
        }

        with patch.object(runtime, "enrich_pnl_with_freqtrade_exit_reasons", return_value=enriched_pnl):
            with patch.object(
                runtime,
                "check_freqtrade_bots",
                return_value=[FreqtradeHealthResult(8136, True, "healthy")],
            ):
                message = runtime.compose_startup_status_message(
                    None,
                    config,
                    state,
                    pnl,
                    spot_balance={"total": 0.0, "breakdown": []},
                )

        self.assertIn("exit: `roi`", message)
        self.assertIn("\n\n🤖 *Bots:*", message)
        self.assertIn("`8136`: ✅ healthy", message)


class DailyRuntimeMessageTests(unittest.TestCase):
    def test_daily_lunar_pin_contains_only_lunar_date(self):
        config = EnvConfig("key", "secret", "token", "chat")
        state = BotState(3600, True, -20, 20, (0, 5))
        now = datetime.datetime(2026, 6, 14, 8, 0, 0)

        with patch.object(runtime, "get_lunar_date_string", return_value="Mùng `1` Tháng `5`"):
            with patch.object(runtime, "send_telegram_message", return_value={"result": {"message_id": 44}}) as send_message:
                with patch.object(runtime, "pin_telegram_message") as pin_message:
                    with patch.object(runtime, "persist_runtime_state"):
                        sent = runtime.send_daily_lunar_pin(None, config, BotStateSettings, state, now)

        self.assertTrue(sent)
        message = send_message.call_args.args[3]
        self.assertIn("Hôm nay là", message)
        self.assertNotIn("Spot", message)
        pin_message.assert_called_once_with(None, config, 44)
        self.assertEqual(state.last_lunar_alert_date, "2026-06-14")

    def test_daily_spot_report_sends_once_per_day(self):
        config = EnvConfig("key", "secret", "token", "chat")
        settings = BotStateSettings
        state = BotState(3600, True, -20, 20, (0, 5))
        now = datetime.datetime(2026, 6, 14, 8, 0, 0)

        with patch.object(runtime.portfolio, "refresh_spot_balance", return_value=({"total": 100.0, "breakdown": []}, True)) as refresh_spot:
            with patch.object(runtime, "send_telegram_message") as send_message:
                with patch.object(runtime, "persist_runtime_state"):
                    sent_first = runtime.send_daily_spot_report(None, config, settings, state, now)
                    sent_second = runtime.send_daily_spot_report(None, config, settings, state, now)

        self.assertTrue(sent_first)
        self.assertFalse(sent_second)
        refresh_spot.assert_called_once()
        self.assertIn("Spot", send_message.call_args.args[3])
        self.assertEqual(state.last_spot_report_date, "2026-06-14")


if __name__ == "__main__":
    unittest.main()
