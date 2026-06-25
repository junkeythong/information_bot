import unittest
from unittest.mock import patch

from pnlbot import runtime
from pnlbot.freqtrade import FreqtradeHealthResult
from pnlbot.models import BotState, EnvConfig


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


if __name__ == "__main__":
    unittest.main()
