import requests
import unittest

from pnlbot.models import EnvConfig


class FakeResponse:
    def __init__(self, payload=None, status_code=200, text=""):
        self.payload = payload if payload is not None else {}
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FreqtradePortParsingTests(unittest.TestCase):
    def test_parse_freqtrade_ports_accepts_comma_separated_ports(self):
        from pnlbot.freqtrade import parse_freqtrade_ports

        self.assertEqual(parse_freqtrade_ports("8123, 8214"), [8123, 8214])

    def test_parse_freqtrade_ports_rejects_invalid_ports(self):
        from pnlbot.freqtrade import parse_freqtrade_ports

        with self.assertRaisesRegex(ValueError, "port"):
            parse_freqtrade_ports("8123,abc")

        with self.assertRaisesRegex(ValueError, "port"):
            parse_freqtrade_ports("0")


class FreqtradeHealthCheckTests(unittest.TestCase):
    def test_health_check_uses_bearer_token_and_marks_healthy_response(self):
        from pnlbot.freqtrade import check_freqtrade_bots

        config = EnvConfig("key", "secret", "token", "chat", freqtrade_api_token="ft-token")
        session = FakeSession([FakeResponse({"status": "running"})])

        results = check_freqtrade_bots(session, config, [8123])

        self.assertEqual(results[0].port, 8123)
        self.assertTrue(results[0].healthy)
        self.assertEqual(results[0].summary, "healthy")
        self.assertEqual(
            session.calls[0][1],
            "http://127.0.0.1:8123/api/v1/health",
        )
        self.assertEqual(session.calls[0][2]["headers"]["Authorization"], "Bearer ft-token")

    def test_health_check_warns_when_token_is_missing(self):
        from pnlbot.freqtrade import check_freqtrade_bots

        config = EnvConfig("key", "secret", "token", "chat")

        results = check_freqtrade_bots(FakeSession([]), config, [8123])

        self.assertFalse(results[0].healthy)
        self.assertEqual(results[0].summary, "auth not configured")


    def test_health_check_logs_in_with_username_password_and_uses_access_token(self):
        from pnlbot.freqtrade import check_freqtrade_bots, clear_freqtrade_token_cache

        clear_freqtrade_token_cache()
        config = EnvConfig(
            "key",
            "secret",
            "token",
            "chat",
            freqtrade_api_username="user",
            freqtrade_api_password="pass",
        )
        session = FakeSession([
            FakeResponse({"access_token": "access-token"}),
            FakeResponse({"status": "running"}),
        ])

        results = check_freqtrade_bots(session, config, [8123])

        self.assertTrue(results[0].healthy)
        self.assertEqual(session.calls[0][0], "POST")
        self.assertEqual(session.calls[0][1], "http://127.0.0.1:8123/api/v1/token/login")
        self.assertEqual(session.calls[0][2]["auth"], ("user", "pass"))
        self.assertEqual(session.calls[1][2]["headers"]["Authorization"], "Bearer access-token")

    def test_health_check_reauthenticates_once_when_cached_token_is_unauthorized(self):
        from pnlbot.freqtrade import check_freqtrade_bots, clear_freqtrade_token_cache

        clear_freqtrade_token_cache()
        config = EnvConfig(
            "key",
            "secret",
            "token",
            "chat",
            freqtrade_api_username="user",
            freqtrade_api_password="pass",
        )
        session = FakeSession([
            FakeResponse({"access_token": "old-token"}),
            FakeResponse({}, status_code=401),
            FakeResponse({"access_token": "new-token"}),
            FakeResponse({"status": "running"}),
        ])

        results = check_freqtrade_bots(session, config, [8123])

        self.assertTrue(results[0].healthy)
        self.assertEqual(session.calls[3][2]["headers"]["Authorization"], "Bearer new-token")


    def test_health_check_hides_raw_auth_failure_details(self):
        from pnlbot.freqtrade import check_freqtrade_bots, clear_freqtrade_token_cache

        clear_freqtrade_token_cache()
        config = EnvConfig(
            "key",
            "secret",
            "token",
            "chat",
            freqtrade_api_username="user",
            freqtrade_api_password="pass",
        )
        session = FakeSession([FakeResponse({}, status_code=401)])

        results = check_freqtrade_bots(session, config, [8123])

        self.assertFalse(results[0].healthy)
        self.assertEqual(results[0].summary, "auth failed")

    def test_health_check_marks_stopped_response_unhealthy(self):
        from pnlbot.freqtrade import check_freqtrade_bots

        config = EnvConfig("key", "secret", "token", "chat", freqtrade_api_token="ft-token")
        session = FakeSession([FakeResponse({"status": "stopped"})])

        results = check_freqtrade_bots(session, config, [8123])

        self.assertFalse(results[0].healthy)
        self.assertIn("stopped", results[0].summary)

    def test_format_freqtrade_status_section_lists_each_port(self):
        from pnlbot.freqtrade import FreqtradeHealthResult, format_freqtrade_status_section

        message = format_freqtrade_status_section(
            [
                FreqtradeHealthResult(port=8123, healthy=True, summary="healthy"),
                FreqtradeHealthResult(port=8214, healthy=False, summary="stopped"),
            ]
        )

        self.assertTrue(message.startswith("\n\n🤖 *Bots:*"))
        self.assertIn("*Bots:*", message)
        self.assertNotIn("Freqtrade Bots", message)
        self.assertIn("`8123`: ✅ healthy", message)
        self.assertIn("`8214`: 🔴 stopped", message)


class FreqtradeExitReasonTests(unittest.TestCase):
    def test_fetch_exit_reasons_maps_closed_freqtrade_trades_by_symbol(self):
        from pnlbot.freqtrade import clear_freqtrade_token_cache, fetch_freqtrade_exit_reasons

        clear_freqtrade_token_cache()
        config = EnvConfig(
            "key",
            "secret",
            "token",
            "chat",
            freqtrade_api_username="user",
            freqtrade_api_password="pass",
        )
        session = FakeSession([
            FakeResponse({"access_token": "access-token"}),
            FakeResponse({
                "trades": [
                    {"pair": "BTC/USDT:USDT", "exit_reason": "roi", "is_open": False},
                    {"pair": "ETH/USDT", "sell_reason": "stop_loss", "is_open": False},
                ]
            }),
        ])

        reasons = fetch_freqtrade_exit_reasons(session, config, [8123])

        self.assertEqual(session.calls[1][2]["params"]["limit"], 500)
        self.assertEqual(session.calls[1][2]["params"]["order_by_id"], "false")
        self.assertEqual(reasons["BTCUSDT"], "roi")
        self.assertEqual(reasons["ETHUSDT"], "stop_loss")

    def test_apply_exit_reasons_adds_reason_to_matching_closed_trades(self):
        from pnlbot.freqtrade import apply_exit_reasons_to_closed_trades

        pnl = {
            "total": 0.0,
            "open_positions": [],
            "closed_trades": [
                {"symbol": "BTCUSDT", "pnl": 10.0},
                {"symbol": "SOLUSDT", "pnl": -2.0},
            ],
        }

        updated = apply_exit_reasons_to_closed_trades(pnl, {"BTCUSDT": "roi"})

        self.assertEqual(updated["closed_trades"][0]["exit_reason"], "roi")
        self.assertNotIn("exit_reason", updated["closed_trades"][1])


if __name__ == "__main__":
    unittest.main()
