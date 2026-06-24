import unittest
from unittest.mock import patch

import requests
from pnlbot import command_handlers, commands, portfolio, telegram
from pnlbot.models import BotSettings, BotState, EnvConfig


class FakeTelegramResponse:
    def __init__(self, message_id):
        self.message_id = message_id

    def raise_for_status(self):
        return None

    def json(self):
        return {"ok": True, "result": {"message_id": self.message_id}}


class FakeTelegramSession:
    def __init__(self):
        self.posts = []

    def post(self, url, data, timeout):
        self.posts.append({"url": url, "data": data, "timeout": timeout})
        return FakeTelegramResponse(len(self.posts))


class FakeConflictResponse:
    status_code = 409

    def raise_for_status(self):
        raise requests.HTTPError("409 Client Error: Conflict", response=self)


class FakeConflictPollingSession:
    def __init__(self):
        self.gets = []

    def get(self, url, params, timeout):
        self.gets.append({"url": url, "params": params, "timeout": timeout})
        return FakeConflictResponse()


class FakeUpdatesResponse:
    def __init__(self, updates):
        self.updates = updates

    def raise_for_status(self):
        return None

    def json(self):
        return {"ok": True, "result": self.updates}


class FakeCommandSession(FakeTelegramSession):
    def __init__(self, updates):
        super().__init__()
        self.updates = updates
        self.gets = []

    def get(self, url, params, timeout):
        self.gets.append({"url": url, "params": params, "timeout": timeout})
        return FakeUpdatesResponse(self.updates)


class TelegramMessageSplittingTests(unittest.TestCase):
    def test_long_markdown_message_splits_on_lines_before_raw_chunks(self):
        session = FakeTelegramSession()
        config = EnvConfig("key", "secret", "token", "chat")
        settings = BotSettings(3600, -20, 20, True, (0, 5))
        message = f"`{'a' * 30}`\n`{'b' * 30}`"

        with patch.object(telegram, "TELEGRAM_MAX_MESSAGE", 50):
            telegram.send_telegram_message(session, config, settings, message, force_send=True)

        posted_texts = [post["data"]["text"] for post in session.posts]
        self.assertEqual(posted_texts, [f"`{'a' * 30}`\n", f"`{'b' * 30}`"])
        self.assertTrue(all(text.count("`") % 2 == 0 for text in posted_texts))

    def test_short_message_still_posts_once(self):
        session = FakeTelegramSession()
        config = EnvConfig("key", "secret", "token", "chat")
        settings = BotSettings(3600, -20, 20, True, (0, 5))

        telegram.send_telegram_message(session, config, settings, "`hello`", force_send=True)

        self.assertEqual(len(session.posts), 1)
        self.assertEqual(session.posts[0]["data"]["text"], "`hello`")

    def test_oversized_single_line_chunks_with_unbalanced_markdown_disable_parse_mode(self):
        session = FakeTelegramSession()
        config = EnvConfig("key", "secret", "token", "chat")
        settings = BotSettings(3600, -20, 20, True, (0, 5))
        message = f"`{'a' * 80}`"

        with patch.object(telegram, "TELEGRAM_MAX_MESSAGE", 40):
            telegram.send_telegram_message(session, config, settings, message, force_send=True)

        odd_backtick_posts = [
            post for post in session.posts if post["data"]["text"].count("`") % 2 == 1
        ]
        self.assertTrue(odd_backtick_posts)
        self.assertTrue(all("parse_mode" not in post["data"] for post in odd_backtick_posts))


class TelegramCommandPollingTests(unittest.TestCase):
    def test_polling_conflict_disables_command_polling(self):
        session = FakeConflictPollingSession()
        config = EnvConfig("key", "secret", "token", "chat")
        settings = BotSettings(3600, -20, 20, True, (0, 5))
        state = BotState(3600, True, -20, 20, (0, 5), last_update_id=123)

        update_id = commands.check_telegram_commands(
            session,
            config,
            settings,
            state,
            state.last_update_id,
            poll_timeout=30,
        )

        self.assertEqual(update_id, 123)
        self.assertFalse(state.telegram_command_polling_enabled)
        self.assertEqual(len(session.gets), 1)

    def test_status_command_tracks_lower_positive_futures_pnl_after_reset(self):
        updates = [
            {
                "update_id": 124,
                "message": {"text": "/status", "chat": {"id": "chat"}},
            }
        ]
        session = FakeCommandSession(updates)
        config = EnvConfig("key", "secret", "token", "chat")
        settings = BotSettings(3600, -20, 20, True, (0, 5))
        state = BotState(
            3600,
            True,
            -20,
            20,
            (0, 5),
            last_update_id=123,
            max_pnl=100.0,
            min_pnl=100.0,
        )

        with patch.object(portfolio, "get_futures_pnl", return_value=50.0):
            with patch.object(portfolio, "get_spot_balance", return_value={"total": 0.0, "breakdown": []}):
                with patch.object(command_handlers, "persist_runtime_state") as persist_state:
                    update_id = commands.check_telegram_commands(
                        session,
                        config,
                        settings,
                        state,
                        state.last_update_id,
                        poll_timeout=30,
                    )

        self.assertEqual(update_id, 124)
        self.assertEqual(state.max_pnl, 100.0)
        self.assertEqual(state.min_pnl, 50.0)
        persist_state.assert_called_once()

    def test_status_command_includes_structured_futures_trade_details(self):
        updates = [
            {
                "update_id": 124,
                "message": {"text": "/status", "chat": {"id": "chat"}},
            }
        ]
        session = FakeCommandSession(updates)
        config = EnvConfig("key", "secret", "token", "chat")
        settings = BotSettings(3600, -20, 20, True, (0, 5))
        state = BotState(
            3600,
            True,
            -20,
            20,
            (0, 5),
            last_update_id=123,
            max_pnl=50.0,
            min_pnl=-10.0,
        )
        futures_payload = {
            "total": 12.5,
            "open_positions": [{"symbol": "BTCUSDT", "unrealized_pnl": 12.5}],
            "closed_trades": [{"symbol": "ETHUSDT", "pnl": -2.25}],
        }

        with patch.object(portfolio, "get_futures_pnl", return_value=futures_payload):
            with patch.object(portfolio, "get_spot_balance", return_value={"total": 0.0, "breakdown": []}):
                commands.check_telegram_commands(
                    session,
                    config,
                    settings,
                    state,
                    state.last_update_id,
                    poll_timeout=30,
                )

        message = session.posts[0]["data"]["text"]
        self.assertIn("Current PnL", message)
        self.assertIn("12.50 USDT", message)
        self.assertIn("Open Positions", message)
        self.assertIn("BTCUSDT", message)
        self.assertIn("Latest Closed Positions", message)
        self.assertIn("ETHUSDT", message)
        self.assertNotIn("{'total'", message)


if __name__ == "__main__":
    unittest.main()
