import unittest
from unittest.mock import patch

import PnLBot
import requests


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


class TelegramMessageSplittingTests(unittest.TestCase):
    def test_long_markdown_message_splits_on_lines_before_raw_chunks(self):
        session = FakeTelegramSession()
        config = PnLBot.EnvConfig("key", "secret", "token", "chat")
        settings = PnLBot.BotSettings(3600, -20, 20, True, (0, 5))
        message = f"`{'a' * 30}`\n`{'b' * 30}`"

        with patch.object(PnLBot, "TELEGRAM_MAX_MESSAGE", 50):
            PnLBot.send_telegram_message(session, config, settings, message, force_send=True)

        posted_texts = [post["data"]["text"] for post in session.posts]
        self.assertEqual(posted_texts, [f"`{'a' * 30}`\n", f"`{'b' * 30}`"])
        self.assertTrue(all(text.count("`") % 2 == 0 for text in posted_texts))

    def test_short_message_still_posts_once(self):
        session = FakeTelegramSession()
        config = PnLBot.EnvConfig("key", "secret", "token", "chat")
        settings = PnLBot.BotSettings(3600, -20, 20, True, (0, 5))

        PnLBot.send_telegram_message(session, config, settings, "`hello`", force_send=True)

        self.assertEqual(len(session.posts), 1)
        self.assertEqual(session.posts[0]["data"]["text"], "`hello`")

    def test_oversized_single_line_chunks_with_unbalanced_markdown_disable_parse_mode(self):
        session = FakeTelegramSession()
        config = PnLBot.EnvConfig("key", "secret", "token", "chat")
        settings = PnLBot.BotSettings(3600, -20, 20, True, (0, 5))
        message = f"`{'a' * 80}`"

        with patch.object(PnLBot, "TELEGRAM_MAX_MESSAGE", 40):
            PnLBot.send_telegram_message(session, config, settings, message, force_send=True)

        odd_backtick_posts = [
            post for post in session.posts if post["data"]["text"].count("`") % 2 == 1
        ]
        self.assertTrue(odd_backtick_posts)
        self.assertTrue(all("parse_mode" not in post["data"] for post in odd_backtick_posts))


class TelegramCommandPollingTests(unittest.TestCase):
    def test_polling_conflict_disables_command_polling(self):
        session = FakeConflictPollingSession()
        config = PnLBot.EnvConfig("key", "secret", "token", "chat")
        settings = PnLBot.BotSettings(3600, -20, 20, True, (0, 5))
        state = PnLBot.BotState(3600, True, -20, 20, (0, 5), last_update_id=123)

        update_id = PnLBot.check_telegram_commands(
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


if __name__ == "__main__":
    unittest.main()
