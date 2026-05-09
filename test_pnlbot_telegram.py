import unittest
from unittest.mock import patch

import PnLBot


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


if __name__ == "__main__":
    unittest.main()
