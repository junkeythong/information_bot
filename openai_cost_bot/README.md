# OpenAI Cost Bot

Reports OpenAI organization month-to-date spend and usage to a Telegram chat. Also includes the previous month’s total for comparison.

What it does
- Aggregates costs via `v1/organization/costs` with daily buckets.
- Aggregates usage via `v1/organization/usage/completions` (requests and tokens).
- Formats a brief message and posts to Telegram.

Prerequisites
- Python 3.9+
- Environment variables set: `OPENAI_ADMIN_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

Run
- `python openai_api_daily_cost.py`
- Schedule via cron, e.g.: `0 9 * * * /usr/bin/python /path/openai_api_daily_cost.py`

Notes
- Times use Asia/Ho_Chi_Minh for month windows; requests sent in UTC epoch.
- End timestamps are exclusive; the script adds +1s to include “now”.
