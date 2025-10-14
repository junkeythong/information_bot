# information_bot

A small collection of Telegram-friendly utility bots I use to automate reporting and monitoring.

What’s inside
- OpenAI Cost Bot: Reports month-to-date and last-month OpenAI organization spend and usage to a Telegram chat. See `openai_cost_bot/README.md`.
- Binance PnL Bot: Monitors Binance Futures unrealized PnL and basic system health, with simple runtime controls via Telegram. See `binance_pnl_bot/README.md`.

How to use
- Each bot is self-contained. Open the subfolder’s README for required environment variables, setup, and run instructions.
- Typical flow: set env vars, install Python deps, then run or schedule the script.
