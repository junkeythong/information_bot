# PnL Bot

A Telegram-enabled helper that fetches Binance Futures unrealized PnL on a schedule, sends alert notifications, and reports host resource usage. The bot respects quiet hours, persists key runtime configuration, and supports a lightweight TODO list.

## Prerequisites

- Python 3.9+
- Binance Futures API key and secret with read-only access
- Telegram bot token and target chat ID

## Configuration

All runtime options are still controlled with environment variables (defaults shown in parentheses),
but any value can also be inspected or overridden live with the `/config` Telegram command. Values
set at runtime are persisted to the state file so they survive restarts.

- `API_KEY`, `API_SECRET` – Binance credentials (required)
- `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` – Telegram bot credentials (required)
- `OPENAI_ADMIN_KEY` – OpenAI org admin key enabling `/openai` usage reports (optional)
- `PNL_BOT_DEFAULT_INTERVAL_SECONDS` (900) – default alert interval in seconds
- `PNL_BOT_DEFAULT_PNL_ALERT_LOW` (-20) – lower unrealized PnL alert threshold
- `PNL_BOT_DEFAULT_PNL_ALERT_HIGH` (20) – upper unrealized PnL alert threshold
- `PNL_BOT_DEFAULT_NIGHT_MODE_ENABLED` (true) – enable quiet hours on start
- `PNL_BOT_NIGHT_MODE_START_HOUR` (0) – start of quiet window (0-23)
- `PNL_BOT_NIGHT_MODE_END_HOUR` (5) – end of quiet window (1-24)

Set the variables in your shell or an `.env` file before launching the bot.

## Runtime Behavior

- Uses a retry-enabled `requests.Session` for Binance and Telegram APIs
- Long-polls Telegram for commands and updates `update_id` tracking automatically
- Persists editable runtime settings (interval, thresholds, run state, PnL bounds) to the JSON state file whenever they change
- Night mode can span midnight (e.g., 22 to 6) and sends start/end notices even during quiet hours
- CPU/RAM/disk alert thresholds are hardcoded and included in status messages
- When `OPENAI_ADMIN_KEY` is supplied, the bot refreshes OpenAI month-to-date cost/usage in the background; `/openai` forces a new fetch and includes the latest API window end time

## Telegram Commands

**Information**

- `/status` – full status report including system metrics
- `/pnl` – fetch the latest unrealized PnL immediately
- `/spot` – fetch the current Spot wallet USDT balance
- `/uptime` – show the running time since launch
- `/sysinfo` – display current CPU, RAM, and disk usage
- `/showtodo` – display the TODO list contents
- `/openai` – report OpenAI month-to-date cost, usage, and last month total (`/openaiusage` alias)
- `/help` – command reference

**Configuration & Actions**

- `/config show|get|set` – inspect or update any runtime parameter
- `/setinterval <seconds>` – update the reporting interval (10-3600)
- `/setlimit <min> <max>` – update unrealized PnL alert bounds
- `/nightmode on|off` – toggle quiet hours
- `/start`, `/stop` – resume or pause automatic monitoring
- `/todo <text>` – append to the local TODO list

## Files & Constants

- PnL bot script: `PnLBot.py`
- State snapshot: `pnl-bot-state.json` (hardcoded)
- TODO entries: `pnl-bot-todo-db.txt` (hardcoded)
- Timezone: `Asia/Ho_Chi_Minh` (hardcoded)
- System Alerts: CPU/RAM 80%, Disk 90% (hardcoded)
- Telegram: Poll 25s, Max Msg 4096 (hardcoded)
- OpenAI Refresh: 300s (hardcoded)

## Quick Start

1. Export required environment variables (see Configuration).
2. Install dependencies: `pip install -r requirements.txt` (ensure `requests`, `psutil`, `pytz` are available).
3. Run the bot: `python PnLBot.py`.
4. Send `/status` from the configured Telegram chat to confirm connectivity.

Extend the script by customizing thresholds, integrating detailed trade reports, or adjusting persistence paths through the provided environment variables.
