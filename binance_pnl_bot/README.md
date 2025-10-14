# PnL Bot

A Telegram-enabled helper that fetches Binance Futures unrealized PnL on a schedule, sends alert notifications, and reports host resource usage. The bot respects quiet hours, persists key runtime configuration, and supports a lightweight TODO list.

## Prerequisites

- Python 3.9+
- Binance Futures API key and secret with read-only access
- Telegram bot token and target chat ID

## Configuration

All runtime options can be controlled with environment variables. Defaults are shown in parentheses.

- `API_KEY`, `API_SECRET` – Binance credentials (required)
- `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` – Telegram bot credentials (required)
- `PNL_BOT_DEFAULT_INTERVAL_SECONDS` (900) – default alert interval in seconds
- `PNL_BOT_DEFAULT_PNL_ALERT_LOW` (-20) – lower unrealized PnL alert threshold
- `PNL_BOT_DEFAULT_PNL_ALERT_HIGH` (20) – upper unrealized PnL alert threshold
- `PNL_BOT_DEFAULT_NIGHT_MODE_ENABLED` (true) – enable quiet hours on start
- `PNL_BOT_NIGHT_MODE_START_HOUR` (0) – start of quiet window (0-23)
- `PNL_BOT_NIGHT_MODE_END_HOUR` (5) – end of quiet window (1-24)
- `PNL_BOT_TIMEZONE` (Asia/Ho_Chi_Minh) – IANA timezone for scheduling and uptime
- `PNL_BOT_TODO_FILE` (pnl-bot-todo-db.txt) – path for persisted TODO entries
- `PNL_BOT_TELEGRAM_MAX_MESSAGE` (4096) – message chunk size for Telegram
- `PNL_BOT_TELEGRAM_API_URL` (https://api.telegram.org) – Telegram API base URL
- `PNL_BOT_CPU_ALERT_THRESHOLD` (80) – CPU percentage that triggers system alerts
- `PNL_BOT_MEMORY_ALERT_THRESHOLD` (80) – RAM percentage that triggers system alerts
- `PNL_BOT_DISK_ALERT_THRESHOLD` (90) – disk percentage that triggers system alerts
- `PNL_BOT_TELEGRAM_POLL_TIMEOUT` (25) – long-poll duration when reading commands
- `PNL_BOT_STATE_FILE` (pnl-bot-state.json) – path used to persist runtime settings

Set the variables in your shell or an `.env` file before launching the bot.

## Runtime Behavior

- Uses a retry-enabled `requests.Session` for Binance and Telegram APIs
- Long-polls Telegram for commands and updates `update_id` tracking automatically
- Persists editable runtime settings (interval, thresholds, run state, PnL bounds) to the JSON state file whenever they change
- Night mode can span midnight (e.g., 22 to 6) and sends start/end notices even during quiet hours
- CPU/RAM/disk alert thresholds are configurable and included in status messages

## Telegram Commands

- `/setinterval <seconds>` – update the reporting interval (10-3600)
- `/setlimit <min> <max>` – update unrealized PnL alert bounds
- `/status` – full status report including system metrics
- `/start`, `/stop` – resume or pause automatic monitoring
- `/pnl` – fetch the latest unrealized PnL immediately
- `/nightmode on|off` – toggle quiet hours
- `/uptime` – show the running time since launch
- `/sysinfo` – display current CPU, RAM, and disk usage
- `/todo <text>` – append to the local TODO list
- `/showtodo` – display the TODO list contents
- `/help` – command reference

## Files Created

- PnL bot script: `ft_userdata/tools/PnLBot.py`
- State snapshot: `PNL_BOT_STATE_FILE` (default `pnl-bot-state.json`)
- TODO entries: `PNL_BOT_TODO_FILE` (default `pnl-bot-todo-db.txt`)

## Quick Start

1. Export required environment variables (see Configuration).
2. Install dependencies: `pip install -r requirements.txt` (ensure `requests`, `psutil`, `pytz` are available).
3. Run the bot: `python ft_userdata/tools/PnLBot.py`.
4. Send `/status` from the configured Telegram chat to confirm connectivity.

Extend the script by customizing thresholds, integrating detailed trade reports, or adjusting persistence paths through the provided environment variables.
