# PnL Bot

A Telegram-enabled helper that fetches Binance Futures unrealized PnL on a schedule, sends alert notifications, and reports host resource usage. The bot respects quiet hours, persists key runtime configuration, and supports a lightweight TODO list.

It aims to be a simple bot and open for any integration, not stop here.

## Prerequisites

- Python 3.9+
- Binance Futures API key and secret with read-only access
- Telegram bot token and target chat ID
- `lunar-vn` library (installed via pip)

## Configuration

All runtime options are still controlled with environment variables (defaults shown in parentheses),
but any value can also be inspected or overridden live with the `/config` Telegram command. Values
set at runtime are persisted to the state file so they survive restarts.

- `API_KEY`, `API_SECRET` – Binance credentials (required)
- `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` – Telegram bot credentials (required)
- `OPENAI_ADMIN_KEY` – OpenAI org admin key enabling `/openai` usage reports (optional)
- `IQAIR_API_KEY` – IQAir API key for air quality monitoring (optional, get free key at [IQAir Dashboard](https://www.iqair.com/dashboard/api))
- `IQAIR_LATITUDE` – Latitude for air quality monitoring (default: 10.8231 - Ho Chi Minh City)
- `IQAIR_LONGITUDE` – Longitude for air quality monitoring (default: 106.6297 - Ho Chi Minh City)
- `PNL_BOT_DEFAULT_INTERVAL_SECONDS` (900) – default alert interval in seconds (max 86400/24h)
- `PNL_BOT_DEFAULT_PNL_ALERT_LOW` (-20) – lower unrealized PnL alert threshold
- `PNL_BOT_DEFAULT_PNL_ALERT_HIGH` (20) – upper unrealized PnL alert threshold
- `PNL_BOT_DEFAULT_NIGHT_MODE_ENABLED` (true) – enable quiet hours on start
- `PNL_BOT_NIGHT_MODE_START_HOUR` (0) – start of quiet window (0-23)
- `PNL_BOT_NIGHT_MODE_END_HOUR` (5) – end of quiet window (1-24)
- `PNL_BOT_INIT_CAPITAL` (0) – initial capital for PnL % calculation (optional)
- `PNL_BOT_OUTAGE_STREET_FILTER` – only report/alert power outages containing this street/area string (optional, e.g., `Phan Dinh Phung`)
- `PNL_BOT_EVN_MADVI` – Ma Don Vi for EVN outage check (optional, default: `PB0100`)
- `PNL_BOT_EVN_AREA_NAME` – Human readable area name for outages (optional, default: `Ho Chi Minh`)
- `PNL_BOT_TIMEZONE` – System timezone (optional, default: `Asia/Ho_Chi_Minh`)
- `PNL_BOT_CPU_ALERT_THRESHOLD` – CPU usage alert threshold (optional, default: `80`)
- `PNL_BOT_MEM_ALERT_THRESHOLD` – RAM usage alert threshold (optional, default: `80`)
- `PNL_BOT_DISK_ALERT_THRESHOLD` – Disk usage alert threshold (optional, default: `90`)

Set the variables in your shell or an `.env` file before launching the bot.

## Runtime Behavior

- Uses a retry-enabled `requests.Session` for Binance and Telegram APIs
- Long-polls Telegram for commands and updates `update_id` tracking automatically
- Persists editable runtime settings (interval, thresholds, run state, PnL and Spot bounds) to the JSON state file whenever they change
- Hides Spot and Futures sections in the notification loop if their respective balance or PnL is zero
- Night mode can span midnight (e.g., 22 to 6) and sends start/end notices even during quiet hours
- CPU/RAM/disk alert thresholds are hardcoded and checked in near real-time via a background worker thread. Alert thresholds are only displayed on alert or via `/sysinfo`
- When `OPENAI_ADMIN_KEY` is supplied, the bot refreshes OpenAI month-to-date cost in the background
- When `IQAIR_API_KEY` is configured, air quality index (AQI) is included in monitoring loop notifications
- **Power Outages**: Can fetch schedules for the configured area from EVN SPC via the `/outage` command. Supports filtering by street using `PNL_BOT_OUTAGE_STREET_FILTER`.
- Status message is organized into **Status** (Uptime, Lunar Date, TODO count, Init Capital, Config), **Spot Balance** (including ranges, token prices, and PnL %), and **Futures PnL** sections
- Sends a simplified daily status report (**Lunar date** and **Spot PnL**) at 8:00 AM, which is automatically **pinned**.

## Telegram Commands

**Information**

- `/status` – Comprehensive snapshot (PnL, Spot, Config)
- `/pnl` – fetch the latest unrealized PnL immediately
- `/spot` – fetch spot wallet breakdown
- `/aqi` – fetch current air quality index (requires IQAir API key)
- `/sysinfo` – display host CPU, RAM, and disk utilization
- `/openai` – report OpenAI Month-to-Date and Last Month costs
- `/showtodo` – display the TODO list contents
- `/outage` – display upcoming power outage schedule for the configured area
- `/help` – command reference

**Configuration & Actions**

- `/config show` – View all runtime parameters
- `/config set <key> <value>` – Update a parameter (interval, limits, bot state, or historical min/max bounds)
- `/start`, `/stop` – Resume or pause automatic monitoring alerts
- `/todo <text>` – append to the local TODO list
- `/spot reset` – reset clear min/max history

## Example Outputs

### /status
Comprehensive bot and portfolio snapshot:
```text
🧭 Status:
• Running: `True`
• Interval: `15.0m`
• Night mode: `True` (active: `False`)
• Alert limit: `-20 USDT ~ 100 USDT`
• Uptime: `24h,12m,5s`
• Lunar: `Mùng 3 Tháng 2 Năm Bính Ngọ`
• TODO Left: `3`
• Init Capital: `5,000.00 USDT`

💰 *Spot Balance:*
• Total: `5,420.50 USDT` (+8.41%)
• Max: `5,600.00 USDT`, Min: `5,200.00 USDT`
  ▫️ `BTC`: `3,200.00 USDT` @ 98,500.2500
  ▫️ `ETH`: `1,500.00 USDT` @ 2,650.1000
  ▫️ `SOL`: `720.50 USDT` @ 165.4500

📊 *Futures PnL:*
• Current PnL: `125.40 USDT`
• Max PnL: `250.00 USDT`, Min: `-40.00 USDT`
```

### /spot
Detailed spot wallet breakdown:
```text
💰 *Spot Balance:* `5,420.50 USDT` (+8.41%)
📊 *Range:* `[5,200.00, 5,600.00]`

*Asset Breakdown:*
• `BTC`: `3,200.00 USDT` @ 98,500.2500
• `ETH`: `1,500.00 USDT` @ 2,650.1000
• `SOL`: `720.50 USDT` @ 165.4500
```

### /aqi
Current air quality information:
```text
🟡 *Air Quality - Ho Chi Minh City*
• AQI (US): `85` - Moderate
• Temperature: `28°C`
• Humidity: `75%`
```

## Files & Constants

- PnL bot script: `PnLBot.py`
- State snapshot: `pnl-bot-state.json` (hardcoded)
- TODO entries: `pnl-bot-todo-db.txt` (hardcoded)
- Timezone: Configurable via `PNL_BOT_TIMEZONE` (default: `Asia/Ho_Chi_Minh`)
- System Alerts: Configurable via `PNL_BOT_*_ALERT_THRESHOLD`
- Telegram: Poll interval and max message size are managed automatically

## Quick Start

1. Export required environment variables (see Configuration).
   Example `.env` file:
   ```bash
   API_KEY=your_binance_key
   API_SECRET=your_binance_secret
   TELEGRAM_TOKEN=123456:ABC-DEF
   TELEGRAM_CHAT_ID=987654321
   PNL_BOT_INIT_CAPITAL=1000.0
   PNL_BOT_OUTAGE_STREET_FILTER="Phan Chu Trinh"
   IQAIR_API_KEY=your_iqair_key
   ```
2. Install dependencies: `pip install -r requirements.txt` (ensure `requests`, `psutil`, `pytz`, `lunar-vn` are available), should be installed in a venv directory.
3. Run the bot: `python PnLBot.py`.
4. Send `/status` from the configured Telegram chat to confirm connectivity.

Extend the script by customizing thresholds, integrating detailed trade reports, or anything you want.
