# PnL Bot

A Telegram bot for monitoring Binance Futures PnL, Spot balance, host health,
air quality, and optional EVN power outage schedules.

## Features

- Scheduled Telegram alerts for open Futures positions and daily Spot balance
- One-time monitor alerts when a newly closed Futures trade appears
- Per-position observed min/max open PnL with mark price, carried into latest closed trades when observed
- Portfolio status with `/status`, `/futures`, and `/spot`
- Runtime configuration with `/config`
- Manual AQI lookup and optional EVN outage reporting
- Host health check with `/sysinfo`
- Local JSON state persistence

## Quick Start

Requirements:

- Python 3.9+
- Binance API key and secret with read-only access
- Telegram bot token and target chat ID

Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp template/env.template .env
# edit .env

python3 main.py
```

Send `/status` from the configured Telegram chat to confirm the bot is running.

## Required Configuration

Set these values in `.env`:

```bash
API_KEY=...
API_SECRET=...
TELEGRAM_TOKEN=...
TELEGRAM_CHAT_ID=...
```

## Optional Configuration

Most optional settings are documented in `template/env.template`.

Common options:

| Variable | Purpose | Default |
| --- | --- | --- |
| `PNL_BOT_DEFAULT_INTERVAL_SECONDS` | Futures scheduled alert interval | `3600` |
| `PNL_BOT_DEFAULT_NIGHT_MODE_ENABLED` | Start with quiet hours enabled | `true` |
| `PNL_BOT_INIT_CAPITAL` | Spot PnL baseline | `0` |
| `IQAIR_API_KEY` | Enable manual `/aqi` reporting | unset |
| `PNL_BOT_OUTAGE_STREET_FILTER` | Filter EVN outage area | unset |

Values can also be inspected or changed from Telegram with `/config`.
Runtime changes are persisted to the local state file.

Runtime interval behavior:

- `interval_seconds` is the Futures polling interval.
- Telegram command polling is separate and remains responsive between Futures polls.
- Per-position Futures min/max PnL and price are observed only at bot poll times, not tick-by-tick.
- Spot is reported once daily at the daily 8:00 AM check, remains available on demand with `/spot` and `/status`, and silently keeps min/max balance history during the Futures loop.
- Spot min/max history includes an asset price snapshot and local observation time at each min/max point.
- The pinned daily message contains only the lunar date; holiday names appear before the lunar date.
- The Futures loop sends when open positions are detected.
- Newly closed Futures trades are deduped in the state file and sent once when first observed by the monitor loop.

## Telegram Commands

| Command | Description |
| --- | --- |
| `/status` | Full bot and portfolio status |
| `/futures` | Futures PnL, open positions with observed min/max, and latest closed positions |
| `/freqtrade logs <port>` | Last 100 Docker log lines for a monitored Freqtrade bot |
| `/spot` | Spot wallet summary |
| `/aqi` | Manual air quality report, when configured |
| `/outage` | EVN power outage schedule, when configured |
| `/sysinfo` | Host CPU, memory, disk, and temperature |
| `/config show` | Show runtime settings |
| `/config set <key> <value>` | Update a runtime setting |
| `/spot reset` | Reset Spot min/max history |
| `/start` | Resume scheduled alerts |
| `/stop` | Pause scheduled alerts |
| `/restart` | Restart `pnl.service` through systemd |
| `/help` | Show available commands |

## Example Outputs

### `/status`

```text
🧭 Status:
• Running: `True`
• Interval: `15.0m`
• Night mode: `True`
• Uptime: `24h,12m,5s`
• Lunar: `Mùng 3 Tháng 2 Năm Bính Ngọ`

💰 *Spot:*
• Init Capital: `5,000.00 USDT`
• Total: `5,420.50 USDT` 🟢 (+8.41%)
• Range:
  • Min: `5,200.00 USDT` (2026-06-28 12:00) (+4.00%)
    ▫️ `BTC`: `0.03248731` @ `98,500.2500` = `3,200.00 USDT`
    ▫️ `ETH`: `0.56601636` @ `2,650.1000` = `1,500.00 USDT`
  • Max: `5,600.00 USDT` (2026-06-29 12:00) (+12.00%)
    ▫️ `BTC`: `0.03500000` @ `100,000.0000` = `3,500.00 USDT`
    ▫️ `ETH`: `0.77777778` @ `2,700.0000` = `2,100.00 USDT`

💰 *Futures:*
• Current PnL: `125.40 USDT` 🟢
• Open Positions:
  ▫️ `BTCUSDT` LONG: `100.00 USDT` 🟢 @ `63,000`
    ▪️ Observed Open Min: `-12.00 USDT` 🔴 @ `61,500`
    ▪️ Observed Open Max: `100.00 USDT` 🟢 @ `63,000`
  ▫️ `ETHUSDT` SHORT: `25.40 USDT` 🟢 @ `3,310`
    ▪️ Observed Open Min: `-4.50 USDT` 🔴 @ `3,520`
    ▪️ Observed Open Max: `25.40 USDT` 🟢 @ `3,310`
• Latest Closed Positions:
  ▫️ `SOLUSDT`: `8.50 USDT` 🟢 (exit: `roi`)
    ▪️ Observed Open Min: `-1.20 USDT` 🔴 @ `142.50`
    ▪️ Observed Open Max: `10.30 USDT` 🟢 @ `149.25`
  ▫️ `BNBUSDT`: `-3.10 USDT` 🔴 (exit: `stop_loss`)
  ▫️ `ADAUSDT`: `0.00 USDT` ⚪
```

### `/spot`

```text
💰 *Spot:* `5,420.50 USDT` 🟢 (+8.41%)
📊 *Range:*
• Min: `5,200.00 USDT` (2026-06-28 12:00)
  ▫️ `BTC`: `0.03248731` @ `98,500.2500` = `3,200.00 USDT`
  ▫️ `ETH`: `0.56601636` @ `2,650.1000` = `1,500.00 USDT`
• Max: `5,600.00 USDT` (2026-06-29 12:00)
  ▫️ `BTC`: `0.03500000` @ `100,000.0000` = `3,500.00 USDT`
  ▫️ `ETH`: `0.77777778` @ `2,700.0000` = `2,100.00 USDT`

*Asset Breakdown:*
• `BTC`: `3,200.00 USDT` @ 98,500.2500
• `ETH`: `1,500.00 USDT` @ 2,650.1000
```

### `/futures`

```text
💰 *Futures:* `125.40 USDT` 🟢

*Open Positions:*
• `BTCUSDT` LONG: `100.00 USDT` 🟢 @ `63,000`
  ▫️ Observed Min: `-12.00 USDT` 🔴 @ `61,500`
  ▫️ Observed Max: `100.00 USDT` 🟢 @ `63,000`
• `ETHUSDT` SHORT: `25.40 USDT` 🟢 @ `3,310`
  ▫️ Observed Min: `-4.50 USDT` 🔴 @ `3,520`
  ▫️ Observed Max: `25.40 USDT` 🟢 @ `3,310`

*Latest Closed Positions:*
• `SOLUSDT`: `8.50 USDT` 🟢 (exit: `roi`)
  ▫️ Observed Open Min: `-1.20 USDT` 🔴 @ `142.50`
  ▫️ Observed Open Max: `10.30 USDT` 🟢 @ `149.25`
• `BNBUSDT`: `-3.10 USDT` 🔴 (exit: `stop_loss`)
• `ADAUSDT`: `0.00 USDT` ⚪
```

### `/aqi`

```text
🟡 *Air Quality - Ho Chi Minh City*
• AQI (US): `85` - `Moderate`
• Temperature: `28°C`
• Humidity: `75%`
```

## Run As A Service

Use `template/pnlbot.service.template` as a starting point for systemd.
Update the paths in the template, then enable the service.

## Development

Run tests:

```bash
python3 -m unittest discover -s tests -t . -p "test_*.py"
```

Check Python syntax:

```bash
python3 -m py_compile main.py pnlbot/*.py tests/*.py
```

## Runtime Files

The bot writes local runtime data to:

```text
pnl-bot-state.json
pnlbot.log
```

The state file stores runtime overrides, Spot min/max snapshots, observed Futures position ranges, and closed-trade dedupe keys. Delete or reset it only when you intentionally want to discard runtime state.

To use `/restart`, install the narrow sudoers rule from `template/pnlbot.sudoers`:

```bash
sudo install -m 0440 template/pnlbot.sudoers /etc/sudoers.d/pnlbot
sudo visudo -cf /etc/sudoers.d/pnlbot
```

The rule only allows `thonggia` to run `/usr/bin/systemctl restart pnl.service` without a password.
