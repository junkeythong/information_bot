# PnL Bot

A Telegram bot for monitoring Binance Futures PnL, Spot balance, host health,
air quality, and optional EVN power outage schedules.

## Features

- Scheduled Telegram alerts for Futures PnL and Spot balance
- Portfolio status with `/status`, `/futures`, and `/spot`
- Runtime configuration with `/config`
- Optional AQI and EVN outage reporting
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
| `PNL_BOT_DEFAULT_INTERVAL_SECONDS` | Scheduled alert interval | `3600` |
| `PNL_BOT_DEFAULT_PNL_ALERT_LOW` | Low PnL alert threshold | `-20` |
| `PNL_BOT_DEFAULT_PNL_ALERT_HIGH` | High PnL alert threshold | `20` |
| `PNL_BOT_DEFAULT_NIGHT_MODE_ENABLED` | Start with quiet hours enabled | `true` |
| `PNL_BOT_INIT_CAPITAL` | Spot PnL baseline | `0` |
| `IQAIR_API_KEY` | Enable AQI reporting | unset |
| `PNL_BOT_OUTAGE_STREET_FILTER` | Filter EVN outage area | unset |

Values can also be inspected or changed from Telegram with `/config`.
Runtime changes are persisted to the local state file.

## Telegram Commands

| Command | Description |
| --- | --- |
| `/status` | Full bot and portfolio status |
| `/futures` | Current Futures PnL |
| `/spot` | Spot wallet summary |
| `/aqi` | Air quality report, when configured |
| `/outage` | EVN power outage schedule, when configured |
| `/sysinfo` | Host CPU, memory, disk, and temperature |
| `/config show` | Show runtime settings |
| `/config set <key> <value>` | Update a runtime setting |
| `/spot reset` | Reset Spot min/max history |
| `/futures reset` | Reset Futures min/max history |
| `/start` | Resume scheduled alerts |
| `/stop` | Pause scheduled alerts |
| `/help` | Show available commands |

## Run As A Service

Use `template/pnlbot.service.template` as a starting point for systemd.
Update the paths in the template, then enable the service.

## Development

Run tests:

```bash
python3 -m unittest discover -s . -p "test_*.py"
```

Check Python syntax:

```bash
python3 -m py_compile main.py pnlbot/*.py test_*.py
```

## Runtime Files

The bot writes local runtime data to:

```text
pnl-bot-state.json
pnlbot.log
```
