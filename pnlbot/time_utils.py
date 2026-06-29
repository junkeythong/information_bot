import datetime
import time

import pytz
from lunar_vn import can_chi, holidays, solar_to_lunar

from .models import BotState


def get_uptime(state: BotState) -> str:
    uptime_seconds = int(time.time() - state.start_time)

    months, remainder = divmod(uptime_seconds, 2592000)
    days, remainder = divmod(remainder, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if months > 0:
        parts.append(f"{months}mo")
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or (not parts):
        parts.append(f"{hours}h")
    if minutes > 0 or (not parts and hours == 0):
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return ",".join(parts)


def get_lunar_date_string(timezone_name: str) -> str:
    try:
        now = datetime.datetime.now(pytz.timezone(timezone_name))
        lunar = solar_to_lunar(now)
        day_str = f"Mùng `{lunar.day}`" if lunar.day <= 10 else f"`{lunar.day}`"
        month_str = f"Tháng `{lunar.month}`"
        if lunar.leap:
            month_str += " (Nhuận)"
        year_can_chi = can_chi.get_year_can_chi(lunar.year)
        holiday = holidays.get_holiday(now.date(), lunar)
        holiday_str = f" - `{holiday}`" if holiday else ""
        return f"{day_str} {month_str} Năm `{year_can_chi}`{holiday_str}"
    except Exception as exc:
        return f"Error: {exc}"


def should_send_daily_status(state: BotState, now: datetime.datetime) -> bool:
    return (
        state.is_running
        and now.hour == 8
        and state.last_lunar_alert_date != now.strftime("%Y-%m-%d")
    )


def should_send_daily_spot_report(state: BotState, now: datetime.datetime) -> bool:
    return (
        state.is_running
        and now.hour == 8
        and state.last_spot_report_date != now.strftime("%Y-%m-%d")
    )
