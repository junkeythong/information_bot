import datetime
import hashlib
import html
import re
import unicodedata
from typing import List

import pytz
import requests

from .constants import EVN_SPC_OUTAGE_URL
from .models import EnvConfig


def remove_accents(input_str: str) -> str:
    if not input_str:
        return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    value = "".join([char for char in nfkd_form if not unicodedata.combining(char)])
    return value.replace('đ', 'd').replace('Đ', 'D')


def get_power_outages(session: requests.Session, config: EnvConfig) -> List[dict]:
    tz = pytz.timezone(config.timezone)
    now = datetime.datetime.now(tz)
    end_date = now + datetime.timedelta(days=7)

    params = {
        "madvi": config.evn_madvi,
        "tuNgay": now.strftime("%d-%m-%Y"),
        "denNgay": end_date.strftime("%d-%m-%Y"),
        "ChucNang": "MaDonVi",
    }

    try:
        response = session.get(EVN_SPC_OUTAGE_URL, params=params, timeout=10)
        response.raise_for_status()
        html_content = response.text

        blocks = re.findall(r'<div class="entry">(.*?)</div>\s*<br />', html_content, re.DOTALL)

        outages = []
        for block in blocks:
            area_match = re.search(r'class="where"><b>KHU VỰC:</b>\s*(.*?)</span>', block, re.DOTALL)
            time_match = re.search(r'class="time">.*?<span style="white-space:nowrap;">\s*(.*?)\s*</span>\s*</span>', block, re.DOTALL)
            reason_match = re.search(r'class="cause">.*?<span>(.*?)</span>\s*</span>', block, re.DOTALL)

            if area_match and time_match:
                area = html.unescape(area_match.group(1).strip())
                time_info = html.unescape(time_match.group(1).strip())
                time_info = re.sub(r'\s+', ' ', time_info)

                filter_str = config.outage_street_filter
                if filter_str:
                    normalized_area = remove_accents(area).lower()
                    normalized_filter = remove_accents(filter_str).lower()
                    if normalized_filter not in normalized_area:
                        continue

                reason = html.unescape(reason_match.group(1).strip()) if reason_match else "N/A"
                reason = re.sub(r'\s+', ' ', reason)

                outage_id = hashlib.md5(f"{area}{time_info}".encode("utf-8")).hexdigest()

                outages.append({
                    "id": outage_id,
                    "area": area,
                    "time": time_info,
                    "reason": reason
                })
        return outages
    except Exception as exc:
        print(f"Error fetching power outages: {exc}")
        return []
