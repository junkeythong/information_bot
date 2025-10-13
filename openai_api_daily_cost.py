#!/usr/bin/env python3
import os, requests, datetime as dt

# ===== ENV =====
OPENAI_KEY = os.environ["OPENAI_ADMIN_KEY"]
BOT_TOKEN  = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID    = os.environ["TELEGRAM_CHAT_ID"]

# ===== CONST =====
TZ = dt.timezone(dt.timedelta(hours=7))  # Asia/Ho_Chi_Minh
COSTS_URL = "https://api.openai.com/v1/organization/costs"
USAGE_URL = "https://api.openai.com/v1/organization/usage/completions"
HDRS = {"Authorization": f"Bearer {OPENAI_KEY}"}

# ===== TIME HELPERS =====
def epoch_utc(d: dt.datetime) -> int:
    return int(d.astimezone(dt.timezone.utc).timestamp())

def vn_now():
    return dt.datetime.now(TZ)

def vn_month_start(d: dt.datetime) -> dt.datetime:
    return dt.datetime(d.year, d.month, 1, tzinfo=TZ)

def vn_prev_month_range(d: dt.datetime):
    first_this = vn_month_start(d)
    last_prev_local = first_this - dt.timedelta(seconds=1)
    first_prev = dt.datetime(last_prev_local.year, last_prev_local.month, 1, tzinfo=TZ)
    end_prev   = dt.datetime(last_prev_local.year, last_prev_local.month, last_prev_local.day, 23, 59, 59, tzinfo=TZ)
    return first_prev, end_prev

# ===== COSTS (sum amount.value over window) =====
def sum_costs_epoch(start_epoch: int, end_epoch: int) -> float:
    # One page only (limit ≤ 31 for daily buckets)
    params = {
        "start_time": start_epoch,   # inclusive
        "end_time":   end_epoch,     # exclusive
        "limit":      31             # per-day buckets, max 31 for 1d windows
    }
    r = requests.get(COSTS_URL, params=params, headers=HDRS, timeout=30)
    r.raise_for_status()
    js = r.json()

    total = 0.0
    for bucket in js.get("data", []):
        rows = bucket.get("results") or bucket.get("result") or []
        for row in rows:
            amt = (row.get("amount") or {}).get("value", 0)
            try:
                total += float(amt or 0)
            except Exception:
                pass
    return total

# ===== USAGE (sum requests + tokens over window) =====
def sum_usage_epoch(start_epoch: int, end_epoch: int):
    params = {
        "start_time":   start_epoch,     # inclusive
        "end_time":     end_epoch,       # exclusive
        "bucket_width": "1d",
        "limit":        31               # <= 31 for daily buckets
    }
    r = requests.get(USAGE_URL, params=params, headers=HDRS, timeout=30)
    r.raise_for_status()
    js = r.json()

    total_requests = 0
    total_tokens   = 0
    for bucket in js.get("data", []):
        rows = bucket.get("results") or bucket.get("result") or []
        for row in rows:
            # Defensive: fields may be absent if zero
            reqs    = int(row.get("num_model_requests", 0) or 0)
            in_txt  = int(row.get("input_tokens", 0) or 0)
            out_txt = int(row.get("output_tokens", 0) or 0)

            total_requests += reqs
            total_tokens   += in_txt + out_txt

            # If input_tokens_details.cached_tokens exists, include it
            itd = row.get("input_tokens_details") or {}
            cached = int(itd.get("cached_tokens", 0) or 0)
            total_tokens += cached

    return total_requests, total_tokens

# ===== WINDOWS =====
now_local = vn_now()
# Month-to-date: [start of month VN, now UTC)
m_start_vn = vn_month_start(now_local)
m_start = epoch_utc(m_start_vn)
# end_time is exclusive → add +1s to ensure "now" is included
m_end   = int(dt.datetime.now(dt.timezone.utc).timestamp()) + 1

# Last month (VN)
lm_start_vn, lm_end_vn = vn_prev_month_range(now_local)
lm_start = epoch_utc(lm_start_vn)
lm_end   = epoch_utc(lm_end_vn) + 1  # exclusive

# ===== FETCH =====
mtd_cost = sum_costs_epoch(m_start, m_end)
lm_cost  = sum_costs_epoch(lm_start, lm_end)
mtd_requests, mtd_tokens = sum_usage_epoch(m_start, m_end)

# ===== MESSAGE =====
title = "📊 OpenAI Month-to-Date\n"
line1 = f"• Period: {m_start_vn.strftime('%Y-%m-%d')} → now"
line2 = f"• Cost: ${mtd_cost:,.4f}"
line3 = f"• Requests: {mtd_requests:,}"
line4 = f"• Tokens: {mtd_tokens:,}"
line5 = f"• Last Month Total: ${lm_cost:,.4f} ({lm_start_vn.strftime('%Y-%m-%d')} → {lm_end_vn.strftime('%Y-%m-%d')})"
msg = "\n".join([title, line1, line2, line3, line4, line5])

# ===== SEND TO TELEGRAM =====
tg_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
tg_payload = {"chat_id": CHAT_ID, "text": msg, "disable_web_page_preview": True}
resp = requests.post(tg_url, data=tg_payload, timeout=30)
resp.raise_for_status()
print("Sent:\n", msg)
