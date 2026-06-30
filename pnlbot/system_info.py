import ipaddress
import time
from typing import Optional, Tuple

import psutil
import requests

from .models import BotSettings, EnvConfig


def get_top_processes(n: int = 5) -> Tuple[str, str]:
    processes = []
    sampled_procs = []

    for proc in psutil.process_iter(attrs=["pid", "name"]):
        try:
            proc.cpu_percent(None)
            sampled_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    time.sleep(0.2)

    for proc in sampled_procs:
        try:
            processes.append(
                {
                    "pid": proc.info["pid"],
                    "name": proc.info.get("name") or "unknown",
                    "cpu_percent": proc.cpu_percent(None),
                    "memory_percent": proc.memory_percent(),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    top_cpu = sorted(processes, key=lambda x: x["cpu_percent"], reverse=True)[:n]
    top_mem = sorted(processes, key=lambda x: x["memory_percent"], reverse=True)[:n]

    cpu_info = "\n".join(
        [f"• `{proc['name']}` (PID `{proc['pid']}`): `{proc['cpu_percent']}%` CPU" for proc in top_cpu]
    )
    mem_info = "\n".join(
        [f"• `{proc['name']}` (PID `{proc['pid']}`): `{round(proc['memory_percent'], 1)}%` RAM" for proc in top_mem]
    )
    return cpu_info, mem_info


PUBLIC_IP_URLS = (
    "https://api.ipify.org",
    "https://ifconfig.me/ip",
    "https://icanhazip.com",
)


def _normalize_public_ip(value: object) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        return None


def get_public_ip(session: requests.Session, *, timeout: int = 10) -> Optional[str]:
    for url in PUBLIC_IP_URLS:
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
        except Exception:
            continue

        ip = _normalize_public_ip(response.text)
        if ip:
            return ip
    return None


def get_system_info_text(config: EnvConfig, settings: BotSettings, show_all: bool = False) -> Optional[str]:
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent

    is_alert = (
        cpu > config.cpu_alert_threshold
        or mem > config.mem_alert_threshold
        or disk > config.disk_alert_threshold
    )
    if not show_all and not is_alert:
        return None

    info_lines = []
    title = "*🖥 System Alert:*" if is_alert else "*📊 Current system metrics:*"
    info_lines.append(title)

    def format_line(label, val, thresh):
        exceeded = val > thresh
        if exceeded:
            return f"🔴 *{label}: `{val}%` (alert when > `{thresh}%`)*"
        return f"• {label}: `{val}%` (alert when > `{thresh}%`)"

    for label, val, thresh in [
        ("CPU", cpu, config.cpu_alert_threshold),
        ("RAM", mem, config.mem_alert_threshold),
        ("Disk", disk, config.disk_alert_threshold),
    ]:
        if show_all or val > thresh:
            info_lines.append(format_line(label, val, thresh))

    top_cpu, top_mem = get_top_processes()
    info_lines.append("\n*⚙️ Top CPU processes:*\n" + top_cpu)
    info_lines.append("\n*💾 Top RAM processes:*\n" + top_mem)

    return "\n".join(info_lines)
