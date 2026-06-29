import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class BotSettings:
    default_interval_seconds: int
    default_pnl_alert_low: int
    default_pnl_alert_high: int
    default_night_mode_enabled: bool
    night_mode_window: Tuple[int, int]
    init_capital: Optional[float] = None


@dataclass
class EnvConfig:
    api_key: str
    api_secret: str
    telegram_token: str
    telegram_chat_id: str
    iqair_api_key: Optional[str] = None
    iqair_latitude: float = 10.8231
    iqair_longitude: float = 106.6297
    outage_street_filter: Optional[str] = None
    evn_madvi: str = "PB0100"
    evn_area_name: str = "Ho Chi Minh"
    timezone: str = "Asia/Ho_Chi_Minh"
    cpu_alert_threshold: int = 80
    mem_alert_threshold: int = 80
    disk_alert_threshold: int = 90
    freqtrade_api_token: Optional[str] = None
    freqtrade_api_username: Optional[str] = None
    freqtrade_api_password: Optional[str] = None


@dataclass
class BotState:
    interval_seconds: int
    night_mode_enabled: bool
    pnl_alert_low: int
    pnl_alert_high: int
    night_mode_window: Tuple[int, int]
    is_running: bool = True
    last_update_id: Optional[int] = None
    max_spot_balance: float = 0.0
    min_spot_balance: float = 0.0
    max_spot_assets: List[dict] = field(default_factory=list)
    min_spot_assets: List[dict] = field(default_factory=list)
    max_spot_observed_at: Optional[str] = None
    min_spot_observed_at: Optional[str] = None
    init_capital: Optional[float] = None
    outage_street_filter: Optional[str] = None
    night_mode_active: bool = False
    telegram_command_polling_enabled: bool = True
    start_time: float = field(default_factory=time.time)
    power_outages: List[dict] = field(default_factory=list)
    last_outage_check: float = 0.0
    last_lunar_alert_date: Optional[str] = None
    last_spot_report_date: Optional[str] = None
    pinned_daily_message_id: Optional[int] = None
    freqtrade_ports: List[int] = field(default_factory=list)
    freqtrade_alert_cooldown_seconds: int = 300
    last_freqtrade_alert_time: float = 0.0
    futures_position_ranges: Dict[str, dict] = field(default_factory=dict)
    closed_position_ranges: List[dict] = field(default_factory=list)
    runtime_config_overrides: List[str] = field(default_factory=list)


@dataclass
class ConfigDefinition:
    description: str
    parser: Callable[[str, BotState, BotSettings], object]
    getter: Callable[[BotState, BotSettings], object]
    applier: Callable[[object, BotState, BotSettings], Optional[str]]
