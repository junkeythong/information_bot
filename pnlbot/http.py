import os
from typing import Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .constants import SYSTEM_CA_BUNDLE_PATHS


def resolve_requests_verify_path() -> Union[bool, str]:
    if os.path.exists(requests.certs.where()):
        return True
    for path in SYSTEM_CA_BUNDLE_PATHS:
        if os.path.exists(path):
            print(f"certifi CA bundle not found; using system CA bundle: {path}", flush=True)
            return path
    return True


def create_retry_session() -> requests.Session:
    session = requests.Session()
    session.verify = resolve_requests_verify_path()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session
