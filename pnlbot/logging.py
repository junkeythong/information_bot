import os
import sys
import threading

from .constants import LOG_FILE_PATH, LOG_MAX_BYTES


class RotatingLogStream:
    encoding = "utf-8"

    def __init__(self, path: str, max_bytes: int) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.backup_path = f"{path}.1"
        self._lock = threading.RLock()
        self._stream = open(path, "a", encoding=self.encoding, buffering=1)
        self._rotate_if_needed(0)

    def write(self, text: str) -> int:
        if not text:
            return 0
        encoded_length = len(text.encode(self.encoding, errors="replace"))
        with self._lock:
            self._rotate_if_needed(encoded_length)
            return self._stream.write(text)

    def flush(self) -> None:
        with self._lock:
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            self._stream.close()

    def isatty(self) -> bool:
        return False

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        self._stream.seek(0, os.SEEK_END)
        if self._stream.tell() + incoming_bytes <= self.max_bytes:
            return
        self._stream.close()
        if os.path.exists(self.backup_path):
            os.remove(self.backup_path)
        if os.path.exists(self.path):
            os.replace(self.path, self.backup_path)
        self._stream = open(self.path, "a", encoding=self.encoding, buffering=1)


def configure_runtime_logging(path: str = LOG_FILE_PATH, max_bytes: int = LOG_MAX_BYTES) -> None:
    stream = RotatingLogStream(path, max_bytes)
    sys.stdout = stream
    sys.stderr = stream
