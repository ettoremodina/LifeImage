"""Summary: Simple CSV metrics logger for fitness and population statistics."""
from __future__ import annotations
from pathlib import Path
import csv
from typing import Dict, Any


class MetricsLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer = None
        self._header_written = False

    def log(self, row: Dict[str, Any]):
        if self._file is None:
            self._file = self.path.open('w', newline='', encoding='utf-8')
            import csv as _csv
            self._writer = _csv.writer(self._file)
        if not self._header_written:
            self._writer.writerow(list(row.keys()))
            self._header_written = True
        self._writer.writerow(list(row.values()))
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
