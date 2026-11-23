"""本地状态存取：承诺台账、分析运行登记与反馈。

state/ 已被 Git 排除；这里的状态是应用的本地事实来源，
所有写操作先写临时文件再替换，避免半截文件。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

LEDGER_VERSION = 1


def _empty_ledger() -> dict:
    return {
        "version": LEDGER_VERSION,
        "commitments": {},
        "holds": {},
        "events_applied": [],
        "outcomes": {},
        "feedback": {"show_up": {}, "excluded_cells": []},
    }


def _empty_runs() -> dict:
    return {"runs": []}


class StateStore:
    def __init__(self, state_dir: str | Path):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.dir / "ledger.json"
        self.runs_path = self.dir / "runs.json"

    def _read(self, path: Path, empty) -> dict:
        if not path.exists():
            return empty()
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, path: Path, payload: dict) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)

    def load_ledger(self) -> dict:
        ledger = self._read(self.ledger_path, _empty_ledger)
        if ledger.get("version") != LEDGER_VERSION:
            raise ValueError("台账版本不符，请备份后清空 state/ledger.json")
        return ledger

    def save_ledger(self, ledger: dict) -> None:
        self._write(self.ledger_path, ledger)

    def load_runs(self) -> dict:
        return self._read(self.runs_path, _empty_runs)

    def save_runs(self, runs: dict) -> None:
        self._write(self.runs_path, runs)
