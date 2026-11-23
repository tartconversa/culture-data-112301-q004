"""测试共用的合成夹具：在内存中构造小规模资料与面板。"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from tourflow.materials import Materials
from tourflow.panel import RoutePanel


def make_materials(tables: dict) -> Materials:
    return Materials(root=Path("."), fingerprint="test-fingerprint", tables=tables)


def base_tables():
    """最小可用的静态结构：3 场馆 1 线路 1 车次。"""
    return {
        "sites": [
            {"id": "S0", "daily_capacity": 50, "protected_limit": 30, "region": "R0"},
            {"id": "S1", "daily_capacity": 60, "protected_limit": 40, "region": "R0"},
            {"id": "S2", "daily_capacity": 70, "protected_limit": 45, "region": "R1"},
        ],
        "routes": [
            {"id": "line-a", "sites": ["S0", "S1"], "bus": "B0", "bus_seats": 25},
        ],
        "campaigns": [
            {"id": "C0", "origin": "C0", "start": "2026-04-10", "routes": ["line-a"],
             "spillover_origins": ["C1"], "intensity": 1.0},
        ],
        "daily_panel": [],
        "itinerary_survey": [],
        "supply": [],
        "group_requests": [],
        "obligations": [],
        "later_events": [],
    }


def make_route_panel(
    rows: list[dict],
    *,
    route: str = "line-a",
    origins: tuple[str, ...] | None = None,
    dates: tuple[dt.date, ...] | None = None,
) -> RoutePanel:
    """从行字典直接构造分析面板（绕过 daily_panel 明细）。"""
    frame = pd.DataFrame(rows)
    if origins is None:
        origins = tuple(sorted(frame["origin"].unique()))
    if dates is None:
        dates = tuple(sorted(frame["on"].unique()))
    return RoutePanel(frame=frame, route=route, origins=origins, dates=dates)


def panel_row(origin, on, visits, *, start, promoted=True, censored=False,
              closed=False, exposed=False, holiday=False, rain=0.0):
    return {
        "origin": origin,
        "on": on,
        "visits": visits,
        "censored": censored,
        "closed_any": closed,
        "rain_max": rain,
        "holiday": holiday,
        "coverage_min": 1.0,
        "start": start,
        "route_promoted": promoted,
        "treated": promoted and on >= start,
        "exposed": exposed,
        "ever_treated_route": promoted,
    }
