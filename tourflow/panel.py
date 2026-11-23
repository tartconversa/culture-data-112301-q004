"""把客源地×场馆×日面板整理为客源地×日×线路的分析面板。

线路层面的一次到访消耗线路全部场馆，因此分析单位取客源地×日×线路：
visits 为线路各场馆到访次数之和（到访次数，不是唯一游客）。
售罄与关闭的口径在此固定：
- 线路任一场馆当日关闭 → 该客源地×日×线路不可用（机械为零，非需求）；
- 任一场馆售罄 → 观测被右截尾，真实需求不小于观测值。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from .materials import Materials
from .model import Structure, parse_date


@dataclass(frozen=True)
class RoutePanel:
    frame: pd.DataFrame  # 列见 build_route_panel
    route: str
    origins: tuple[str, ...]
    dates: tuple[dt.date, ...]


def build_route_panel(materials: Materials, structure: Structure, route: str) -> RoutePanel:
    if route not in structure.routes:
        raise ValueError("未知线路 " + route)
    sites = set(structure.routes[route].sites)
    exposure = structure.spillover_exposure()

    rows: dict[tuple[str, str], dict] = {}
    for record in materials.table("daily_panel"):
        if record["site"] not in sites:
            continue
        key = (record["origin"], record["on"])
        cell = rows.setdefault(
            key,
            {
                "origin": record["origin"],
                "on": record["on"],
                "visits": 0,
                "censored": False,
                "closed_any": False,
                "rain_max": 0.0,
                "holiday": record["holiday"],
                "coverage_min": 1.0,
            },
        )
        cell["visits"] += record["visits"]
        cell["censored"] = cell["censored"] or record["sold_out"]
        cell["closed_any"] = cell["closed_any"] or record["closed"]
        cell["rain_max"] = max(cell["rain_max"], record["rain_mm"])
        cell["coverage_min"] = min(cell["coverage_min"], record["survey_coverage"])

    frame = pd.DataFrame(rows.values())
    frame["on"] = pd.to_datetime(frame["on"]).dt.date
    starts = {o: c.start for o, c in structure.campaign_by_origin.items()}
    promoted = {o: set(structure.promoted_routes(o)) for o in starts}
    frame["start"] = frame["origin"].map(starts)
    frame["route_promoted"] = frame["origin"].map(lambda o: route in promoted.get(o, set()))
    frame["treated"] = [
        promoted_flag and day >= start
        for promoted_flag, day, start in zip(frame["route_promoted"], frame["on"], frame["start"])
    ]
    frame["exposed"] = [
        origin in exposure and day >= exposure[origin]
        for origin, day in zip(frame["origin"], frame["on"])
    ]
    frame["ever_treated_route"] = frame["origin"].map(
        lambda o: route in promoted.get(o, set())
    )
    frame = frame.sort_values(["origin", "on"]).reset_index(drop=True)
    origins = tuple(sorted(frame["origin"].unique()))
    dates = tuple(sorted(frame["on"].unique()))
    return RoutePanel(frame=frame, route=route, origins=origins, dates=dates)
