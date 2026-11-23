"""抽样行程调查：唯一游客估计与跨馆去重。

纪律（对应资料契约）：
- 面板计数是到访次数，一人同日跨馆会在多个场馆各计一次；把跨馆到访
  相加当作新增人数是被禁止的，必须经调查去重换算。
- linked=false 的记录只有本次匿名身份，不支持跨次去重：唯一游客只能
  给出区间，不能假装是点估计。
- 没有观测到行程关联不等于确认是不同游客。
- 调查未覆盖的线路（本资料中 line-c 的场馆组合没有抽样行程）不做
  凭空换算，只给出无信息区间并标注识别缺口。
- 覆盖率与纳入概率共同决定 Horvitz–Thompson 权重：权重 = 1 / 纳入概率，
  再按覆盖率放大到总体。覆盖率不足 1 的场次与完全可观测场次分开报告。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from .materials import Materials
from .model import Structure, parse_date


@dataclass
class RouteTripStats:
    """单条线路在某客源地的行程去重依据。"""

    origin: str
    route: str
    covered: bool  # 调查是否覆盖该线路的场馆组合
    n_records: int
    linked_share: float | None
    visits_per_person: float | None  # 每人每日在该线路场馆的到访次数（HT 加权）
    visits_per_person_se: float | None
    fully_observable: bool  # 覆盖率为 1 的完全可观测场次
    notes: list[str] = field(default_factory=list)


@dataclass
class UniqueEstimate:
    """某客源地×线路×日期集合的唯一游客估计。"""

    point: float | None  # 可去重部分给出的点估计；无法估计时为 None
    low: float | None
    high: float | None
    fully_observable: bool
    linked_share: float | None
    notes: list[str] = field(default_factory=list)


def _matched_records(materials: Materials, structure: Structure, origin: str, route: str):
    sites = set(structure.routes[route].sites)
    matched = []
    for record in materials.table("itinerary_survey"):
        if record["origin"] != origin:
            continue
        if set(record["sites"]) <= sites:
            matched.append(record)
    return matched


def route_trip_stats(
    materials: Materials, structure: Structure, origin: str, route: str
) -> RouteTripStats:
    """估计每人每日线路到访次数（visits_per_person）及其不确定度。"""
    records = _matched_records(materials, structure, origin, route)
    coverage = {
        (r["origin"], r["on"]): r["survey_coverage"]
        for r in materials.table("daily_panel")
    }
    fully = bool(records) and all(
        coverage.get((origin, r["on"]), 0.0) >= 1.0 for r in records
    )
    if not records:
        n_sites = len(structure.routes[route].sites)
        return RouteTripStats(
            origin=origin,
            route=route,
            covered=False,
            n_records=0,
            linked_share=None,
            visits_per_person=None,
            visits_per_person_se=None,
            fully_observable=False,
            notes=[
                f"调查未覆盖该客源地在此线路的行程；每人每日到访次数只能取无信息区间 [1, {n_sites}]"
            ],
        )
    weights = np.array([1.0 / r["inclusion_probability"] for r in records])
    sizes = np.array([len(r["sites"]) for r in records], dtype=float)
    vpp = float(np.sum(weights * sizes) / np.sum(weights))
    # 对记录做朴素自助法，给出 vpp 的标准误
    rng = np.random.default_rng(20261123)
    reps = []
    for _ in range(500):
        idx = rng.integers(0, len(records), len(records))
        reps.append(float(np.sum(weights[idx] * sizes[idx]) / np.sum(weights[idx])))
    linked_share = float(np.mean([r["linked"] for r in records]))
    notes = []
    if not fully:
        notes.append("调查覆盖率不足 1，唯一游客为部分可观测估计，与完全可观测场次区分报告")
    return RouteTripStats(
        origin=origin,
        route=route,
        covered=True,
        n_records=len(records),
        linked_share=linked_share,
        visits_per_person=vpp,
        visits_per_person_se=float(np.std(reps, ddof=1)),
        fully_observable=fully,
        notes=notes,
    )


def unique_visitors(
    materials: Materials,
    structure: Structure,
    origin: str,
    route: str,
    dates: set[dt.date] | None = None,
) -> UniqueEstimate:
    """估计客源地×线路在指定日期集合内的唯一游客数。

    linked 记录按匿名人跨日去重（点估计）；unlinked 记录无法跨次去重，
    给出 [max(单日), Σ(单日)] 区间。两者相加得到总量区间。
    """
    records = _matched_records(materials, structure, origin, route)
    if dates is not None:
        records = [r for r in records if parse_date(r["on"]) in dates]
    if not records:
        return UniqueEstimate(
            point=None,
            low=None,
            high=None,
            fully_observable=False,
            linked_share=None,
            notes=["调查未覆盖，无法给出唯一游客估计"],
        )
    coverage = {
        (r["origin"], r["on"]): r["survey_coverage"]
        for r in materials.table("daily_panel")
    }

    def weight(record):
        cov = coverage.get((origin, record["on"]), 0.0)
        if cov <= 0:
            return 0.0
        return 1.0 / record["inclusion_probability"] / cov

    linked = [r for r in records if r["linked"]]
    unlinked = [r for r in records if not r["linked"]]
    # linked：按匿名人去重，每人计一次（权重取首次出现记录）
    seen: dict[str, float] = {}
    for record in linked:
        seen.setdefault(record["person"], weight(record))
    linked_unique = float(sum(seen.values()))
    # unlinked：单日人数按 HT 加权；跨日区间 [最大单日, 单日之和]
    by_day: dict[str, float] = {}
    for record in unlinked:
        by_day[record["on"]] = by_day.get(record["on"], 0.0) + weight(record)
    unlinked_low = max(by_day.values()) if by_day else 0.0
    unlinked_high = float(sum(by_day.values()))
    fully = all(
        coverage.get((origin, r["on"]), 0.0) >= 1.0 for r in records
    )
    linked_share = len(linked) / len(records)
    notes = []
    if unlinked:
        notes.append(
            "unlinked 记录不支持跨次去重：唯一游客为区间而非点估计；"
            "没有观测到行程关联不等于确认是不同游客"
        )
    if not fully:
        notes.append("覆盖率不足 1，估计依赖 HT 加权外推，与完全可观测场次区分")
    return UniqueEstimate(
        point=linked_unique if not unlinked else None,
        low=linked_unique + unlinked_low,
        high=linked_unique + unlinked_high,
        fully_observable=fully,
        linked_share=linked_share,
        notes=notes,
    )
