"""把双重差分与调查去重合并为线路级增量报告。

输出纪律：
- 每条线路给出归因判定：支持归因 / 证据不足 / 无法识别；
- 无法支持归因的增量只进入“待验证”，不能当成已证实成果；
- 唯一游客换算依赖调查覆盖，未覆盖线路只给区间并标注缺口。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .did import DidConfig, RouteDid, estimate_route
from .materials import Materials
from .model import Structure
from .panel import build_route_panel
from .survey import route_trip_stats

# 归因判定
SUPPORTED = "支持归因"
INSUFFICIENT = "证据不足"
UNIDENTIFIED = "无法识别"


@dataclass
class RouteIncrement:
    route: str
    verdict: str
    att_visits: float | None
    att_visits_ci: tuple[float, float] | None
    att_unique: tuple[float, float] | None  # 唯一游客增量区间
    att_unique_note: str
    did: RouteDid
    gaps: list[str] = field(default_factory=list)


@dataclass
class IncrementalityReport:
    created_at: str
    data_fingerprint: str
    config: dict
    routes: dict[str, RouteIncrement]
    sensitivity: dict[str, dict]  # 敏感性口径结果（保留截尾单元）
    gaps: list[str]

    def to_dict(self) -> dict:
        def route_dict(r: RouteIncrement) -> dict:
            return {
                "route": r.route,
                "verdict": r.verdict,
                "att_visits": r.att_visits,
                "att_visits_ci": list(r.att_visits_ci) if r.att_visits_ci else None,
                "att_unique": list(r.att_unique) if r.att_unique else None,
                "att_unique_note": r.att_unique_note,
                "cells_used": r.did.cells_used,
                "cells_unidentified": len(r.did.cells_unidentified),
                "censored_cells_dropped": r.did.censored_cells_dropped,
                "closed_cells_dropped": r.did.closed_cells_dropped,
                "n_treated_origins": r.did.n_treated_origins,
                "control_scope": r.did.control_scope,
                "pretrend": r.did.pretrend,
                "balance": r.did.balance,
                "spillover": r.did.spillover,
                "event_study": [
                    {
                        "event_time": p.event_time,
                        "estimate": p.estimate,
                        "n_origins": p.n_origins,
                        "ci": [p.ci_low, p.ci_high],
                    }
                    for p in r.did.event_study
                ],
                "notes": r.did.notes,
                "gaps": r.gaps,
            }

        return {
            "created_at": self.created_at,
            "data_fingerprint": self.data_fingerprint,
            "config": self.config,
            "routes": {k: route_dict(v) for k, v in self.routes.items()},
            "sensitivity": self.sensitivity,
            "gaps": self.gaps,
        }


def _verdict(did: RouteDid) -> tuple[str, list[str]]:
    """归因判定：不能识别 > 证据不足 > 支持归因。

    前置趋势不按二元检验一刀切：报告前置偏差量级 M（事件周前置系数
    最大绝对值）及其占估计效应的比例，占比过半才降级为证据不足；
    判定支持的结论也必须带着前置偏差说明一起呈现。
    """
    gaps: list[str] = []
    if not did.identified:
        gaps.append("处理后窗口内干净对照不足，整体效应无法识别")
        return UNIDENTIFIED, gaps
    pretrend = did.pretrend
    weekly = [abs(w["estimate"]) for w in pretrend.get("weekly_leads", [])]
    bound = max(weekly) if weekly else 0.0
    ratio = bound / abs(did.att) if did.att else float("inf")
    if did.att_ci is not None and did.att_ci[0] <= 0 <= did.att_ci[1]:
        gaps.append("效应区间跨过 0，统计上不能排除无效应")
        return INSUFFICIENT, gaps
    if ratio > 0.5:
        gaps.append(
            f"前置趋势偏差量级 {bound:.1f} 占估计效应 {ratio:.0%}，平行趋势存疑"
        )
        return INSUFFICIENT, gaps
    if did.control_scope == "include_spillover":
        gaps.append("对照池包含外溢客源，归因强度降级")
        return INSUFFICIENT, gaps
    if weekly:
        wald_p = pretrend.get("wald_p")
        wald_text = "<0.001" if isinstance(wald_p, float) and wald_p < 0.001 else (
            f"{wald_p:.3f}" if isinstance(wald_p, float) else "—"
        )
        gaps.append(
            f"前置偏差量级 {bound:.1f}（占效应 {ratio:.0%}），结论在此幅度内稳健；"
            f"Wald p={wald_text}"
        )
    return SUPPORTED, gaps


def analyze(
    materials: Materials,
    structure: Structure,
    config: DidConfig | None = None,
    *,
    created_at: str | None = None,
) -> IncrementalityReport:
    config = config or DidConfig()
    created_at = created_at or dt.datetime.now(dt.timezone.utc).isoformat()
    routes: dict[str, RouteIncrement] = {}
    sensitivity: dict[str, dict] = {}
    gaps: list[str] = []

    for route_id in structure.routes:
        panel = build_route_panel(materials, structure, route_id)
        did = estimate_route(panel, config)
        verdict, route_gaps = _verdict(did)

        # 唯一游客换算：逐客源地估计每人每日到访次数，加权平均
        stats = [
            route_trip_stats(materials, structure, o, route_id)
            for o in panel.origins
        ]
        covered = [s for s in stats if s.covered and s.visits_per_person]
        att_unique = None
        att_unique_note = ""
        if did.att is not None and covered:
            vpp = sum(s.visits_per_person for s in covered) / len(covered)
            if len(covered) < len(stats):
                att_unique_note = (
                    "部分客源地调查未覆盖，唯一游客换算基于已覆盖客源地，存在外推"
                )
            lo = did.att_ci[0] if did.att_ci else did.att
            hi = did.att_ci[1] if did.att_ci else did.att
            att_unique = (lo / vpp, hi / vpp)
            if not att_unique_note:
                att_unique_note = f"按每人每日 {vpp:.2f} 次到访换算"
        elif did.att is not None:
            n_sites = len(structure.routes[route_id].sites)
            route_gaps.append(
                f"调查未覆盖本线路行程，唯一游客只能取无信息区间（每人每日 1–{n_sites} 次）"
            )
            lo = did.att_ci[0] if did.att_ci else did.att
            hi = did.att_ci[1] if did.att_ci else did.att
            att_unique = (min(lo, hi) / n_sites, max(lo, hi) / 1.0)
            att_unique_note = "无调查覆盖，区间按每人每日 1 至场馆数次到访展开"
        if did.censored_cells_dropped:
            route_gaps.append(
                f"剔除售罄截尾单元 {did.censored_cells_dropped} 个；敏感性口径见 sensitivity"
            )
        routes[route_id] = RouteIncrement(
            route=route_id,
            verdict=verdict,
            att_visits=did.att,
            att_visits_ci=did.att_ci,
            att_unique=att_unique,
            att_unique_note=att_unique_note,
            did=did,
            gaps=route_gaps,
        )

        # 敏感性：保留售罄截尾单元
        alt = estimate_route(
            panel,
            DidConfig(
                min_controls=config.min_controls,
                bootstrap=config.bootstrap,
                seed=config.seed,
                control_scope=config.control_scope,
                include_censored=True,
            ),
        )
        sensitivity[route_id] = {
            "include_censored": True,
            "att": alt.att,
            "att_ci": list(alt.att_ci) if alt.att_ci else None,
            "note": "保留截尾单元会低估真实需求，方向为向下偏误",
        }

    if any(r.verdict != SUPPORTED for r in routes.values()):
        gaps.append("存在证据不足或无法识别的线路，相关增量不得计入已证实成果")
    return IncrementalityReport(
        created_at=created_at,
        data_fingerprint=materials.fingerprint,
        config={
            "min_controls": config.min_controls,
            "bootstrap": config.bootstrap,
            "control_scope": config.control_scope,
            "include_censored": config.include_censored,
            "seed": config.seed,
        },
        routes=routes,
        sensitivity=sensitivity,
        gaps=gaps,
    )
