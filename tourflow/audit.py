"""溯源审计：承诺依赖什么分析、实际完成什么、下一轮可承诺量为何变化。

- 每次增量测算/需求预测/承诺试算都登记为一次运行：配置、资料指纹、
  结果摘要哈希，运行 id 由内容派生，可复验。
- 每项承诺记录其依据（分析运行 id、策略、容量指纹）。
- 两轮可承诺量的差异按“容量变化 → 需求预测变化 → 反馈变化”的顺序
  分解，让负责人能追查新到访资料为何改变下一轮可承诺量。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass


def _digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


@dataclass
class RunRegistry:
    runs: dict  # state/runs.json 的内容

    def record(self, kind: str, config: dict, summary: dict, data_fingerprint: str,
               *, at: str | None = None) -> str:
        at = at or dt.datetime.now(dt.timezone.utc).isoformat()
        seq = len(self.runs["runs"])
        run_id = f"{kind}-{seq:03d}-{_digest({'kind': kind, 'config': config, 'data': data_fingerprint, 'seq': seq})}"
        self.runs["runs"].append(
            {
                "id": run_id,
                "kind": kind,
                "ts": at,
                "config": config,
                "data_fingerprint": data_fingerprint,
                "summary_digest": _digest(summary),
                "summary": summary,
            }
        )
        return run_id

    def get(self, run_id: str) -> dict | None:
        for run in self.runs["runs"]:
            if run["id"] == run_id:
                return run
        return None


def commitment_lineage(commitment: dict, registry: RunRegistry,
                       outcomes: dict[str, dict]) -> dict:
    """一项承诺的完整链条：依据 → 决策 → 履约。"""
    basis = commitment.get("basis", {})
    runs = {}
    for key in ("forecast_run", "incrementality_run"):
        run_id = basis.get(key)
        if run_id:
            run = registry.get(run_id)
            runs[key] = run if run else {"id": run_id, "missing": True}
    outcome = outcomes.get(commitment["group"], {})
    return {
        "commitment": commitment["id"],
        "group": commitment["group"],
        "route": commitment["route"],
        "on": commitment["on"],
        "people": commitment["people"],
        "state": commitment["state"],
        "source": commitment["source"],
        "basis": basis,
        "runs": runs,
        "outcome": {
            "classification": outcome.get("classification"),
            "committed_people": outcome.get("committed_people"),
            "actual_people": outcome.get("actual_people"),
            "show_up_rate": outcome.get("show_up_rate"),
        },
        "history": commitment.get("history", []),
    }


def explain_committable_change(
    before: dict[tuple, int],
    after: dict[tuple, int],
    *,
    capacity_before: dict[tuple, int],
    capacity_after: dict[tuple, int],
    demand_before: dict[tuple, int],
    demand_after: dict[tuple, int],
) -> list[dict]:
    """分解可承诺量变化：先容量、后需求（顺序固定并写入说明）。

    before/after: (origin, route, on) -> 可承诺量
    capacity_*: 同键的容量约束值；demand_*: 同键的需求上限值。
    """
    keys = sorted(set(before) | set(after))
    explanations = []
    for key in keys:
        b = before.get(key, 0)
        a = after.get(key, 0)
        if a == b:
            continue
        cap_delta = min(capacity_after.get(key, 0), demand_before.get(key, 0)) - b
        dem_delta = a - min(capacity_after.get(key, 0), demand_before.get(key, 0))
        explanations.append(
            {
                "key": "|".join(str(k) for k in key),
                "before": b,
                "after": a,
                "capacity_effect": cap_delta,
                "demand_effect": dem_delta,
                "note": "先替换容量约束、再替换需求上限的有序分解",
            }
        )
    return explanations
