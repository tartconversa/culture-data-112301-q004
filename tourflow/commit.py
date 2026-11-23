"""整团承诺引擎：报价占用、失效、幂等接受、全有或全无。

纪律：
- 整团承诺是“全有或全无”：线路全部场馆与接驳同时满足才成立，
  单个场馆同意不代表整团落实；请求线路不可行时按 group_requests
  列出的整条替代线路尝试，人数不自动缩减。
- 同一团体重复接受只产生一次承诺（幂等）；待确认报价只占用一次容量，
  过期即释放，不能无限重复占量。
- 已接受的团体不被牺牲：新承诺只使用剩余容量，引擎不驱逐既有承诺。
- 确认结果与资源余量必须一致：每次接受都重新过台账校验。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .capacity import CapacityLedger
from .demand import Forecast
from .model import Structure, parse_date, parse_moment
from .policy import Policy, committable


@dataclass
class Decision:
    group: str
    status: str  # accepted | pilot | rejected | duplicate | expired
    route: str | None
    people: int
    committable: int
    reasons: list[str] = field(default_factory=list)


@dataclass
class PlanResult:
    decisions: list[Decision]
    accepted_people: int
    rejected_people: int
    idle_risk_people: float
    protected_pressure: list[dict]
    notes: list[str] = field(default_factory=list)


def commitment_id(group: str, route: str, on: str) -> str:
    return f"CM-{group}-{route}-{on}"


class CommitEngine:
    """在一份台账与容量账上执行承诺决策；所有变更先校验后落账。"""

    def __init__(
        self,
        structure: Structure,
        ledger: dict,
        capacity: CapacityLedger,
        forecasts: dict[tuple[str, str, dt.date], Forecast],
        evidence: dict[str, str],
        policy: Policy,
        basis: dict | None = None,
        show_up: dict[str, float] | None = None,
    ):
        self.structure = structure
        self.ledger = ledger
        self.capacity = capacity
        self.forecasts = forecasts
        self.evidence = evidence
        self.policy = policy
        self.basis = basis or {}
        self.show_up = show_up or {}

    # ---- 内部工具 -----------------------------------------------------
    def _request(self, requests: dict[str, dict], group: str) -> dict:
        if group not in requests:
            raise KeyError("未知团体请求 " + group)
        return requests[group]

    def _existing_for_group(self, group: str) -> dict | None:
        for commitment in self.ledger["commitments"].values():
            if commitment["group"] == group:
                return commitment
        return None

    def _committable_for(self, request: dict, route: str) -> tuple[int, list[str], bool]:
        day = parse_date(request["on"])
        forecast = self.forecasts.get((request["origin"], route, day))
        verdict = self.evidence.get(route, "无法识别")
        result = committable(forecast, verdict, self.policy)
        return result.quantity, result.notes, result.pilot

    def _protected_ok(self, day: dt.date, route: str, people: int) -> list[str]:
        problems = []
        for site in self.structure.routes[route].sites:
            limit = self.structure.sites[site].protected_limit
            remaining = self.capacity.remaining_protected.get((day, site), limit)
            used_after = limit - (remaining - people)
            if limit and used_after / limit > self.policy.protected_util_cap:
                problems.append(
                    f"{site} 保护区利用率将达 {used_after / limit:.0%}，超过上限 {self.policy.protected_util_cap:.0%}"
                )
        return problems

    def _candidate_routes(self, request: dict) -> list[str]:
        return [request["route"], *[r for r in request["allowed_substitutes"] if r != request["route"]]]

    # ---- 报价与接受 ---------------------------------------------------
    def evaluate(self, request: dict, now: dt.datetime) -> Decision:
        """只评估不落账；返回决策与理由。"""
        group = request["id"]
        if self._existing_for_group(group) is not None:
            existing = self._existing_for_group(group)
            return Decision(group, "duplicate", existing["route"], existing["people"], 0,
                            ["该团体已有承诺，重复接受不产生新承诺"])
        expires = parse_moment(request["expires_at"])
        if now > expires:
            return Decision(group, "expired", None, request["people"], 0,
                            [f"报价已于 {request['expires_at']} 失效"])
        day = parse_date(request["on"])
        reasons: list[str] = []
        for route in self._candidate_routes(request):
            feasible = self.capacity.route_feasible(self.structure, day, route, request["people"])
            cap, notes, pilot = self._committable_for(request, route)
            problems = list(feasible.binding)
            problems.extend(self._protected_ok(day, route, request["people"]))
            if request["people"] > cap:
                problems.append(f"可承诺量 {cap} < 请求人数 {request['people']}")
            if not problems:
                status = "pilot" if pilot else "accepted"
                return Decision(group, status, route, request["people"], cap, notes)
            reasons.append(f"{route}: " + "；".join(problems))
        cap, _, _ = self._committable_for(request, request["route"])
        return Decision(group, "rejected", None, request["people"], cap, reasons)

    def accept(self, requests: dict[str, dict], group: str, now: dt.datetime,
               *, at: str | None = None) -> dict:
        """接受整团承诺；幂等：同一团体重复接受返回既有承诺。"""
        existing = self._existing_for_group(group)
        if existing is not None:
            return existing
        request = self._request(requests, group)
        decision = self.evaluate(request, now)
        if decision.status not in ("accepted", "pilot"):
            raise ValueError("不可接受: " + "；".join(decision.reasons))
        day = parse_date(request["on"])
        route = decision.route
        cid = commitment_id(group, route, request["on"])
        if cid in self.ledger["commitments"]:
            return self.ledger["commitments"][cid]
        # 落账：先扣容量，再写承诺，保持确认结果与资源余量一致
        self.capacity.apply_route(self.structure, day, route, request["people"], sign=-1)
        commitment = {
            "id": cid,
            "group": group,
            "route": route,
            "on": request["on"],
            "people": request["people"],
            "origin": request["origin"],
            "state": "confirmed",
            "source": "engine",
            "basis": {
                **self.basis,
                "policy": {
                    "conservatism": self.policy.conservatism,
                    "pilot_fraction": self.policy.pilot_fraction,
                    "protected_util_cap": self.policy.protected_util_cap,
                },
                "capacity_fingerprint": self.capacity.fingerprint(),
                "decision": decision.status,
            },
            "history": [
                {"ts": at or now.isoformat(), "action": "accepted", "route": route}
            ],
        }
        self.ledger["commitments"][cid] = commitment
        self.ledger["holds"].pop(group, None)
        return commitment

    def plan(self, requests: dict[str, dict], now: dt.datetime, *,
             commit: bool = False, at: str | None = None) -> PlanResult:
        """按失效时间顺序评估全部请求。

        无论是否落账，已接受的决策都在同一本容量账上顺序扣减，
        保证试算结果与资源余量一致；commit 为假时在深拷贝上试算，
        不改变引擎状态。
        """
        import copy

        engine = self if commit else copy.deepcopy(self)
        ordered = sorted(
            requests.values(),
            key=lambda r: (r["expires_at"], r["id"]),
        )
        decisions: list[Decision] = []
        accepted_people = 0
        rejected_people = 0
        for request in ordered:
            decision = engine.evaluate(request, now)
            decisions.append(decision)
            if decision.status in ("accepted", "pilot"):
                accepted_people += request["people"]
                engine.accept(requests, request["id"], now, at=at)
            elif decision.status == "rejected":
                rejected_people += request["people"]
        idle = engine._idle_risk()
        pressure = engine._protected_pressure()
        notes = []
        if any(d.status == "pilot" for d in decisions):
            notes.append("含小规模试行承诺：证据不足的推广变化只按试行比例计入")
        return PlanResult(
            decisions=decisions,
            accepted_people=accepted_people,
            rejected_people=rejected_people,
            idle_risk_people=idle,
            protected_pressure=pressure,
            notes=notes,
        )

    # ---- 汇总指标 -----------------------------------------------------
    def _idle_risk(self) -> float:
        """已承诺人数中预计不到访的部分（按客源地的核销反馈率）。"""
        idle = 0.0
        for commitment in self.ledger["commitments"].values():
            if commitment["state"] not in ("confirmed", "substituted"):
                continue
            group_origin = commitment.get("origin")
            rate = self.show_up.get(group_origin, 1.0) if group_origin else 1.0
            idle += commitment["people"] * (1.0 - rate)
        return idle

    def _protected_pressure(self) -> list[dict]:
        out = []
        for (day, site), remaining in sorted(self.capacity.remaining_protected.items()):
            limit = self.structure.sites[site].protected_limit
            if limit:
                out.append(
                    {
                        "on": day.isoformat(),
                        "site": site,
                        "protected_utilization": round(1.0 - remaining / limit, 4),
                        "over_cap": (1.0 - remaining / limit) > self.policy.protected_util_cap,
                    }
                )
        return out


def seed_obligations(
    structure: Structure,
    ledger: dict,
    obligations: list[dict],
    requests: dict[str, dict],
) -> list[str]:
    """把资料中已成立的承诺（obligations）迁入台账；幂等，返回新迁入的 id。

    只迁移记录，不扣容量；容量统一由 replay_commitments 按台账重放，
    保证跨命令的一致性。
    """
    added = []
    for row in obligations:
        request = requests[row["group"]]
        cid = commitment_id(row["group"], row["route"], request["on"])
        if cid in ledger["commitments"]:
            continue
        ledger["commitments"][cid] = {
            "id": cid,
            "group": row["group"],
            "route": row["route"],
            "on": request["on"],
            "people": request["people"],
            "origin": request["origin"],
            "state": row["state"],
            "source": "seed",
            "basis": {"note": "资料中已成立的承诺，无分析依据记录", "obligation": row["id"]},
            "history": [
                {"ts": row["accepted_on"], "action": "accepted", "route": row["route"]}
            ],
        }
        added.append(cid)
    return added


def replay_commitments(structure: Structure, ledger: dict, capacity: CapacityLedger) -> None:
    """把台账中仍然有效的承诺重放到容量账上（确认结果与资源余量一致）。"""
    for commitment in ledger["commitments"].values():
        if commitment["state"] not in ("confirmed", "substituted"):
            continue
        capacity.apply_route(
            structure,
            parse_date(commitment["on"]),
            commitment["route"],
            commitment["people"],
            sign=-1,
        )
