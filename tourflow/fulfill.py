"""后续事实处理：临时关闭的替代、核销回执分类与反馈。

纪律：
- 场馆临时关闭时，为受影响承诺在 group_requests 列出的整条替代线路中
  寻找仍满足团队已接受权益的方案；人数不自动缩减；替代会改变整条行程
  与其他场馆的容量，台账同步改写。
- 无法履约的部分明确进入退款或人工协商，承诺记录保留，不能简单删除。
- 核销与交通回执用于区分未到访、线路替代与服务取消，并回写到需求
  预测与推广观察期所用的反馈中。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .capacity import CapacityLedger
from .model import Structure, parse_date

# 核销分类
FULFILLED = "已核销"
NO_SHOW = "未到访"
ROUTE_SUBSTITUTED = "线路替代"
CANCELLED = "服务取消"


@dataclass
class FulfillResult:
    substitutions: list[dict] = field(default_factory=list)
    refunds: list[dict] = field(default_factory=list)
    manual_reviews: list[dict] = field(default_factory=list)
    outcomes: dict[str, dict] = field(default_factory=dict)
    feedback: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _commitments_on(ledger: dict, day: dt.date) -> list[dict]:
    return [
        c
        for c in ledger["commitments"].values()
        if c["on"] == day.isoformat() and c["state"] in ("confirmed", "substituted")
    ]


def apply_closure(
    structure: Structure,
    ledger: dict,
    capacity: CapacityLedger,
    requests: dict[str, dict],
    day: dt.date,
    site: str,
    *,
    at: str,
) -> FulfillResult:
    """处理一日一馆的临时关闭：先补登记关闭，再为受影响承诺找替代。"""
    result = FulfillResult()
    capacity.closed.add((day, site))
    for commitment in _commitments_on(ledger, day):
        route = structure.routes[commitment["route"]]
        if site not in route.sites:
            continue
        request = requests[commitment["group"]]
        people = commitment["people"]
        replaced = False
        for substitute in request["allowed_substitutes"]:
            if substitute == commitment["route"]:
                continue
            sub_route = structure.routes[substitute]
            if any((day, s) in capacity.closed for s in sub_route.sites):
                continue
            # 先释放原线路占用，再校验替代线路整团可行性
            capacity.apply_route(structure, day, commitment["route"], people, sign=+1)
            feasible = capacity.route_feasible(structure, day, substitute, people)
            if feasible.ok:
                capacity.apply_route(structure, day, substitute, people, sign=-1)
                commitment["history"].append(
                    {
                        "ts": at,
                        "action": "substituted",
                        "from": commitment["route"],
                        "to": substitute,
                        "reason": f"{site} 于 {day.isoformat()} 临时关闭",
                    }
                )
                commitment["route"] = substitute
                commitment["state"] = "substituted"
                result.substitutions.append(
                    {
                        "group": commitment["group"],
                        "from": route.id,
                        "to": substitute,
                        "people": people,
                    }
                )
                replaced = True
                break
            # 替代不可行，恢复原占用后再试下一条
            capacity.apply_route(structure, day, commitment["route"], people, sign=-1)
        if replaced:
            continue
        if request["allowed_substitutes"]:
            commitment["state"] = "manual_review"
            commitment["history"].append(
                {
                    "ts": at,
                    "action": "manual_review",
                    "reason": f"{site} 关闭且替代线路均不可行，转人工协商",
                }
            )
            result.manual_reviews.append(
                {"group": commitment["group"], "people": people, "reason": "替代线路均不可行"}
            )
        else:
            commitment["state"] = "refund"
            commitment["history"].append(
                {
                    "ts": at,
                    "action": "refund",
                    "reason": f"{site} 关闭且无约定替代线路，进入退款",
                }
            )
            result.refunds.append({"group": commitment["group"], "people": people})
        # 无法履约：释放原线路占用，但承诺记录保留
        capacity.apply_route(structure, day, commitment["route"], people, sign=+1)
    return result


def apply_redemption(
    ledger: dict,
    requests: dict[str, dict],
    structure: Structure,
    event: dict,
    *,
    at: str,
) -> dict:
    """登记一次入馆核销或接驳回执，返回该团体的最新履约分类。"""
    group = event["group"]
    outcome = ledger["outcomes"].setdefault(
        group,
        {"redeemed": {}, "transport": {}, "classification": None, "history": []},
    )
    bucket = "redeemed" if event["kind"] == "redeem" else "transport"
    outcome[bucket][event["site"]] = event["actual_people"]
    outcome["history"].append({"ts": at, "event": event["id"], "kind": event["kind"]})
    _classify(ledger, requests, structure, group)
    return outcome


def _classify(ledger: dict, requests: dict[str, dict], structure: Structure, group: str) -> None:
    """按承诺与回执区分：未到访 / 线路替代 / 服务取消 / 已核销。"""
    commitment = next(
        (c for c in ledger["commitments"].values() if c["group"] == group), None
    )
    outcome = ledger["outcomes"].get(group)
    if commitment is None or outcome is None:
        return
    if commitment["state"] in ("refund", "manual_review"):
        outcome["classification"] = CANCELLED
        return
    committed_sites = set(structure.routes[commitment["route"]].sites)
    redeemed_sites = set(outcome["redeemed"])
    if not redeemed_sites:
        outcome["classification"] = NO_SHOW
    elif redeemed_sites <= committed_sites:
        outcome["classification"] = FULFILLED
    else:
        outcome["classification"] = ROUTE_SUBSTITUTED
    commitment_people = commitment["people"]
    actual = max(outcome["redeemed"].values(), default=0)
    outcome["committed_people"] = commitment_people
    outcome["actual_people"] = actual
    outcome["show_up_rate"] = actual / commitment_people if commitment_people else None


def apply_feedback_closures(capacity: CapacityLedger, ledger: dict) -> None:
    """把台账反馈中的关闭单元落到容量账上（后续承诺不得再占用）。"""
    for cell in ledger.get("feedback", {}).get("excluded_cells", []):
        capacity.closed.add((parse_date(cell["on"]), cell["site"]))


def apply_events(
    structure: Structure,
    ledger: dict,
    capacity: CapacityLedger,
    requests: dict[str, dict],
    events: list[dict],
    *,
    at: str,
) -> FulfillResult:
    """按顺序应用后续事实；已处理的事件不重复应用（幂等）。"""
    result = FulfillResult()
    for event in events:
        if event["id"] in ledger["events_applied"]:
            continue
        day = parse_date(event["on"])
        if event["kind"] == "closure":
            sub = apply_closure(
                structure, ledger, capacity, requests, day, event["site"], at=at
            )
            result.substitutions.extend(sub.substitutions)
            result.refunds.extend(sub.refunds)
            result.manual_reviews.extend(sub.manual_reviews)
        elif event["kind"] in ("redeem", "transport"):
            apply_redemption(ledger, requests, structure, event, at=at)
        else:
            raise ValueError("未知事件类型 " + event["kind"])
        ledger["events_applied"].append(event["id"])
    result.outcomes = ledger["outcomes"]
    # 反馈：核销率回写需求预测；关闭单元进入推广观察期的排除清单
    show_up: dict[str, list[float]] = {}
    for group, outcome in ledger["outcomes"].items():
        rate = outcome.get("show_up_rate")
        if rate is None:
            continue
        origin = requests[group]["origin"]
        show_up.setdefault(origin, []).append(rate)
    ledger["feedback"]["show_up"] = {
        origin: sum(rates) / len(rates) for origin, rates in show_up.items()
    }
    excluded = ledger["feedback"]["excluded_cells"]
    for event in events:
        if event["kind"] == "closure":
            cell = {"on": event["on"], "site": event["site"]}
            if cell not in excluded:
                excluded.append(cell)
    result.feedback = ledger["feedback"]
    return result
