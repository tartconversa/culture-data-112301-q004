"""资源台账：场馆日容量、保护区上限与共享接驳席位。

口径（对应资料契约）：
- 日容量约束场馆总人数，保护区上限约束进入保护区人数；线路全体成员
  均进入线路所列场馆的保护区，散客与已确认团体保守起见同样计入保护区
  压力（可在 Policy 中调整）。
- 相同 bus 和日期表示共享同一趟接驳；bus_seats 是车次容量，不能按线路相加。
- 已售散客票、其他已确认团体与已占接驳席位先从供给中扣减，剩余部分
  才可用于新的团体承诺。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field

from .materials import Materials
from .model import Structure, parse_date


@dataclass(frozen=True)
class Feasibility:
    ok: bool
    binding: tuple[str, ...] = ()  # 不满足的资源描述


@dataclass
class CapacityLedger:
    """按日期记录各类资源剩余量；承诺与报价占用都从这里扣。"""

    remaining_total: dict[tuple[dt.date, str], int] = field(default_factory=dict)
    remaining_protected: dict[tuple[dt.date, str], int] = field(default_factory=dict)
    remaining_bus: dict[tuple[dt.date, str], int] = field(default_factory=dict)
    closed: set[tuple[dt.date, str]] = field(default_factory=set)

    def route_feasible(
        self, structure: Structure, day: dt.date, route: str, people: int
    ) -> Feasibility:
        binding = []
        for site in structure.routes[route].sites:
            if (day, site) in self.closed:
                binding.append(f"{site} 当日关闭")
                continue
            if self.remaining_total.get((day, site), 0) < people:
                binding.append(
                    f"{site} 日容量剩余 {self.remaining_total.get((day, site), 0)} < {people}"
                )
            if self.remaining_protected.get((day, site), 0) < people:
                binding.append(
                    f"{site} 保护区剩余 {self.remaining_protected.get((day, site), 0)} < {people}"
                )
        bus = structure.routes[route].bus
        if self.remaining_bus.get((day, bus), 0) < people:
            binding.append(
                f"接驳 {bus} 剩余座位 {self.remaining_bus.get((day, bus), 0)} < {people}"
            )
        return Feasibility(ok=not binding, binding=tuple(binding))

    def apply_route(
        self, structure: Structure, day: dt.date, route: str, people: int, sign: int = -1
    ) -> None:
        """占用（sign=-1）或释放（sign=+1）线路全部资源。"""
        for site in structure.routes[route].sites:
            self.remaining_total[(day, site)] = (
                self.remaining_total.get((day, site), 0) + sign * people
            )
            self.remaining_protected[(day, site)] = (
                self.remaining_protected.get((day, site), 0) + sign * people
            )
        bus = structure.routes[route].bus
        self.remaining_bus[(day, bus)] = self.remaining_bus.get((day, bus), 0) + sign * people

    def utilization(self, structure: Structure, day: dt.date, site: str) -> dict:
        limit = structure.sites[site].protected_limit
        remaining = self.remaining_protected.get((day, site), limit)
        return {
            "site": site,
            "on": day.isoformat(),
            "protected_remaining": remaining,
            "protected_utilization": 1.0 - remaining / limit if limit else None,
        }

    def fingerprint(self) -> str:
        payload = {
            "total": {f"{d.isoformat()}|{s}": v for (d, s), v in sorted(self.remaining_total.items())},
            "protected": {
                f"{d.isoformat()}|{s}": v for (d, s), v in sorted(self.remaining_protected.items())
            },
            "bus": {f"{d.isoformat()}|{b}": v for (d, b), v in sorted(self.remaining_bus.items())},
            "closed": sorted(f"{d.isoformat()}|{s}" for d, s in self.closed),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def build_ledger(materials: Materials, structure: Structure) -> CapacityLedger:
    """从 supply 表构造台账；同一车次多行不一致时取最保守（最小剩余）。"""
    ledger = CapacityLedger()
    bus_reserved: dict[tuple[dt.date, str], int] = {}
    for row in materials.table("supply"):
        day = parse_date(row["on"])
        site = structure.sites[row["site"]]
        ledger.remaining_total[(day, site.id)] = (
            site.daily_capacity - row["sold_individual"] - row["reserved_groups"]
        )
        # 保守口径：已售散客与已确认团体全部计入保护区压力
        ledger.remaining_protected[(day, site.id)] = (
            site.protected_limit - row["sold_individual"] - row["reserved_groups"]
        )
        if row["closed"]:
            ledger.closed.add((day, site.id))
        bus = structure.bus_by_site.get(row["site"])
        if bus is not None:
            key = (day, bus)
            bus_reserved[key] = max(bus_reserved.get(key, 0), row["bus_reserved"])
    for (day, bus), reserved in bus_reserved.items():
        ledger.remaining_bus[(day, bus)] = structure.bus_seats[bus] - reserved
    return ledger
