"""承诺策略：保守程度如何映射为可承诺量。

- conservatism ∈ [0,1] 选择需求分布的分位作为可承诺上限：
  越保守，取越低的分位，拒绝越多、闲置越少。
- 推广带来的需求变化只有在增量测算支持归因时全额计入；
  证据不足或无法识别时，只按 pilot_fraction 计入小规模试行，
  不能凭一份高热度指数直接占满旺季供给。
- protected_util_cap 给保护区人流上限再留一道缓冲。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .demand import Forecast

# 归因判定（与 incrementality 模块一致，避免循环导入）
SUPPORTED = "支持归因"


@dataclass(frozen=True)
class Policy:
    conservatism: float = 0.5
    pilot_fraction: float = 0.25
    protected_util_cap: float = 0.95

    def __post_init__(self):
        if not 0.0 <= self.conservatism <= 1.0:
            raise ValueError("conservatism 需在 [0,1]")
        if not 0.0 <= self.pilot_fraction <= 1.0:
            raise ValueError("pilot_fraction 需在 [0,1]")
        if not 0.0 < self.protected_util_cap <= 1.0:
            raise ValueError("protected_util_cap 需在 (0,1]")

    @property
    def cap_quantile(self) -> str:
        c = self.conservatism
        if c < 0.25:
            return "q50"
        if c < 0.5:
            return "q25"
        if c < 0.75:
            return "q10"
        return "q05"


@dataclass
class Committable:
    quantity: int
    cap_quantile: str
    evidence: str
    pilot: bool
    notes: list[str] = field(default_factory=list)


def committable(forecast: Forecast | None, evidence: str, policy: Policy) -> Committable:
    """把预测与归因证据折算为某 客源地×线路×日 的可承诺人数上限。"""
    if forecast is None:
        return Committable(0, policy.cap_quantile, evidence, False, ["无需求预测，不可承诺"])
    cap = forecast.quantiles[policy.cap_quantile]
    notes = list(forecast.notes)
    if evidence == SUPPORTED:
        quantity = math.floor(cap)
        return Committable(quantity, policy.cap_quantile, evidence, False, notes)
    # 证据不足：自然需求照常，推广变化只按试行比例计入
    allowed = min(cap, forecast.natural_lam + policy.pilot_fraction * max(forecast.uplift_lam, 0.0))
    notes.append(
        f"推广变化证据不足（{evidence}），仅按试行比例 {policy.pilot_fraction:.0%} 计入"
    )
    return Committable(math.floor(allowed), policy.cap_quantile, evidence, True, notes)
