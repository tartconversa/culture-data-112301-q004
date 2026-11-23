"""分期推广的双重差分估计。

采用 Callaway–Sant'Anna 风格的 2x2 干净比较：对每个已推广客源地 o、
每个日期 t，用“尚未被推广且未被外溢波及”的客源地作为对照，估计
    ATT(o, t) = (y[o,t] - base[o]) - mean_c (y[c,t] - base[c])
其中 base 为推广前 base_window 天内可用观测的均值（基期窗口平均可以
避免售罄/关闭日恰好落在 g-1 时整行丢弃，与合成双重差分的预处理一致）。
事件时间系数按相对日聚合，整体 ATT 先按客源地平均再跨客源地平均，
避免推广日早的客源地在长后窗中占过高权重。

识别纪律：
- 对照池默认剔除资料确认的外溢客源（strict）；选择 include_spillover 时
  必须在报告中标注识别风险。
- 售罄单元为右截尾，主口径剔除，敏感性口径保留并标注向下偏误方向。
- 关闭单元恒剔除（机械为零，不是需求）。
- 对照不足的 (o, t) 单元不估计，计入无法识别清单，不外推。
- 推断使用客源地层面的区块自助法；客源地数量很少，区间仅作量级参考。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from .panel import RoutePanel


@dataclass(frozen=True)
class DidConfig:
    min_controls: int = 2
    bootstrap: int = 500
    seed: int = 20261123
    control_scope: str = "strict"  # strict | include_spillover
    include_censored: bool = False  # 敏感性口径：保留售罄截尾单元
    base_window: int = 14  # 基期窗口长度（天）
    min_base_obs: int = 5  # 基期窗口内最少可用天数

    def __post_init__(self):
        if self.control_scope not in ("strict", "include_spillover"):
            raise ValueError("未知对照范围 " + self.control_scope)
        if self.base_window < 1 or self.min_base_obs < 1:
            raise ValueError("基期窗口参数必须为正")
        if self.min_base_obs > self.base_window:
            raise ValueError("min_base_obs 不能超过 base_window")


@dataclass
class CellEstimate:
    origin: str
    on: dt.date
    event_time: int
    att: float
    n_controls: int


@dataclass
class EventPoint:
    event_time: int
    estimate: float
    n_origins: int
    ci_low: float | None = None
    ci_high: float | None = None


@dataclass
class RouteDid:
    route: str
    att: float | None
    att_ci: tuple[float, float] | None
    event_study: list[EventPoint]
    pretrend: dict
    cells_used: int
    cells_unidentified: list[dict]
    n_treated_origins: int
    control_scope: str
    include_censored: bool
    censored_cells_dropped: int
    closed_cells_dropped: int
    balance: dict
    spillover: dict
    notes: list[str] = field(default_factory=list)

    @property
    def identified(self) -> bool:
        return self.att is not None


def _arrays(panel: RoutePanel, config: DidConfig):
    origins = list(panel.origins)
    dates = list(panel.dates)
    o_idx = {o: i for i, o in enumerate(origins)}
    d_idx = {d: i for i, d in enumerate(dates)}
    n_o, n_d = len(origins), len(dates)
    y = np.full((n_o, n_d), np.nan)
    usable = np.zeros((n_o, n_d), dtype=bool)
    closed = np.zeros((n_o, n_d), dtype=bool)
    censored = np.zeros((n_o, n_d), dtype=bool)
    exposed = np.zeros((n_o, n_d), dtype=bool)
    start_idx = np.full(n_o, -1)
    ever = np.zeros(n_o, dtype=bool)
    for row in panel.frame.itertuples():
        i, j = o_idx[row.origin], d_idx[row.on]
        y[i, j] = row.visits
        exposed[i, j] = row.exposed
        closed[i, j] = row.closed_any
        censored[i, j] = bool(row.censored)
        if row.closed_any:
            continue
        if row.censored and not config.include_censored:
            continue
        usable[i, j] = True
    starts = panel.frame.groupby("origin")["start"].first().to_dict()
    evers = panel.frame.groupby("origin")["ever_treated_route"].first().to_dict()
    own_start_idx = np.full(n_o, -1)
    for o in origins:
        ever[o_idx[o]] = bool(evers[o])
        start = starts[o]
        if start is not None and start in d_idx:
            own_start_idx[o_idx[o]] = d_idx[start]
            if ever[o_idx[o]]:
                start_idx[o_idx[o]] = d_idx[start]
    return {
        "origins": origins,
        "dates": dates,
        "y": y,
        "usable": usable,
        "closed": closed,
        "censored": censored,
        "exposed": exposed,
        "start_idx": start_idx,
        "own_start_idx": own_start_idx,
        "ever": ever,
    }


def _base_values(arr, g: int, config: DidConfig):
    """各客源地在 [g-base_window, g-1] 窗口内的基期均值与可用标记。"""
    y, usable = arr["y"], arr["usable"]
    lo = max(0, g - config.base_window)
    base = np.full(len(arr["origins"]), np.nan)
    for i in range(len(arr["origins"])):
        window = [y[i, t] for t in range(lo, g) if usable[i, t]]
        if len(window) >= config.min_base_obs:
            base[i] = float(np.mean(window))
    return base


def _control_pool(arr, t: int, g: int, base, config: DidConfig) -> list[int]:
    """t 期的干净对照：整个比较窗口内未启动任何推广、未被外溢波及。

    任何已启动的推广都会抬高该客源地的整体到访（宣传带来目的地热度，
    不限于被推广的线路），因此对照必须在 [g-1, t] 全段未启动自有推广；
    基期窗口内的外溢同样污染基期，外溢为吸收状态，检查到 limit 即可。
    """
    limit = max(t, g - 1)
    pool = []
    for c in range(len(arr["origins"])):
        own = arr["own_start_idx"][c]
        if own >= 0 and own <= limit:
            continue  # 已启动自有推广，不能作为对照
        if config.control_scope == "strict" and arr["exposed"][c, limit]:
            continue  # 资料确认的外溢客源不属于干净对照
        if not arr["usable"][c, t] or np.isnan(base[c]):
            continue
        pool.append(c)
    return pool


def _cells(arr, config: DidConfig, treated_sample=None, control_sampler=None):
    """计算全部可识别 (o, t) 单元的 ATT；treated_sample/control_sampler 供自助法使用。"""
    y, usable = arr["y"], arr["usable"]
    dates = arr["dates"]
    treated = [i for i in range(len(arr["origins"])) if arr["ever"][i]]
    if treated_sample is not None:
        treated = treated_sample
    estimates: list[CellEstimate] = []
    unidentified: list[dict] = []
    for i in treated:
        g = arr["start_idx"][i]
        base = _base_values(arr, g, config)
        if np.isnan(base[i]):
            unidentified.append({"origin": arr["origins"][i], "reason": "基期窗口可用观测不足"})
            continue
        for t in range(len(dates)):
            if t >= g - config.base_window and t < g:
                continue  # 基期窗口不参与事件时间估计
            if not usable[i, t]:
                continue
            pool = _control_pool(arr, t, g, base, config)
            pool = [c for c in pool if c != i]
            if len(pool) < config.min_controls:
                unidentified.append(
                    {
                        "origin": arr["origins"][i],
                        "on": dates[t].isoformat(),
                        "reason": "干净对照不足",
                        "n_controls": len(pool),
                    }
                )
                continue
            if control_sampler is not None:
                chosen = [pool[k] for k in control_sampler(len(pool))]
            else:
                chosen = pool
            control_change = float(np.mean(y[chosen, t] - base[chosen]))
            estimates.append(
                CellEstimate(
                    origin=arr["origins"][i],
                    on=dates[t],
                    event_time=(dates[t] - dates[g]).days,
                    att=float(y[i, t] - base[i] - control_change),
                    n_controls=len(pool),
                )
            )
    return estimates, unidentified


def _aggregate(estimates: list[CellEstimate]):
    """事件时间系数与整体 ATT（先客源地内平均，再跨客源地平均）。"""
    by_event: dict[int, list[float]] = {}
    for cell in estimates:
        by_event.setdefault(cell.event_time, []).append(cell.att)
    event_points = [
        EventPoint(event_time=e, estimate=float(np.mean(v)), n_origins=len(v))
        for e, v in sorted(by_event.items())
    ]
    by_origin: dict[str, list[float]] = {}
    for cell in estimates:
        if cell.event_time >= 0:
            by_origin.setdefault(cell.origin, []).append(cell.att)
    att = None
    if by_origin:
        att = float(np.mean([np.mean(v) for v in by_origin.values()]))
    return event_points, att


def _week_bin(event_time: int) -> int:
    """事件日 → 事件周（-1..-7 → -1，-8..-14 → -2，……）。"""
    return -((-event_time + 6) // 7)


def _pretrend(event_points: list[EventPoint], boot_leads: list[dict[int, float]]):
    leads = {p.event_time: p.estimate for p in event_points if p.event_time < 0}
    if not leads:
        return {"leads": [], "joint_pass": None, "note": "无前期系数可检验"}
    # 日度前置系数噪声大，按事件周分箱后再做联合检验（事件研究的常规做法）
    def binned(series: dict[int, float]) -> dict[int, float]:
        groups: dict[int, list[float]] = {}
        for e, v in series.items():
            if e < 0:
                groups.setdefault(_week_bin(e), []).append(v)
        return {b: float(np.mean(vs)) for b, vs in groups.items()}

    observed = binned(leads)
    rep_bins = [binned(rep) for rep in boot_leads]
    weeks = sorted(observed)
    # 对角 Wald 联合检验：噪声大的周自动降权；同时置信带作为补充描述
    variances = {}
    for w in weeks:
        reps = [rep[w] for rep in rep_bins if w in rep]
        variances[w] = float(np.var(reps, ddof=1)) if len(reps) > 1 else float("nan")
    usable_weeks = [w for w in weeks if variances.get(w, 0) == variances.get(w, 0) and variances[w] > 0]
    stat = sum(observed[w] ** 2 / variances[w] for w in usable_weeks)
    rep_stats = []
    for rep in rep_bins:
        common = [w for w in usable_weeks if w in rep]
        if len(common) == len(usable_weeks):
            rep_stats.append(
                sum((rep[w] - observed[w]) ** 2 / variances[w] for w in common)
            )
    p_value = (
        float(np.mean([s >= stat for s in rep_stats])) if rep_stats else None
    )
    deviations = []
    for rep in rep_bins:
        common = [b for b in observed if b in rep]
        if common:
            deviations.append(max(abs(rep[b] - observed[b]) for b in common))
    band = float(np.quantile(deviations, 0.95)) if deviations else float("nan")
    joint_pass = bool(p_value is not None and p_value >= 0.05)
    mean_lead = float(np.mean(list(observed.values())))
    return {
        "leads": [{"event_time": e, "estimate": v} for e, v in sorted(leads.items())],
        "weekly_leads": [
            {"event_week": b, "estimate": v} for b, v in sorted(observed.items())
        ],
        "mean_weekly_lead": mean_lead,
        "wald_stat": stat,
        "wald_p": p_value,
        "simultaneous_band": band,
        "joint_pass": joint_pass,
        "note": "前置系数按事件周分箱，对角 Wald 自助检验 p≥0.05 视为通过；客源地很少，检验功效有限",
    }


def _balance(arr, config: DidConfig, panel: RoutePanel):
    """实施前处理组与干净对照的降雨/假日均衡性（共同冲击诊断）。"""
    frame = panel.frame
    pre = frame[frame["on"] < frame["start"]]
    if config.control_scope == "strict":
        pre = pre[~pre["exposed"]]
    treated = pre[pre["ever_treated_route"]]
    control = pre[~pre["ever_treated_route"]]
    def _avg(df, col):
        return float(df[col].mean()) if len(df) else None
    return {
        "treated_rain_mean": _avg(treated, "rain_max"),
        "control_rain_mean": _avg(control, "rain_max"),
        "treated_holiday_share": _avg(treated, "holiday"),
        "control_holiday_share": _avg(control, "holiday"),
    }


def _spillover(arr, config: DidConfig):
    """外溢的描述性证据：被波及客源地在自身推广启动前相对干净对照的变化。

    只看 [被波及日, 自有推广启动日) 的窗口，避免把自身推广效应误读为外溢。
    """
    y, usable, exposed = arr["y"], arr["usable"], arr["exposed"]
    dates = arr["dates"]
    results = []
    for i in range(len(arr["origins"])):
        exp_dates = np.where(exposed[i])[0]
        if len(exp_dates) == 0:
            continue
        e0 = int(exp_dates[0])
        own = arr["own_start_idx"][i]
        end = own if own >= 0 else len(dates)
        if e0 >= end:
            continue
        base = _base_values(arr, e0, config)
        if np.isnan(base[i]):
            continue
        diffs = []
        for t in range(e0, end):
            if not usable[i, t]:
                continue
            pool = _control_pool(arr, t, e0, base, config)
            pool = [c for c in pool if c != i]
            if len(pool) < config.min_controls:
                continue
            diffs.append(float(y[i, t] - base[i] - np.mean(y[pool, t] - base[pool])))
        if diffs:
            results.append(
                {
                    "origin": arr["origins"][i],
                    "exposed_from": dates[e0].isoformat(),
                    "window_end": dates[end - 1].isoformat(),
                    "mean_diff": float(np.mean(diffs)),
                    "n_periods": len(diffs),
                }
            )
    return {
        "items": results,
        "note": "外溢为资料确认的联系；此处仅在自身推广启动前窗口内描述，不作为归因结论",
    }


def estimate_route(panel: RoutePanel, config: DidConfig | None = None) -> RouteDid:
    config = config or DidConfig()
    arr = _arrays(panel, config)
    estimates, unidentified = _cells(arr, config)
    event_points, att = _aggregate(estimates)

    # 丢弃计数只统计已推广客源地的候选单元（基期窗口之外），与 cells_used 同口径
    censored_dropped = 0
    closed_dropped = 0
    for i in range(len(arr["origins"])):
        if not arr["ever"][i]:
            continue
        g = arr["start_idx"][i]
        lo = max(0, g - config.base_window)
        for t in range(len(arr["dates"])):
            if lo <= t < g:
                continue
            if arr["closed"][i, t]:
                closed_dropped += 1
            elif arr["censored"][i, t] and not config.include_censored:
                censored_dropped += 1

    rng = np.random.default_rng(config.seed)
    treated_idx = [i for i in range(len(arr["origins"])) if arr["ever"][i]]
    boot_atts: list[float] = []
    boot_leads: list[dict[int, float]] = []
    boot_event: dict[int, list[float]] = {}
    for _ in range(config.bootstrap):
        sample = [treated_idx[k] for k in rng.integers(0, len(treated_idx), len(treated_idx))]
        sampler = lambda n: rng.integers(0, n, n)  # noqa: E731
        rep_estimates, _ = _cells(arr, config, treated_sample=sample, control_sampler=sampler)
        rep_points, rep_att = _aggregate(rep_estimates)
        if rep_att is not None:
            boot_atts.append(rep_att)
        lead_map = {}
        for p in rep_points:
            boot_event.setdefault(p.event_time, []).append(p.estimate)
            if p.event_time < 0:
                lead_map[p.event_time] = p.estimate
        boot_leads.append(lead_map)

    ci = None
    if boot_atts:
        lo, hi = np.percentile(boot_atts, [2.5, 97.5])
        ci = (float(lo), float(hi))
    for p in event_points:
        reps = boot_event.get(p.event_time, [])
        if reps:
            lo, hi = np.percentile(reps, [2.5, 97.5])
            p.ci_low, p.ci_high = float(lo), float(hi)

    pretrend = _pretrend(event_points, boot_leads)
    notes = []
    n_treated = len(treated_idx)
    if n_treated < 4:
        notes.append("已推广客源地很少，自助法区间仅作量级参考")
    if config.control_scope == "include_spillover":
        notes.append("对照池包含资料确认的外溢客源，估计可能吸收外溢效应，识别风险需自行评估")
    if config.include_censored:
        notes.append("售罄截尾单元被保留，真实需求不小于观测值，估计存在向下偏误")
    if att is None:
        notes.append("没有可识别的处理后单元，不给出整体 ATT")

    return RouteDid(
        route=panel.route,
        att=att,
        att_ci=ci,
        event_study=event_points,
        pretrend=pretrend,
        cells_used=len(estimates),
        cells_unidentified=unidentified,
        n_treated_origins=n_treated,
        control_scope=config.control_scope,
        include_censored=config.include_censored,
        censored_cells_dropped=censored_dropped,
        closed_cells_dropped=closed_dropped,
        balance=_balance(arr, config, panel),
        spillover=_spillover(arr, config),
        notes=notes,
    )
