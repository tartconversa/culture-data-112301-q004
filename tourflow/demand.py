"""可解释需求预测：自然需求 + 推广变化 + 售罄截尾。

每个 客源地×线路 拟合一个截尾泊松模型：
    log λ = β0 + β1·(距首日天数/30) + β2·推广后 + β3·假日 + β4·降雨
- 售罄日的真实需求不小于观测值，按右截尾处理（贡献 P(Y ≥ y) 而非 P(Y = y)），
  忽略截尾会系统性低估需求；
- 关闭日机械为零，不进入拟合；
- 自然需求 = 令“推广后”为 0 的 λ；推广变化 = 两者之差。推广变化是否
  可计入承诺，由增量测算的归因判定决定（见 policy）。
- 预测区间同时含参数不确定（Hessian 近似）与泊松抽样噪声，用固定种子
  的蒙特卡洛分位数，保证可复验。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from .materials import Materials
from .model import Structure
from .panel import build_route_panel

FEATURES = ("intercept", "trend_per_30d", "post", "holiday", "rain_mm")


@dataclass
class DemandModel:
    origin: str
    route: str
    theta: np.ndarray
    cov: np.ndarray
    n_obs: int
    n_censored: int
    first_date: dt.date
    last_date: dt.date
    start: dt.date | None
    loglik: float
    converged: bool

    def design(self, day: dt.date, post: bool, holiday: bool, rain_mm: float) -> np.ndarray:
        return np.array(
            [
                1.0,
                (day - self.first_date).days / 30.0,
                1.0 if post else 0.0,
                1.0 if holiday else 0.0,
                rain_mm,
            ]
        )


def _nll_and_grad(theta, X, y, censored):
    lam = np.exp(np.clip(X @ theta, -20, 20))
    nll = 0.0
    grad = np.zeros_like(theta)
    unc = ~censored
    if unc.any():
        # -log pmf = λ - y log λ + log(y!)
        nll += float(np.sum(lam[unc] - y[unc] * np.log(lam[unc])))
        grad += X[unc].T @ (lam[unc] - y[unc])
    if censored.any():
        yc = y[censored]
        lamc = lam[censored]
        sf = np.clip(poisson.sf(yc - 1, lamc), 1e-12, None)
        pmf = poisson.pmf(yc - 1, lamc)
        nll -= float(np.sum(np.log(sf)))
        # d/dθ log sf(y-1) = λ · pmf(y-1)/sf(y-1) · x
        grad -= X[censored].T @ (lamc * pmf / sf)
    return nll, grad


def _fit(origin, route, rows, first_date, start):
    X, y, cen = [], [], []
    for row in rows:
        day = row["on"]
        X.append(
            [
                1.0,
                (day - first_date).days / 30.0,
                1.0 if (start is not None and day >= start) else 0.0,
                1.0 if row["holiday"] else 0.0,
                row["rain_max"],
            ]
        )
        y.append(row["visits"])
        cen.append(row["censored"])
    X = np.array(X)
    y = np.array(y, dtype=float)
    cen = np.array(cen, dtype=bool)
    result = minimize(
        _nll_and_grad,
        np.zeros(X.shape[1]),
        args=(X, y, cen),
        jac=True,
        method="BFGS",
        options={"maxiter": 500},
    )
    cov = result.hess_inv if hasattr(result, "hess_inv") else np.full((5, 5), np.nan)
    cov = np.asarray(cov, dtype=float)
    return DemandModel(
        origin=origin,
        route=route,
        theta=result.x,
        cov=cov,
        n_obs=int(len(y)),
        n_censored=int(cen.sum()),
        first_date=first_date,
        last_date=max(r["on"] for r in rows),
        start=start,
        loglik=float(-result.fun),
        converged=bool(result.success),
    )


def fit_demand(
    materials: Materials,
    structure: Structure,
    *,
    cutoff: dt.date | None = None,
) -> dict[tuple[str, str], DemandModel]:
    """拟合全部 客源地×线路 组合；cutoff 之后的数据不进入拟合（回测用）。"""
    models: dict[tuple[str, str], DemandModel] = {}
    starts = {o: c.start for o, c in structure.campaign_by_origin.items()}
    for route_id in structure.routes:
        panel = build_route_panel(materials, structure, route_id)
        frame = panel.frame[~panel.frame["closed_any"]]
        if cutoff is not None:
            frame = frame[frame["on"] <= cutoff]
        first_date = min(panel.dates) if cutoff is None else min(
            d for d in panel.dates if d <= cutoff
        )
        for origin, group in frame.groupby("origin"):
            rows = list(group.to_dict("records"))
            if len(rows) < 30:
                continue
            models[(origin, route_id)] = _fit(
                origin, route_id, rows, first_date, starts.get(origin)
            )
    return models


@dataclass
class Forecast:
    origin: str
    route: str
    on: dt.date
    lam: float
    quantiles: dict[str, float]
    natural_lam: float
    uplift_lam: float
    extrapolated: bool
    notes: list[str] = field(default_factory=list)


def predict(
    model: DemandModel,
    dates: list[dt.date],
    *,
    holiday_dates: set[dt.date] | None = None,
    rain_by_date: dict[dt.date, float] | None = None,
    seed: int = 20261123,
    draws: int = 4000,
) -> list[Forecast]:
    """对未来日期给出需求分布；推广后情景按该客源地推广已启动处理。"""
    holiday_dates = holiday_dates or set()
    rain_by_date = rain_by_date or {}
    rng = np.random.default_rng(seed)
    out = []
    for day in dates:
        post = model.start is not None and day >= model.start
        x = model.design(day, post, day in holiday_dates, rain_by_date.get(day, 0.0))
        x0 = model.design(day, False, day in holiday_dates, rain_by_date.get(day, 0.0))
        log_lam = float(x @ model.theta)
        lam = float(np.exp(log_lam))
        natural = float(np.exp(x0 @ model.theta))
        se = float(np.sqrt(max(x @ model.cov @ x, 0.0))) if model.cov.shape == (5, 5) else 0.0
        # 参数不确定 + 泊松噪声的蒙特卡洛分位数
        lam_draws = np.exp(rng.normal(log_lam, se, draws))
        y_draws = rng.poisson(np.clip(lam_draws, 1e-9, None))
        qs = np.quantile(y_draws, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        notes = []
        extrapolated = day > model.last_date
        if extrapolated:
            notes.append("预测日期超出拟合窗口，趋势外推风险自负")
        if model.n_censored:
            notes.append(f"历史含 {model.n_censored} 个售罄截尾观测，已按右截尾处理")
        out.append(
            Forecast(
                origin=model.origin,
                route=model.route,
                on=day,
                lam=lam,
                quantiles={
                    "q05": float(qs[0]),
                    "q10": float(qs[1]),
                    "q25": float(qs[2]),
                    "q50": float(qs[3]),
                    "q75": float(qs[4]),
                    "q90": float(qs[5]),
                    "q95": float(qs[6]),
                },
                natural_lam=natural,
                uplift_lam=lam - natural,
                extrapolated=extrapolated,
                notes=notes,
            )
        )
    return out


def backtest(
    materials: Materials,
    structure: Structure,
    *,
    cutoff: dt.date,
    horizon_days: int = 28,
    seed: int = 20261123,
) -> dict:
    """时间外推回测：cutoff 前拟合，评估其后 horizon 天在非截尾单元上的表现。"""
    models = fit_demand(materials, structure, cutoff=cutoff)
    holiday_dates = {
        dt.date.fromisoformat(r["on"])
        for r in materials.table("daily_panel")
        if r["holiday"]
    }
    end = cutoff + dt.timedelta(days=horizon_days)
    per_key: dict[tuple[str, str], dict] = {}
    all_rows = []
    for (origin, route), model in models.items():
        panel = build_route_panel(materials, structure, route)
        actual = panel.frame[
            (panel.frame["origin"] == origin)
            & (panel.frame["on"] > cutoff)
            & (panel.frame["on"] <= end)
            & (~panel.frame["closed_any"])
            & (~panel.frame["censored"])
        ]
        if actual.empty:
            continue
        dates = sorted(actual["on"])
        forecasts = {
            f.on: f
            for f in predict(
                model, dates, holiday_dates=holiday_dates, seed=seed
            )
        }
        errs, covers = [], []
        for row in actual.itertuples():
            f = forecasts[row.on]
            errs.append(f.lam - row.visits)
            covers.append(
                f.quantiles["q10"] <= row.visits <= f.quantiles["q90"]
            )
        per_key[(origin, route)] = {
            "n": len(errs),
            "mae": float(np.mean(np.abs(errs))),
            "bias": float(np.mean(errs)),
            "coverage_80": float(np.mean(covers)),
        }
        all_rows.extend(zip(errs, covers))
    pooled = None
    if all_rows:
        errs = np.array([e for e, _ in all_rows])
        covers = np.array([c for _, c in all_rows])
        pooled = {
            "n": int(len(errs)),
            "mae": float(np.mean(np.abs(errs))),
            "bias": float(np.mean(errs)),
            "coverage_80": float(np.mean(covers)),
        }
    return {
        "cutoff": cutoff.isoformat(),
        "horizon_days": horizon_days,
        "per_origin_route": {f"{o}|{r}": v for (o, r), v in sorted(per_key.items())},
        "pooled": pooled,
    }
