"""需求模型：截尾泊松找回已知参数，忽略截尾会低估，回测可复验。"""
import datetime as dt
import unittest

import numpy as np

from tourflow.demand import _fit, predict

D0 = dt.date(2026, 1, 1)


def simulate(beta=(2.0, 0.3, 0.5), n=160, censor_share=0.3, seed=1,
             censor_post_extra=0.0):
    """按已知系数生成右截尾泊松数据；售罄日的观测低于真实需求。

    censor_post_extra 模拟需求高涨期更容易售罄（截尾与处理相关）。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n):
        day = D0 + dt.timedelta(days=t)
        post = 1.0 if t >= n // 2 else 0.0
        x = np.array([1.0, t / 30.0, post])
        lam = float(np.exp(x @ np.array(beta)))
        demand = rng.poisson(lam)
        share = censor_share + (censor_post_extra if post else 0.0)
        censored = rng.random() < share
        observed = int(demand * 0.6) if censored else int(demand)
        rows.append(
            {
                "on": day,
                "visits": observed,
                "censored": censored,
                "holiday": False,
                "rain_max": 0.0,
            }
        )
    return rows


class CensoredPoissonTests(unittest.TestCase):
    def test_recovers_known_coefficients(self):
        rows = simulate()
        model = _fit("C0", "line-a", rows, D0, D0 + dt.timedelta(days=80))
        self.assertTrue(model.converged)
        # [截距, 趋势/30天, 推广后]
        self.assertAlmostEqual(model.theta[0], 2.0, delta=0.15)
        self.assertAlmostEqual(model.theta[1], 0.3, delta=0.2)
        self.assertAlmostEqual(model.theta[2], 0.5, delta=0.15)

    def test_ignoring_censoring_biases_downward(self):
        # 推广后需求高、售罄更多：忽略截尾会低估推广效应
        rows = simulate(censor_share=0.1, censor_post_extra=0.5)
        start = D0 + dt.timedelta(days=80)
        censored_fit = _fit("C0", "line-a", rows, D0, start)
        naive_rows = [dict(r, censored=False) for r in rows]
        naive_fit = _fit("C0", "line-a", naive_rows, D0, start)
        # 真实推广效应 0.5；忽略截尾的估计应明显更低
        self.assertGreater(censored_fit.theta[2], naive_fit.theta[2])
        self.assertAlmostEqual(censored_fit.theta[2], 0.5, delta=0.15)
        self.assertLess(naive_fit.theta[2], 0.45)

    def test_prediction_quantiles_ordered(self):
        rows = simulate()
        model = _fit("C0", "line-a", rows, D0, D0 + dt.timedelta(days=80))
        future = [D0 + dt.timedelta(days=200 + i) for i in range(5)]
        forecasts = predict(model, future, seed=3)
        for f in forecasts:
            q = f.quantiles
            self.assertLessEqual(q["q05"], q["q50"])
            self.assertLessEqual(q["q50"], q["q95"])
            self.assertTrue(f.extrapolated)
            self.assertGreater(f.uplift_lam, 0)

    def test_predictions_reproducible(self):
        rows = simulate()
        model = _fit("C0", "line-a", rows, D0, D0 + dt.timedelta(days=80))
        future = [D0 + dt.timedelta(days=200)]
        a = predict(model, future, seed=3)[0].quantiles
        b = predict(model, future, seed=3)[0].quantiles
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
