"""双重差分估计的可复验测试：已知效应应被找回，识别纪律应被执行。"""
import datetime as dt
import unittest

from tourflow.did import DidConfig, _arrays, _control_pool, estimate_route

from tests.support import make_route_panel, panel_row

D0 = dt.date(2026, 3, 1)


def d(n):
    return D0 + dt.timedelta(days=n)


def build_dgp(effect=5.0, *, n_days=100, treated=("T1", "T2"), controls=("N1", "N2", "N3"),
              start=60, exposed_from=None, pre_divergence=0.0, censored_cells=(),
              noise=1.0):
    """构造已知真值的合成面板：共同水平 + 处理后 +effect + 确定性噪声。"""
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    starts = {o: d(start) for o in treated}
    starts.update({o: d(10**6) for o in controls})  # 对照永不推广
    levels = {o: 20.0 + 3 * i for i, o in enumerate([*treated, *controls])}
    jitters = {
        (o, t): noise * rng.standard_normal()
        for o in [*treated, *controls]
        for t in range(n_days)
    }
    for o in [*treated, *controls]:
        for t in range(n_days):
            day = d(t)
            y = levels[o] + jitters[(o, t)]
            if pre_divergence and o in treated:
                y += pre_divergence * max(0, start - t) / 30.0
            if o in treated and t >= start:
                y += effect
            rows.append(
                panel_row(
                    o, day, y,
                    start=starts[o],
                    promoted=o in treated,
                    censored=(o, t) in censored_cells,
                    exposed=bool(exposed_from and o in exposed_from and t >= exposed_from[o]),
                )
            )
    # 永不推广的对照：start 设到面板外，ever_treated_route=False
    for row in rows:
        if row["origin"] in controls:
            row["ever_treated_route"] = False
            row["route_promoted"] = False
            row["treated"] = False
    return make_route_panel(rows, origins=tuple([*treated, *controls]),
                            dates=tuple(d(t) for t in range(n_days)))


class DidRecoveryTests(unittest.TestCase):
    def test_recovers_known_effect(self):
        panel = build_dgp(effect=5.0)
        result = estimate_route(panel, DidConfig(bootstrap=200, seed=7))
        self.assertIsNotNone(result.att)
        self.assertAlmostEqual(result.att, 5.0, delta=0.6)
        self.assertIsNotNone(result.att_ci)
        self.assertLess(result.att_ci[0], 5.0)
        self.assertGreater(result.att_ci[1], 5.0)

    def test_pretrend_passes_on_parallel_dgp(self):
        panel = build_dgp(effect=5.0)
        result = estimate_route(panel, DidConfig(bootstrap=200, seed=7))
        weekly = [abs(w["estimate"]) for w in result.pretrend["weekly_leads"]]
        self.assertTrue(all(v < 1.0 for v in weekly))

    def test_pretrend_flags_diverging_dgp(self):
        panel = build_dgp(effect=5.0, pre_divergence=6.0)
        result = estimate_route(panel, DidConfig(bootstrap=200, seed=7))
        weekly = [abs(w["estimate"]) for w in result.pretrend["weekly_leads"]]
        self.assertTrue(any(v > 1.0 for v in weekly))

    def test_censored_cells_excluded_in_primary_spec(self):
        censored = {("T1", 70), ("T1", 71), ("T2", 72)}
        panel = build_dgp(effect=5.0, censored_cells=censored)
        primary = estimate_route(panel, DidConfig(bootstrap=50, seed=7))
        sensitivity = estimate_route(
            panel, DidConfig(bootstrap=50, seed=7, include_censored=True)
        )
        self.assertEqual(primary.censored_cells_dropped, 3)
        self.assertEqual(sensitivity.censored_cells_dropped, 0)
        self.assertGreater(sensitivity.cells_used, primary.cells_used)

    def test_unidentified_when_no_clean_controls(self):
        # 全部客源地都在第 30 天前启动推广：之后没有干净对照
        panel = build_dgp(effect=5.0, treated=("T1", "T2"), controls=(), start=30)
        result = estimate_route(panel, DidConfig(bootstrap=50, seed=7))
        post = [p for p in result.event_study if p.event_time >= 0]
        self.assertEqual(post, [])
        self.assertIsNone(result.att)
        self.assertTrue(result.cells_unidentified)


class ControlPoolTests(unittest.TestCase):
    def test_pool_excludes_started_campaigns_and_spillover(self):
        panel = build_dgp(
            effect=5.0,
            controls=("N1", "N2"),
            exposed_from={"N2": 30},
        )
        config = DidConfig()
        arr = _arrays(panel, config)
        base = [0.0] * len(arr["origins"])
        origins = arr["origins"]
        # 第 70 天：T1/T2 已推广、N2 已被外溢 → 只剩 N1
        pool = _control_pool(arr, 70, 60, base, config)
        self.assertEqual([origins[c] for c in pool], ["N1"])
        # include_spillover 时 N2 回到对照池，但报告必须标注风险
        loose = _control_pool(arr, 70, 60, base, DidConfig(control_scope="include_spillover"))
        self.assertEqual(sorted(origins[c] for c in loose), ["N1", "N2"])

    def test_pool_requires_usable_cells(self):
        censored = {("N1", 70)}
        panel = build_dgp(effect=5.0, controls=("N1", "N2"), censored_cells=censored)
        config = DidConfig()
        arr = _arrays(panel, config)
        base = [0.0] * len(arr["origins"])
        pool = _control_pool(arr, 70, 60, base, config)
        self.assertEqual([arr["origins"][c] for c in pool], ["N2"])


if __name__ == "__main__":
    unittest.main()
