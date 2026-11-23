"""业务使用路径的端到端可复验测试。

用仓库中的合成资料，在临时目录里依次跑：
validate → analyze → forecast → plan --commit → fulfill → forecast → audit，
并校验关键业务不变量。各测试只读取产出，执行顺序在 setUpClass 中固定。
"""
import json
import tempfile
import unittest
from pathlib import Path

from tourflow.capacity import build_ledger
from tourflow.cli import main
from tourflow.commit import replay_commitments, seed_obligations
from tourflow.fulfill import apply_feedback_closures
from tourflow.materials import load
from tourflow.model import build_structure

ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-09-18T09:00:00Z"


def run_cli(state, out, *args):
    argv = [
        "--root", str(ROOT),
        "--state-dir", str(state),
        "--output-dir", str(out),
        "--bootstrap", "100",
        "--now", NOW,
        *args,
    ]
    return main(argv)


class BusinessPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.state, cls.out = tmp / "state", tmp / "out"
        steps = (
            ["validate"], ["analyze"], ["forecast"], ["plan", "--commit"],
            ["fulfill"], ["forecast"], ["audit"],
        )
        for step in steps:
            rc = run_cli(cls.state, cls.out, *step)
            assert rc == 0, step

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def read(self, name):
        return json.loads((self.out / name).read_text(encoding="utf-8"))

    def ledger(self):
        return json.loads((self.state / "ledger.json").read_text(encoding="utf-8"))

    def test_reports_written(self):
        for name in ("incrementality", "forecast", "plan", "fulfill", "audit"):
            self.assertTrue((self.out / f"{name}.json").exists(), name)
            self.assertTrue((self.out / f"{name}.md").exists(), name)

    def test_incrementality_has_verdicts_and_gaps(self):
        report = self.read("incrementality.json")
        self.assertEqual(set(report["routes"]), {"line-a", "line-b", "line-c"})
        for route in report["routes"].values():
            self.assertIn(route["verdict"], ("支持归因", "证据不足", "无法识别"))
            self.assertIn("pretrend", route)
        # 唯一游客增量不得超过到访次数增量（不允许跨馆重复计人）
        for route in report["routes"].values():
            if route["att_unique"] and route["att_visits_ci"]:
                self.assertLessEqual(route["att_unique"][1],
                                     max(abs(route["att_visits_ci"][0]),
                                         abs(route["att_visits_ci"][1])) + 1e-9)

    def test_plan_decisions_consistent_with_capacity(self):
        plan = self.read("plan.json")
        self.assertGreater(plan["accepted_people"], 0)
        # 重复接受只产生一次承诺：G0/G3 已有历史承诺
        duplicates = [d for d in plan["decisions"] if d["status"] == "duplicate"]
        self.assertEqual({d["group"] for d in duplicates}, {"G0", "G3"})
        # 台账与容量一致：重建容量账并重放承诺后，所有余量非负
        materials = load(ROOT)
        structure = build_structure(materials)
        ledger = self.ledger()
        capacity = build_ledger(materials, structure)
        requests = {r["id"]: r for r in materials.table("group_requests")}
        seed_obligations(structure, ledger, materials.table("obligations"), requests)
        replay_commitments(structure, ledger, capacity)
        apply_feedback_closures(capacity, ledger)
        for mapping in (capacity.remaining_total, capacity.remaining_protected,
                        capacity.remaining_bus):
            for value in mapping.values():
                self.assertGreaterEqual(value, 0)

    def test_accept_is_idempotent_across_runs(self):
        before = self.ledger()
        n_before = len(before["commitments"])
        rc = run_cli(self.state, self.out, "accept", "--group", "G1")
        self.assertEqual(rc, 0)
        after = self.ledger()
        self.assertEqual(len(after["commitments"]), n_before)

    def test_fulfill_substitutes_and_classifies(self):
        report = self.read("fulfill.json")
        # S0 在 2026-10-01 临时关闭：G0 的 line-a 承诺应被替代到约定的 line-b
        self.assertTrue(any(s["group"] == "G0" and s["to"] == "line-b"
                            for s in report["substitutions"]))
        outcomes = report["outcomes"]
        self.assertEqual(outcomes["G3"]["classification"], "已核销")
        self.assertEqual(outcomes["G3"]["actual_people"], 19)
        # 核销反馈进入台账，供下一轮预测与观察期使用
        self.assertIn("C3", report["feedback"]["show_up"])
        self.assertIn({"on": "2026-10-01", "site": "S0"},
                      report["feedback"]["excluded_cells"])

    def test_audit_traces_commitment_basis(self):
        audit = self.read("audit.json")
        by_group = {item["group"]: item for item in audit["items"]}
        # 引擎承诺能追到分析与策略
        engine_items = [i for i in audit["items"] if i["source"] == "engine"]
        self.assertTrue(engine_items)
        for item in engine_items:
            self.assertIn("forecast_run", item["basis"])
            self.assertIn("incrementality_run", item["basis"])
        # 历史承诺如实标注无分析依据
        self.assertEqual(by_group["G0"]["source"], "seed")
        # G0 的替代历史可追查
        actions = [h["action"] for h in by_group["G0"]["history"]]
        self.assertIn("substituted", actions)

    def test_committable_change_explained(self):
        # 履约后跑了第二轮预测：审计应分解两轮可承诺量变化
        audit = self.read("audit.json")
        change = audit["committable_change"]
        self.assertTrue(change, "第二轮预测后应存在可承诺量变化（关闭与承诺占用）")
        for row in change:
            self.assertEqual(
                row["capacity_effect"] + row["demand_effect"],
                row["after"] - row["before"],
            )
        # S0 关闭使 line-a 在 2026-10-01 的可承诺量降为 0
        closed_keys = [r for r in change
                       if r["key"].endswith("line-a|2026-10-01") and r["after"] == 0]
        self.assertTrue(closed_keys)


if __name__ == "__main__":
    unittest.main()
