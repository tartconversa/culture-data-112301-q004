"""运行登记、承诺溯源与可承诺量变化分解。"""
import unittest

from tourflow.audit import RunRegistry, commitment_lineage, explain_committable_change


class RegistryTests(unittest.TestCase):
    def test_record_and_get(self):
        registry = RunRegistry({"runs": []})
        run_id = registry.record("forecast", {"seed": 1}, {"n": 3}, "fp", at="2026-09-18T00:00:00Z")
        self.assertTrue(run_id.startswith("forecast-"))
        self.assertEqual(registry.get(run_id)["config"], {"seed": 1})
        self.assertIsNone(registry.get("missing"))

    def test_run_ids_deterministic_given_same_sequence(self):
        a = RunRegistry({"runs": []})
        b = RunRegistry({"runs": []})
        ida = a.record("analyze", {"x": 1}, {"y": 2}, "fp", at="t")
        idb = b.record("analyze", {"x": 1}, {"y": 2}, "fp", at="t")
        self.assertEqual(ida, idb)


class LineageTests(unittest.TestCase):
    def test_lineage_links_runs_and_outcome(self):
        registry = RunRegistry({"runs": []})
        run_id = registry.record("forecast", {}, {"n": 1}, "fp", at="t")
        commitment = {
            "id": "CM-G0-line-a-2026-10-01",
            "group": "G0",
            "route": "line-a",
            "on": "2026-10-01",
            "people": 10,
            "state": "confirmed",
            "source": "engine",
            "basis": {"forecast_run": run_id, "policy": {"conservatism": 0.5}},
            "history": [{"ts": "t", "action": "accepted"}],
        }
        outcomes = {"G0": {"classification": "已核销", "committed_people": 10,
                           "actual_people": 9, "show_up_rate": 0.9}}
        lineage = commitment_lineage(commitment, registry, outcomes)
        self.assertEqual(lineage["runs"]["forecast_run"]["id"], run_id)
        self.assertEqual(lineage["outcome"]["actual_people"], 9)


class ExplainTests(unittest.TestCase):
    def test_decomposition_sums_to_total_change(self):
        key = ("C0", "line-a", "2026-10-01")
        before = {key: 10}
        after = {key: 4}
        capacity_before = {key: 20}
        capacity_after = {key: 8}   # 容量收紧
        demand_before = {key: 10}
        demand_after = {key: 12}    # 需求上限反而变宽
        rows = explain_committable_change(
            before, after,
            capacity_before=capacity_before, capacity_after=capacity_after,
            demand_before=demand_before, demand_after=demand_after,
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["capacity_effect"] + row["demand_effect"], 4 - 10)
        # 先换容量：min(8, 10) - 10 = -2；再换需求：4 - 8 = -4
        self.assertEqual(row["capacity_effect"], -2)
        self.assertEqual(row["demand_effect"], -4)

    def test_unchanged_keys_not_reported(self):
        key = ("C0", "line-a", "2026-10-01")
        rows = explain_committable_change(
            {key: 5}, {key: 5},
            capacity_before={key: 9}, capacity_after={key: 9},
            demand_before={key: 5}, demand_after={key: 5},
        )
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
