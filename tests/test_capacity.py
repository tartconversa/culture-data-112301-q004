"""资源台账：容量扣减、共享接驳与关闭。"""
import datetime as dt
import unittest

from tourflow.capacity import build_ledger
from tourflow.model import build_structure

from tests.support import base_tables, make_materials

DAY = dt.date(2026, 10, 1)


def build(shared_bus=True):
    tables = base_tables()
    tables["routes"] = [
        {"id": "line-a", "sites": ["S0", "S1"], "bus": "B0", "bus_seats": 25},
        {"id": "line-b", "sites": ["S1"], "bus": "B0" if shared_bus else "B1",
         "bus_seats": 25},
    ]
    tables["supply"] = [
        {"id": "r1", "on": "2026-10-01", "site": "S0", "sold_individual": 10,
         "reserved_groups": 5, "bus_reserved": 3, "closed": False},
        {"id": "r2", "on": "2026-10-01", "site": "S1", "sold_individual": 20,
         "reserved_groups": 5, "bus_reserved": 3, "closed": False},
    ]
    materials = make_materials(tables)
    return materials, build_structure(materials)


class CapacityTests(unittest.TestCase):
    def test_remaining_after_deductions(self):
        materials, structure = build()
        ledger = build_ledger(materials, structure)
        # S0: 50 - 10 - 5 = 35；保护区 30 - 15 = 15
        self.assertEqual(ledger.remaining_total[(DAY, "S0")], 35)
        self.assertEqual(ledger.remaining_protected[(DAY, "S0")], 15)
        self.assertEqual(ledger.remaining_total[(DAY, "S1")], 35)

    def test_shared_bus_not_summed_per_route(self):
        materials, structure = build(shared_bus=True)
        ledger = build_ledger(materials, structure)
        # 两条线路共用 B0：25 - 3 = 22，不能按线路相加
        self.assertEqual(ledger.remaining_bus[(DAY, "B0")], 22)
        ledger.apply_route(structure, DAY, "line-a", 20)
        self.assertEqual(ledger.remaining_bus[(DAY, "B0")], 2)
        # line-b 虽只占用 S1，但接驳余量同样只剩 2
        feasible = ledger.route_feasible(structure, DAY, "line-b", 5)
        self.assertFalse(feasible.ok)

    def test_route_feasible_checks_all_resources(self):
        materials, structure = build()
        ledger = build_ledger(materials, structure)
        ok = ledger.route_feasible(structure, DAY, "line-a", 15)
        self.assertTrue(ok.ok)
        too_many = ledger.route_feasible(structure, DAY, "line-a", 20)
        self.assertFalse(too_many.ok)  # S0 保护区只剩 15
        self.assertTrue(any("保护区" in b for b in too_many.binding))

    def test_closed_site_blocks_route(self):
        tables = base_tables()
        tables["supply"] = [
            {"id": "r1", "on": "2026-10-01", "site": "S0", "sold_individual": 0,
             "reserved_groups": 0, "bus_reserved": 0, "closed": True},
            {"id": "r2", "on": "2026-10-01", "site": "S1", "sold_individual": 0,
             "reserved_groups": 0, "bus_reserved": 0, "closed": False},
        ]
        materials = make_materials(tables)
        structure = build_structure(materials)
        ledger = build_ledger(materials, structure)
        feasible = ledger.route_feasible(structure, DAY, "line-a", 5)
        self.assertFalse(feasible.ok)
        self.assertTrue(any("关闭" in b for b in feasible.binding))

    def test_apply_and_release_symmetric(self):
        materials, structure = build()
        ledger = build_ledger(materials, structure)
        before = ledger.remaining_total[(DAY, "S0")]
        ledger.apply_route(structure, DAY, "line-a", 10)
        ledger.apply_route(structure, DAY, "line-a", 10, sign=+1)
        self.assertEqual(ledger.remaining_total[(DAY, "S0")], before)


if __name__ == "__main__":
    unittest.main()
