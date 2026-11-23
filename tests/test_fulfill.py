"""后续事实：关闭替代、退款/协商、核销分类与反馈。"""
import datetime as dt
import unittest

from tourflow.capacity import build_ledger
from tourflow.commit import replay_commitments, seed_obligations
from tourflow.fulfill import (
    CANCELLED,
    FULFILLED,
    NO_SHOW,
    ROUTE_SUBSTITUTED,
    apply_events,
)
from tourflow.model import build_structure

from tests.support import base_tables, make_materials

DAY = dt.date(2026, 10, 1)
AT = "2026-10-02T09:00:00+00:00"


def build(substitutes=(), sub_capacity=True):
    tables = base_tables()
    tables["routes"] = [
        {"id": "line-a", "sites": ["S0", "S1"], "bus": "B0", "bus_seats": 25},
        {"id": "line-b", "sites": ["S2"], "bus": "B1", "bus_seats": 40},
    ]
    tables["supply"] = [
        {"id": "r1", "on": "2026-10-01", "site": "S0", "sold_individual": 0,
         "reserved_groups": 0, "bus_reserved": 0, "closed": False},
        {"id": "r2", "on": "2026-10-01", "site": "S1", "sold_individual": 0,
         "reserved_groups": 0, "bus_reserved": 0, "closed": False},
        {"id": "r3", "on": "2026-10-01", "site": "S2", "sold_individual": 0,
         "reserved_groups": 0 if sub_capacity else 44, "bus_reserved": 0,
         "closed": False},
    ]
    tables["group_requests"] = [
        {"id": "G0", "agency": "A0", "on": "2026-10-01", "origin": "C0",
         "route": "line-a", "people": 10, "expires_at": "2026-09-20T18:00:00Z",
         "allowed_substitutes": list(substitutes)},
    ]
    tables["obligations"] = [
        {"id": "OB1", "group": "G0", "accepted_on": "2026-09-10",
         "route": "line-a", "state": "confirmed"},
    ]
    materials = make_materials(tables)
    structure = build_structure(materials)
    ledger = {"version": 1, "commitments": {}, "holds": {}, "events_applied": [],
              "outcomes": {}, "feedback": {"show_up": {}, "excluded_cells": []}}
    capacity = build_ledger(materials, structure)
    requests = {r["id"]: r for r in tables["group_requests"]}
    seed_obligations(structure, ledger, materials.table("obligations"), requests)
    replay_commitments(structure, ledger, capacity)
    return materials, structure, ledger, capacity, requests


def closure_event(site="S0", on="2026-10-01"):
    return {"id": "E-close", "kind": "closure", "on": on, "site": site,
            "group": None, "actual_people": None}


class ClosureTests(unittest.TestCase):
    def test_substitution_preserves_people_and_moves_capacity(self):
        materials, structure, ledger, capacity, requests = build(substitutes=["line-b"])
        result = apply_events(structure, ledger, capacity, requests, [closure_event()], at=AT)
        self.assertEqual(len(result.substitutions), 1)
        commitment = next(iter(ledger["commitments"].values()))
        self.assertEqual(commitment["state"], "substituted")
        self.assertEqual(commitment["route"], "line-b")
        self.assertEqual(commitment["people"], 10)  # 人数不自动缩减
        # 容量随之改写：S0/S1 释放，S2 占用
        self.assertEqual(capacity.remaining_protected[(DAY, "S0")], 30)
        self.assertEqual(capacity.remaining_protected[(DAY, "S2")], 45 - 10)

    def test_refund_when_no_substitute_and_record_kept(self):
        materials, structure, ledger, capacity, requests = build(substitutes=())
        result = apply_events(structure, ledger, capacity, requests, [closure_event()], at=AT)
        self.assertEqual(len(result.refunds), 1)
        commitment = next(iter(ledger["commitments"].values()))
        # 承诺记录保留，不能简单删除
        self.assertEqual(commitment["state"], "refund")
        self.assertEqual(len(ledger["commitments"]), 1)
        # 无法履约部分的占用被释放
        self.assertEqual(capacity.remaining_protected[(DAY, "S1")], 40)

    def test_manual_review_when_substitutes_infeasible(self):
        materials, structure, ledger, capacity, requests = build(
            substitutes=["line-b"], sub_capacity=False
        )
        result = apply_events(structure, ledger, capacity, requests, [closure_event()], at=AT)
        self.assertEqual(len(result.manual_reviews), 1)
        commitment = next(iter(ledger["commitments"].values()))
        self.assertEqual(commitment["state"], "manual_review")

    def test_events_applied_once(self):
        materials, structure, ledger, capacity, requests = build(substitutes=["line-b"])
        events = [closure_event()]
        apply_events(structure, ledger, capacity, requests, events, at=AT)
        again = apply_events(structure, ledger, capacity, requests, events, at=AT)
        self.assertEqual(again.substitutions, [])
        self.assertEqual(len(ledger["events_applied"]), 1)


class RedemptionTests(unittest.TestCase):
    def _apply(self, events, substitutes=()):
        materials, structure, ledger, capacity, requests = build(substitutes=substitutes)
        apply_events(structure, ledger, capacity, requests, events, at=AT)
        return ledger, requests

    def test_fulfilled_when_redeemed_on_committed_route(self):
        ledger, _ = self._apply([
            {"id": "E1", "kind": "redeem", "on": "2026-10-01", "site": "S0",
             "group": "G0", "actual_people": 9},
        ])
        outcome = ledger["outcomes"]["G0"]
        self.assertEqual(outcome["classification"], FULFILLED)
        self.assertAlmostEqual(outcome["show_up_rate"], 0.9)
        # 核销率回写到客源地的需求反馈
        self.assertAlmostEqual(ledger["feedback"]["show_up"]["C0"], 0.9)

    def test_no_show_when_no_redemption(self):
        ledger, _ = self._apply([])
        # 没有任何回执时还没有 outcome；补一条交通以外的空分类路径
        from tourflow.fulfill import _classify
        ledger["outcomes"]["G0"] = {"redeemed": {}, "transport": {}, "classification": None,
                                    "history": []}
        requests = {"G0": {"origin": "C0"}}
        materials, structure, _, _, _ = build()
        _classify(ledger, requests, structure, "G0")
        self.assertEqual(ledger["outcomes"]["G0"]["classification"], NO_SHOW)

    def test_route_substitution_detected_from_sites(self):
        ledger, _ = self._apply([
            {"id": "E1", "kind": "redeem", "on": "2026-10-01", "site": "S2",
             "group": "G0", "actual_people": 10},
        ])
        self.assertEqual(ledger["outcomes"]["G0"]["classification"], ROUTE_SUBSTITUTED)

    def test_cancelled_after_refund(self):
        ledger, _ = self._apply([closure_event()])
        self.assertEqual(ledger["outcomes"].get("G0", {}).get("classification"), None)
        from tourflow.fulfill import _classify
        ledger["outcomes"]["G0"] = {"redeemed": {}, "transport": {}, "classification": None,
                                    "history": []}
        materials, structure, _, _, _ = build()
        _classify(ledger, {"G0": {"origin": "C0"}}, structure, "G0")
        self.assertEqual(ledger["outcomes"]["G0"]["classification"], CANCELLED)

    def test_closure_recorded_in_excluded_cells(self):
        ledger, _ = self._apply([closure_event()], substitutes=["line-b"])
        self.assertIn({"on": "2026-10-01", "site": "S0"},
                      ledger["feedback"]["excluded_cells"])


if __name__ == "__main__":
    unittest.main()
