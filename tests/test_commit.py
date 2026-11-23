"""整团承诺引擎：全有或全无、幂等、失效、替代与不驱逐既有承诺。"""
import datetime as dt
import unittest

from tourflow.capacity import build_ledger
from tourflow.commit import CommitEngine, commitment_id, replay_commitments, seed_obligations
from tourflow.demand import Forecast
from tourflow.model import build_structure
from tourflow.policy import Policy

from tests.support import base_tables, make_materials

DAY = dt.date(2026, 10, 1)
NOW = dt.datetime(2026, 9, 18, 9, 0, tzinfo=dt.timezone.utc)


def make_forecast(origin, route, q50=40.0):
    return Forecast(
        origin=origin,
        route=route,
        on=DAY,
        lam=q50,
        quantiles={"q05": 20.0, "q10": 25.0, "q25": 30.0, "q50": q50,
                   "q75": 45.0, "q90": 50.0, "q95": 55.0},
        natural_lam=q50 * 0.6,
        uplift_lam=q50 * 0.4,
        extrapolated=False,
        notes=[],
    )


def build(people=12, substitutes=(), expires="2026-09-20T18:00:00Z", route="line-a"):
    tables = base_tables()
    tables["routes"] = [
        {"id": "line-a", "sites": ["S0", "S1"], "bus": "B0", "bus_seats": 25},
        {"id": "line-b", "sites": ["S2"], "bus": "B1", "bus_seats": 40},
    ]
    tables["supply"] = [
        {"id": "r1", "on": "2026-10-01", "site": "S0", "sold_individual": 10,
         "reserved_groups": 5, "bus_reserved": 3, "closed": False},
        {"id": "r2", "on": "2026-10-01", "site": "S1", "sold_individual": 20,
         "reserved_groups": 5, "bus_reserved": 3, "closed": False},
        {"id": "r3", "on": "2026-10-01", "site": "S2", "sold_individual": 5,
         "reserved_groups": 0, "bus_reserved": 3, "closed": False},
    ]
    tables["group_requests"] = [
        {"id": "G0", "agency": "A0", "on": "2026-10-01", "origin": "C0",
         "route": route, "people": people, "expires_at": expires,
         "allowed_substitutes": list(substitutes)},
    ]
    materials = make_materials(tables)
    structure = build_structure(materials)
    ledger = {"version": 1, "commitments": {}, "holds": {}, "events_applied": [],
              "outcomes": {}, "feedback": {"show_up": {}, "excluded_cells": []}}
    capacity = build_ledger(materials, structure)
    requests = {r["id"]: r for r in tables["group_requests"]}
    forecasts = {}
    for rid in ("line-a", "line-b"):
        forecasts[("C0", rid, DAY)] = make_forecast("C0", rid)
    engine = CommitEngine(
        structure, ledger, capacity, forecasts,
        {"line-a": "支持归因", "line-b": "支持归因"}, Policy(),
    )
    return materials, structure, engine, requests, ledger, capacity


class AcceptTests(unittest.TestCase):
    def test_accept_consumes_capacity_once(self):
        _, structure, engine, requests, ledger, capacity = build(people=12)
        first = engine.accept(requests, "G0", NOW)
        second = engine.accept(requests, "G0", NOW)
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(len(ledger["commitments"]), 1)
        # S0 保护区：30 - 15(散客/团体) - 12(本团) = 3，只扣一次
        self.assertEqual(capacity.remaining_protected[(DAY, "S0")], 3)
        self.assertEqual(capacity.remaining_bus[(DAY, "B0")], 25 - 3 - 12)

    def test_all_or_nothing(self):
        # S0 保护区只剩 15，请求 16 人 → 整团拒绝，部分场馆同意不算数
        _, structure, engine, requests, ledger, capacity = build(people=16)
        decision = engine.evaluate(requests["G0"], NOW)
        self.assertEqual(decision.status, "rejected")
        self.assertEqual(ledger["commitments"], {})
        self.assertEqual(capacity.remaining_protected[(DAY, "S1")], 40 - 25)

    def test_expired_quote_rejected(self):
        _, _, engine, requests, _, _ = build(expires="2026-09-01T00:00:00Z")
        decision = engine.evaluate(requests["G0"], NOW)
        self.assertEqual(decision.status, "expired")

    def test_substitute_route_when_requested_infeasible(self):
        # 请求线路保护区不足，但约定的替代线路可行 → 整团落到替代线路
        _, structure, engine, requests, ledger, capacity = build(
            people=16, substitutes=["line-b"]
        )
        decision = engine.evaluate(requests["G0"], NOW)
        self.assertEqual(decision.status, "accepted")
        self.assertEqual(decision.route, "line-b")
        commitment = engine.accept(requests, "G0", NOW)
        self.assertEqual(commitment["route"], "line-b")
        self.assertEqual(capacity.remaining_bus[(DAY, "B1")], 40 - 3 - 16)

    def test_existing_commitments_never_evicted(self):
        _, structure, engine, requests, ledger, capacity = build(people=12)
        engine.accept(requests, "G0", NOW)
        # 第二个请求挤占同一保护区剩余（只剩 3）
        requests["G9"] = dict(requests["G0"], id="G9", people=5)
        decision = engine.evaluate(requests["G9"], NOW)
        self.assertEqual(decision.status, "rejected")
        self.assertIn(commitment_id("G0", "line-a", "2026-10-01"), ledger["commitments"])
        self.assertEqual(ledger["commitments"][commitment_id("G0", "line-a", "2026-10-01")]["state"],
                         "confirmed")

    def test_conservatism_changes_committable(self):
        _, structure, engine, requests, _, _ = build()
        liberal = engine._committable_for(requests["G0"], "line-a")[0]
        engine.policy = Policy(conservatism=0.9)
        strict = engine._committable_for(requests["G0"], "line-a")[0]
        self.assertLess(strict, liberal)

    def test_pilot_when_evidence_insufficient(self):
        _, structure, engine, requests, _, _ = build()
        engine.evidence = {"line-a": "证据不足", "line-b": "证据不足"}
        quantity, notes, pilot = engine._committable_for(requests["G0"], "line-a")
        self.assertTrue(pilot)
        self.assertTrue(any("试行" in n for n in notes))

    def test_plan_dry_run_matches_commit(self):
        _, structure, engine, requests, ledger, capacity = build(people=12)
        requests["G9"] = dict(requests["G0"], id="G9", people=20,
                              expires_at="2026-09-21T18:00:00Z")
        dry = engine.plan(requests, NOW, commit=False)
        _, structure2, engine2, requests2, _, _ = build(people=12)
        requests2["G9"] = dict(requests2["G0"], id="G9", people=20,
                               expires_at="2026-09-21T18:00:00Z")
        committed = engine2.plan(requests2, NOW, commit=True)
        self.assertEqual(dry.accepted_people, committed.accepted_people)
        self.assertEqual(dry.rejected_people, committed.rejected_people)
        # 干跑不落账
        self.assertEqual(ledger["commitments"], {})


class SeedTests(unittest.TestCase):
    def test_seed_and_replay_idempotent(self):
        tables = base_tables()
        tables["supply"] = [
            {"id": "r1", "on": "2026-10-01", "site": "S0", "sold_individual": 0,
             "reserved_groups": 0, "bus_reserved": 0, "closed": False},
            {"id": "r2", "on": "2026-10-01", "site": "S1", "sold_individual": 0,
             "reserved_groups": 0, "bus_reserved": 0, "closed": False},
        ]
        tables["group_requests"] = [
            {"id": "G0", "agency": "A0", "on": "2026-10-01", "origin": "C0",
             "route": "line-a", "people": 10, "expires_at": "2026-09-20T18:00:00Z",
             "allowed_substitutes": []},
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
        added1 = seed_obligations(structure, ledger, materials.table("obligations"), requests)
        added2 = seed_obligations(structure, ledger, materials.table("obligations"), requests)
        self.assertEqual(len(added1), 1)
        self.assertEqual(added2, [])
        # 正确用法：容量账从供给重建后，把台账重放一次，余量只扣一次
        capacity = build_ledger(materials, structure)
        replay_commitments(structure, ledger, capacity)
        self.assertEqual(capacity.remaining_protected[(DAY, "S0")], 30 - 10)


if __name__ == "__main__":
    unittest.main()
