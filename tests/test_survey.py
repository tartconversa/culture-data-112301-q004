"""唯一游客估计：跨馆去重、linked/unlinked 区分与完全可观测场次。"""
import unittest

from tourflow.model import build_structure
from tourflow.survey import route_trip_stats, unique_visitors

from tests.support import base_tables, make_materials


def survey_record(rid, person, on, origin, sites, prob, linked):
    return {
        "id": rid,
        "person": person,
        "on": on,
        "origin": origin,
        "sites": sites,
        "inclusion_probability": prob,
        "linked": linked,
    }


def panel_cell(origin, on, site, coverage):
    return {
        "id": f"{origin}-{on}-{site}",
        "on": on,
        "origin": origin,
        "site": site,
        "visits": 10,
        "ticket_capacity": 50,
        "sold_out": False,
        "rain_mm": 0.0,
        "holiday": False,
        "closed": False,
        "survey_coverage": coverage,
    }


def build(survey, coverage=0.5):
    tables = base_tables()
    tables["itinerary_survey"] = survey
    tables["daily_panel"] = [
        panel_cell("C0", "2026-05-01", s, coverage) for s in ("S0", "S1")
    ] + [panel_cell("C0", "2026-05-02", s, coverage) for s in ("S0", "S1")]
    materials = make_materials(tables)
    return materials, build_structure(materials)


class UniqueVisitorTests(unittest.TestCase):
    def test_linked_person_deduplicated_across_days(self):
        survey = [
            survey_record("J1", "V1", "2026-05-01", "C0", ["S0", "S1"], 0.5, True),
            survey_record("J2", "V1", "2026-05-02", "C0", ["S0", "S1"], 0.5, True),
        ]
        materials, structure = build(survey)
        estimate = unique_visitors(materials, structure, "C0", "line-a")
        # 同一匿名人跨日只计一次：1 / 0.5 / 0.5 = 4
        self.assertEqual(estimate.point, 4.0)
        self.assertEqual(estimate.low, estimate.high)

    def test_unlinked_records_give_interval_not_point(self):
        survey = [
            survey_record("J1", "V1", "2026-05-01", "C0", ["S0", "S1"], 0.5, True),
            survey_record("J2", "V9", "2026-05-01", "C0", ["S0", "S1"], 0.5, False),
            survey_record("J3", "V9", "2026-05-02", "C0", ["S0", "S1"], 0.5, False),
        ]
        materials, structure = build(survey)
        estimate = unique_visitors(materials, structure, "C0", "line-a")
        self.assertIsNone(estimate.point)  # 有 unlinked，不能给点估计
        self.assertLess(estimate.low, estimate.high)
        # linked 部分 4.0；unlinked 两日各 4.0 → [4, 8]
        self.assertEqual(estimate.low, 4.0 + 4.0)
        self.assertEqual(estimate.high, 4.0 + 8.0)

    def test_fully_observable_sessions_kept_distinct(self):
        survey = [
            survey_record("J1", "V1", "2026-05-01", "C0", ["S0", "S1"], 1.0, True),
        ]
        materials, structure = build(survey, coverage=1.0)
        estimate = unique_visitors(materials, structure, "C0", "line-a")
        self.assertTrue(estimate.fully_observable)
        materials2, structure2 = build(survey, coverage=0.5)
        estimate2 = unique_visitors(materials2, structure2, "C0", "line-a")
        self.assertFalse(estimate2.fully_observable)

    def test_uncovered_route_reports_gap(self):
        tables = base_tables()
        tables["routes"].append(
            {"id": "line-z", "sites": ["S2"], "bus": "B9", "bus_seats": 10}
        )
        materials = make_materials(tables)
        structure = build_structure(materials)
        stats = route_trip_stats(materials, structure, "C0", "line-z")
        self.assertFalse(stats.covered)
        self.assertIsNone(stats.visits_per_person)
        estimate = unique_visitors(materials, structure, "C0", "line-z")
        self.assertIsNone(estimate.point)

    def test_visits_per_person_weighted(self):
        survey = [
            survey_record("J1", "V1", "2026-05-01", "C0", ["S0", "S1"], 0.5, True),
            survey_record("J2", "V2", "2026-05-01", "C0", ["S0", "S1"], 0.5, True),
        ]
        materials, structure = build(survey)
        stats = route_trip_stats(materials, structure, "C0", "line-a")
        self.assertTrue(stats.covered)
        self.assertAlmostEqual(stats.visits_per_person, 2.0)


if __name__ == "__main__":
    unittest.main()
