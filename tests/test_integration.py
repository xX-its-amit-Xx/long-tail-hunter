"""Integration tests: load examples/test_cases.json and assert each case
produces a plausibly useful plan."""
from __future__ import annotations
import json
import unittest
from pathlib import Path

from long_tail_hunter.topic import Topic, TopicKind
from long_tail_hunter.hunter import plan
from long_tail_hunter.sources import as_mcp, as_http

CASES_PATH = Path(__file__).resolve().parents[1] / "examples" / "test_cases.json"


def _load_cases():
    with open(CASES_PATH) as f:
        return json.load(f)


def _topic_from_case(case: dict) -> Topic:
    return Topic(
        term=case["term"],
        kind=TopicKind(case["kind"]),
        synonyms=case.get("synonyms", []),
        obscure_synonyms=case.get("obscure_synonyms", []),
        adjacent_methods=case.get("adjacent_methods", []),
        organism=case.get("organism"),
        famous_hits=case.get("famous_hits", []),
    )


class TestEachCaseProducesValidPlan(unittest.TestCase):
    def test_all_cases(self):
        cases = _load_cases()
        # Bulk-expansion floors (May 2026 phenotype batch + late-May
        # cross-domain batch). Each floor below catches a regression where
        # someone deletes cases without replacing them.
        self.assertGreaterEqual(len(cases), 5)
        self.assertGreaterEqual(len(cases), 11,
                                "Expected the May 2026 phenotype expansion (>= 11 cases)")
        self.assertGreaterEqual(len(cases), 30,
                                "Expected the late-May 2026 cross-domain expansion (>= 30 cases)")
        phenotype_cases = [c for c in cases if c["kind"] == "phenotype"]
        self.assertGreaterEqual(len(phenotype_cases), 5,
                                "Expected >= 5 phenotype-kind cases after expansion")
        self.assertGreaterEqual(len(phenotype_cases), 8,
                                "Expected >= 8 phenotype-kind cases after late-May expansion")
        # Every kind must have at least one case to keep the integration
        # surface diverse.
        kinds = {c["kind"] for c in cases}
        for k in ("method", "target", "phenotype"):
            self.assertIn(k, kinds, f"Missing case of kind '{k}'")
        for case in cases:
            with self.subTest(name=case["name"]):
                topic = _topic_from_case(case)
                sp = plan(topic)
                # Every case should produce a non-trivial multi-source plan.
                self.assertGreater(len(sp.queries), 15,
                                   f"{case['name']}: too few queries ({len(sp.queries)})")
                self.assertGreaterEqual(len({q.source for q in sp.queries}), 4,
                                        f"{case['name']}: too few sources covered")
                # Every query must carry its provenance.
                for q in sp.queries:
                    self.assertTrue(q.rationale)
                    self.assertTrue(q.strategy)
                # Target cases should have chemistry queries.
                if topic.kind == TopicKind.TARGET:
                    chem = [q for q in sp.queries if q.source == "chembl"]
                    self.assertGreater(len(chem), 0,
                                       f"{case['name']}: target case missing chembl queries")
                # Method cases should have methodology_focus expansion of adjacent methods.
                if topic.kind == TopicKind.METHOD and topic.adjacent_methods:
                    method_qs = sp.by_strategy("methodology_focus")
                    anchors_seen = set()
                    for q in method_qs:
                        for anchor in topic.adjacent_methods:
                            if anchor in q.text:
                                anchors_seen.add(anchor)
                    self.assertEqual(anchors_seen, set(topic.adjacent_methods),
                                     f"{case['name']}: not all adjacent methods anchored")


class TestPlansAreDispatchable(unittest.TestCase):
    """Every query should be either MCP-runnable or HTTP-runnable (or both)."""
    def test_dispatch_coverage(self):
        cases = _load_cases()
        for case in cases:
            with self.subTest(name=case["name"]):
                sp = plan(_topic_from_case(case))
                unroutable = [q for q in sp.queries
                              if as_mcp(q) is None and as_http(q) is None]
                self.assertEqual(unroutable, [],
                                 f"{case['name']}: undispatchable queries: {unroutable}")


class TestEachCaseRoutesToRealCategory(unittest.TestCase):
    """No case should fall through to the 'bioinformatics' fallback. If one
    does, the category map needs another keyword (which is fine — but make
    it explicit, don't paper it over with bioinformatics-catch-all)."""

    def test_no_case_in_bioinformatics_fallback(self):
        from long_tail_hunter.categories import categories_for
        cases = _load_cases()
        bad = []
        for case in cases:
            t = _topic_from_case(case)
            cats = categories_for(t)
            if cats == ["bioinformatics"]:
                bad.append(case["name"])
        self.assertEqual(bad, [],
                         f"Cases routed to fallback only: {bad}. "
                         "Add domain-appropriate keywords to categories.py.")


if __name__ == "__main__":
    unittest.main()
