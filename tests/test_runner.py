"""Tests for the dispatcher, category mapping, and client-side filter."""
from __future__ import annotations
import unittest

from long_tail_hunter.topic import Topic, TopicKind
from long_tail_hunter.hunter import plan
from long_tail_hunter.runner import (
    build_dispatches, filter_biorxiv_results, summarise_dispatches,
)
from long_tail_hunter.categories import categories_for, BIORXIV_CATEGORIES


class TestCategoryMapping(unittest.TestCase):
    def test_falls_back_to_bioinformatics(self):
        topic = Topic(term="completely-unrecognised-jargon", kind=TopicKind.MIXED)
        self.assertEqual(categories_for(topic), ["bioinformatics"])

    def test_crispr_hits_multiple_categories(self):
        topic = Topic(term="CRISPR screen analysis", kind=TopicKind.METHOD)
        cats = categories_for(topic)
        self.assertIn("genetics", cats)
        self.assertIn("genomics", cats)
        self.assertIn("bioinformatics", cats)

    def test_brd4_target_pulls_pharma(self):
        topic = Topic(term="BRD4", kind=TopicKind.TARGET, synonyms=["bromodomain"])
        cats = categories_for(topic)
        self.assertIn("pharmacology and toxicology", cats)
        self.assertIn("cancer biology", cats)

    def test_all_returned_cats_are_valid(self):
        topic = Topic(term="CRISPR base editor neoantigen plant",
                      synonyms=["base editing", "scRNA-seq"])
        for c in categories_for(topic):
            self.assertIn(c, BIORXIV_CATEGORIES, f"Unknown category: {c}")


class TestDispatchBuilder(unittest.TestCase):
    def test_paperclip_queries_batched(self):
        sp = plan("CRISPR screen analysis")
        ds = build_dispatches(sp)
        paperclip_ds = [d for d in ds if "Paperclip" in d.tool]
        paperclip_origin_count = sum(len(d.origin) for d in paperclip_ds)
        paperclip_in_plan = sum(1 for q in sp.queries if q.source == "paperclip")
        self.assertEqual(paperclip_origin_count, paperclip_in_plan)
        # Should be many fewer dispatches than queries.
        if paperclip_in_plan > 5:
            self.assertLess(len(paperclip_ds), paperclip_in_plan)
        for d in paperclip_ds:
            self.assertIn("searches", d.args["command"])

    def test_biorxiv_dispatches_carry_client_filter(self):
        sp = plan(Topic(term="CRISPR screen analysis", kind=TopicKind.METHOD,
                        synonyms=["pooled screen"],
                        obscure_synonyms=["functional genomics screen"]))
        ds = build_dispatches(sp)
        biorxiv_ds = [d for d in ds if "bioRxiv" in d.tool]
        self.assertGreater(len(biorxiv_ds), 0)
        for d in biorxiv_ds:
            # Every bioRxiv dispatch must carry the client_filter (since MCP
            # can't do text search) and a valid category.
            self.assertIsNotNone(d.client_filter,
                                 "bioRxiv dispatch missing client_filter")
            self.assertTrue(len(d.client_filter) >= 3,
                            f"client_filter too narrow: {d.client_filter}")
            self.assertIn("CRISPR screen analysis", d.client_filter)
            self.assertIn("pooled screen", d.client_filter)
            self.assertIn("functional genomics screen", d.client_filter)
            self.assertIn("category", d.args)
            self.assertIn(d.args["category"], BIORXIV_CATEGORIES)

    def test_summary_counts_match(self):
        sp = plan("BRD4", kind=TopicKind.TARGET)
        ds = build_dispatches(sp)
        rep = summarise_dispatches(ds)
        self.assertEqual(rep["dispatch_count"], len(ds))
        self.assertEqual(rep["queries_covered"], sum(len(d.origin) for d in ds))


class TestClientSideFilter(unittest.TestCase):
    def _mk(self, **kw):
        return {"title": "", "abstract_preview": "", "category": "", **kw}

    def test_passes_when_title_matches(self):
        rs = [self._mk(title="A new CRISPR base editor with reduced off-targets")]
        out = filter_biorxiv_results(rs, ["base editor"])
        self.assertEqual(len(out), 1)

    def test_passes_when_abstract_matches(self):
        rs = [self._mk(abstract_preview="we used base editing in primary T cells")]
        out = filter_biorxiv_results(rs, ["base editing"])
        self.assertEqual(len(out), 1)

    def test_drops_unrelated(self):
        rs = [self._mk(title="Zebrafish heart regeneration",
                       abstract_preview="cardiac progenitor cells")]
        out = filter_biorxiv_results(rs, ["base editing", "CRISPR"])
        self.assertEqual(out, [])

    def test_case_insensitive(self):
        rs = [self._mk(title="CRISPR SCREEN OF LUNG ADENOCARCINOMA")]
        out = filter_biorxiv_results(rs, ["crispr screen"])
        self.assertEqual(len(out), 1)

    def test_empty_filter_passes_all(self):
        rs = [self._mk(title="anything")]
        out = filter_biorxiv_results(rs, [])
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
