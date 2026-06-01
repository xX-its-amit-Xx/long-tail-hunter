"""Unit tests for the strategy generators.

Run with:  python -m unittest discover -s tests
"""
from __future__ import annotations
import json
import unittest

from long_tail_hunter.topic import Topic, TopicKind
from long_tail_hunter import strategies as S
from long_tail_hunter.hunter import plan
from long_tail_hunter.query import Query, SearchPlan
from long_tail_hunter.sources import as_http, as_mcp, coverage_report


def _t(**kw) -> Topic:
    return Topic(term=kw.pop("term", "CRISPR base editor off-target"), **kw)


class TestRecentPreprints(unittest.TestCase):
    def test_one_query_per_category_with_client_filter(self):
        topic = _t(synonyms=["base editing"], obscure_synonyms=["cytidine deaminase fusion"])
        qs = S.recent_preprints(topic)
        # bioRxiv has no text search, so we emit one query per matching
        # category and stash the synonyms in client_filter.
        self.assertGreaterEqual(len(qs), 1)
        for q in qs:
            self.assertEqual(q.source, "biorxiv")
            self.assertEqual(q.strategy, "recent_preprints")
            self.assertIn("recent", q.tags)
            self.assertIn("date_from", q.params)
            self.assertIn("date_to", q.params)
            self.assertIn("category", q.params)
            self.assertEqual(q.params["sort"], "date_desc")
            self.assertIn("client_filter", q.params)
            self.assertIn(topic.term, q.params["client_filter"])
            self.assertIn("base editing", q.params["client_filter"])
            self.assertIn("cytidine deaminase fusion", q.params["client_filter"])

    def test_date_window_is_six_months(self):
        from datetime import date, timedelta
        qs = S.recent_preprints(_t())
        params = qs[0].params
        end = date.fromisoformat(params["date_to"])
        start = date.fromisoformat(params["date_from"])
        days = (end - start).days
        # 6 months ≈ 180 days +/- 5
        self.assertGreater(days, 175)
        self.assertLess(days, 200)


class TestMethodologyFocus(unittest.TestCase):
    def test_emits_qualifier_per_anchor_paperclip_only(self):
        topic = _t(adjacent_methods=["scRNA-seq", "scATAC-seq"])
        qs = S.methodology_focus(topic)
        # 2 anchors * 5 qualifiers * 1 source (paperclip) = 10
        self.assertEqual(len(qs), 10)
        self.assertEqual({q.source for q in qs}, {"paperclip"})
        self.assertTrue(all(q.strategy == "methodology_focus" for q in qs))

    def test_falls_back_to_term_when_no_adjacent_methods(self):
        topic = _t(adjacent_methods=[])
        qs = S.methodology_focus(topic)
        # 1 anchor * 5 qualifiers * 1 source = 5
        self.assertEqual(len(qs), 5)


class TestNegativeSpace(unittest.TestCase):
    def test_all_qualifiers_appear(self):
        qs = S.negative_space(_t())
        texts = [q.text for q in qs]
        self.assertTrue(any("failed to replicate" in t for t in texts))
        self.assertTrue(any("artifact" in t for t in texts))
        self.assertTrue(any("contradicts" in t for t in texts))


class TestSoftwareFirst(unittest.TestCase):
    def test_github_is_sorted_by_updated_not_stars(self):
        qs = S.software_first(_t())
        github_qs = [q for q in qs if q.source == "github"]
        self.assertGreater(len(github_qs), 0)
        for q in github_qs:
            # The whole point: NOT stars.
            self.assertEqual(q.params.get("sort"), "updated")
            self.assertNotIn("stars", str(q.params).lower())

    def test_includes_bioconductor(self):
        qs = S.software_first(_t())
        self.assertTrue(any(q.source == "bioconductor" for q in qs))

    def test_language_filters_present(self):
        qs = S.software_first(_t())
        github_texts = [q.text for q in qs if q.source == "github"]
        self.assertTrue(any("language:R" in t for t in github_texts))
        self.assertTrue(any("language:Python" in t for t in github_texts))


class TestNicheForums(unittest.TestCase):
    def test_three_forum_sources(self):
        qs = S.niche_forums(_t())
        self.assertEqual({q.source for q in qs}, {"biostars", "bioc_support", "stackoverflow"})

    def test_so_query_uses_bioinformatics_tag(self):
        qs = S.niche_forums(_t())
        so = [q for q in qs if q.source == "stackoverflow"][0]
        self.assertIn("[bioinformatics]", so.text)
        self.assertEqual(so.params.get("sort"), "newest")


class TestObscureSynonyms(unittest.TestCase):
    def test_silent_when_none_supplied(self):
        qs = S.obscure_synonyms(_t(obscure_synonyms=[]))
        self.assertEqual(qs, [])

    def test_one_per_synonym_via_paperclip(self):
        qs = S.obscure_synonyms(_t(obscure_synonyms=["A", "B"]))
        self.assertEqual(len(qs), 2)
        self.assertEqual({q.source for q in qs}, {"paperclip"})


class TestCrossDomain(unittest.TestCase):
    def test_drops_user_organism(self):
        qs = S.cross_domain_transfer(_t(organism="zebrafish"))
        texts = " ".join(q.text for q in qs)
        self.assertNotIn("zebrafish", texts.lower())

    def test_keeps_default_set_when_no_organism(self):
        qs = S.cross_domain_transfer(_t())
        bridges = {"plant", "yeast", "drosophila", "zebrafish", "c. elegans"}
        text_blob = " ".join(q.text for q in qs).lower()
        for b in bridges:
            self.assertIn(b, text_blob)

    def test_paperclip_only(self):
        qs = S.cross_domain_transfer(_t())
        self.assertEqual({q.source for q in qs}, {"paperclip"})


class TestReagentAccession(unittest.TestCase):
    def test_includes_addgene_geo_rrid(self):
        qs = S.reagent_and_accession(_t())
        kinds = {tag for q in qs for tag in q.tags}
        self.assertIn("addgene", kinds)
        self.assertIn("geo", kinds)
        self.assertIn("rrid", kinds)


class TestChemistrySide(unittest.TestCase):
    def test_emits_target_and_mechanism_for_target(self):
        topic = _t(kind=TopicKind.TARGET, term="BRD4")
        qs = S.chemistry_side(topic)
        endpoints = {q.params.get("endpoint") for q in qs}
        self.assertEqual(endpoints, {"target_search", "get_mechanism"})

    def test_skipped_for_method(self):
        topic = _t(kind=TopicKind.METHOD, term="ribosome profiling")
        qs = S.chemistry_side(topic)
        self.assertEqual(qs, [])


class TestApplyAll(unittest.TestCase):
    def test_only_and_exclude_filter(self):
        topic = _t()
        qs_only = S.apply_all(topic, only=["recent_preprints"])
        self.assertTrue(all(q.strategy == "recent_preprints" for q in qs_only))

        qs_excl = S.apply_all(topic, exclude=["recent_preprints"])
        self.assertFalse(any(q.strategy == "recent_preprints" for q in qs_excl))


class TestPlan(unittest.TestCase):
    def test_string_input_works(self):
        sp = plan("neoantigen prediction")
        self.assertIsInstance(sp, SearchPlan)
        self.assertGreater(len(sp.queries), 10)

    def test_serialises_clean_json(self):
        sp = plan("base editing")
        as_text = sp.to_json()
        reparsed = json.loads(as_text)
        self.assertEqual(reparsed["topic_term"], "base editing")
        self.assertIn("query_count", reparsed)
        self.assertIn("sources", reparsed)

    def test_dedupe_actually_dedupes(self):
        # Construct two strategies emitting identical paperclip queries —
        # they should collapse. recent_preprints now uses category-keyed
        # queries so it's an unsuitable demo; pick obscure_synonyms which
        # is paperclip-only and uses topic.term-style text.
        topic = Topic(
            term="BRD4",
            kind=TopicKind.TARGET,
            obscure_synonyms=["BRD4", "BRD4"],  # duplicate intentionally
        )
        sp = plan(topic, dedupe=True)
        # Two identical (source=paperclip, text='BRD4') should collapse.
        from collections import Counter
        text_counts = Counter(
            (q.source, q.text, q.params.get("category", ""),
             q.params.get("endpoint", ""))
            for q in sp.queries
        )
        for key, count in text_counts.items():
            self.assertEqual(count, 1, f"Duplicate survived dedupe: {key} x {count}")

    def test_dedupe_keeps_different_categories(self):
        # The dedupe key includes category — a single term that hits multiple
        # bioRxiv categories should yield multiple recent_preprints queries,
        # not be collapsed to one.
        topic = Topic(term="CRISPR base editor", kind=TopicKind.METHOD)
        sp = plan(topic, dedupe=True)
        recent = [q for q in sp.queries if q.strategy == "recent_preprints"]
        cats = {q.params["category"] for q in recent}
        self.assertGreater(len(cats), 1,
                           "Multi-category topic collapsed to one bioRxiv query — dedupe is too aggressive")

    def test_notes_warn_about_missing_obscure_synonyms(self):
        sp = plan("CRISPR")
        self.assertTrue(any("obscure synonym" in n for n in sp.notes))

    def test_notes_warn_about_missing_famous_hits(self):
        sp = plan("CRISPR")
        self.assertTrue(any("famous_hits" in n for n in sp.notes))


class TestQueryShape(unittest.TestCase):
    def test_every_query_has_rationale(self):
        sp = plan("CRISPR screen analysis")
        for q in sp.queries:
            self.assertTrue(q.rationale, f"Empty rationale in {q}")

    def test_every_query_has_strategy_name_that_exists(self):
        sp = plan("CRISPR screen analysis")
        names = set(S.ALL_STRATEGIES.keys())
        for q in sp.queries:
            self.assertIn(q.strategy, names)


class TestSources(unittest.TestCase):
    def test_http_adapter_returns_dict_or_none(self):
        sp = plan("CRISPR")
        for q in sp.queries:
            result = as_http(q)
            if result is not None:
                self.assertIn("method", result)
                self.assertIn("url", result)

    def test_mcp_adapter_returns_tool_name(self):
        sp = plan("BRD4", kind=TopicKind.TARGET)
        mcp_results = [as_mcp(q) for q in sp.queries]
        # At least the biorxiv, paperclip, and chembl ones should resolve.
        resolved = [r for r in mcp_results if r is not None]
        self.assertGreater(len(resolved), 5)
        for r in resolved:
            self.assertTrue(r["tool"].startswith("mcp__"))

    def test_coverage_report(self):
        sp = plan("CRISPR")
        rep = coverage_report(sp.queries)
        self.assertIn("by_source", rep)
        self.assertGreater(rep["total"], 0)


class TestAntiPopularityInvariants(unittest.TestCase):
    """The defining invariants of the tool — if these break, the tool stopped
    being a counterweight to the popularity prior."""

    def test_no_github_query_sorts_by_stars(self):
        sp = plan("scanpy")
        github_qs = sp.by_source("github")
        self.assertGreater(len(github_qs), 0)
        for q in github_qs:
            sort = q.params.get("sort", "")
            self.assertNotEqual(sort, "stars",
                                f"GitHub query sorted by stars — that's the popularity prior we're fighting: {q}")

    def test_at_least_one_negative_space_query(self):
        sp = plan("anything")
        self.assertGreater(len(sp.by_strategy("negative_space")), 0)

    def test_at_least_one_recent_window_query(self):
        sp = plan("anything")
        recent = [q for q in sp.queries
                  if "recent" in q.tags or q.params.get("sort") == "date_desc"]
        self.assertGreater(len(recent), 0)

    def test_at_least_three_distinct_sources(self):
        sp = plan("anything")
        self.assertGreaterEqual(len({q.source for q in sp.queries}), 3,
                                "Plan must hit multiple sources or it's not routing around any source's prior.")


if __name__ == "__main__":
    unittest.main()
