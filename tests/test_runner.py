"""Tests for the dispatcher, category mapping, and client-side filter."""
from __future__ import annotations
import unittest

from long_tail_hunter.topic import Topic, TopicKind
from long_tail_hunter.hunter import plan
from long_tail_hunter.runner import (
    build_dispatches, filter_biorxiv_results, filter_paperclip_results,
    filter_results, score_long_tailness, summarise_dispatches,
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

    def test_keyword_map_grew_by_at_least_30(self):
        from long_tail_hunter.categories import _KEYWORD_TO_CAT
        # Previous baseline ~50 entries; after expansion expect 80+
        self.assertGreaterEqual(len(_KEYWORD_TO_CAT), 80,
                                f"Keyword map only has {len(_KEYWORD_TO_CAT)} entries")

    def test_organoid_maps_to_developmental_biology(self):
        topic = Topic(term="brain organoid model", kind=TopicKind.MIXED)
        self.assertIn("developmental biology", categories_for(topic))

    def test_tcr_maps_to_immunology(self):
        topic = Topic(term="TCR repertoire analysis", kind=TopicKind.MIXED)
        self.assertIn("immunology", categories_for(topic))

    def test_alphafold_maps_to_bioinformatics(self):
        topic = Topic(term="AlphaFold structure prediction", kind=TopicKind.METHOD)
        cats = categories_for(topic)
        self.assertIn("bioinformatics", cats)
        self.assertIn("biophysics", cats)

    def test_alzheimer_maps_to_neuroscience(self):
        topic = Topic(term="Alzheimer disease biomarkers", kind=TopicKind.MIXED)
        self.assertIn("neuroscience", categories_for(topic))

    def test_cite_seq_maps_to_multi_categories(self):
        topic = Topic(term="CITE-seq immune profiling", kind=TopicKind.METHOD)
        cats = categories_for(topic)
        self.assertIn("genomics", cats)
        self.assertIn("immunology", cats)

    def test_merfish_maps_to_spatial_omics(self):
        topic = Topic(term="MERFISH analysis", kind=TopicKind.METHOD)
        cats = categories_for(topic)
        self.assertIn("genomics", cats)
        self.assertIn("bioinformatics", cats)

    def test_multiple_sclerosis_immunology_plus_neuro(self):
        topic = Topic(term="multiple sclerosis lesion staging", kind=TopicKind.MIXED)
        cats = categories_for(topic)
        self.assertIn("immunology", cats)
        self.assertIn("neuroscience", cats)

    def test_all_new_keyword_values_are_valid_cats(self):
        # Sanity: every category mentioned in the keyword map must be valid.
        from long_tail_hunter.categories import _KEYWORD_TO_CAT
        for kw, cats in _KEYWORD_TO_CAT.items():
            for c in cats:
                self.assertIn(c, BIORXIV_CATEGORIES,
                              f"keyword {kw!r} maps to invalid category {c!r}")


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


class TestLooksLikeGene(unittest.TestCase):
    def test_classic_gene_symbols(self):
        from long_tail_hunter.sources import _looks_like_gene
        for sym in ["BRD4", "BRCA1", "TP53", "KRAS", "MTOR", "ALK", "MYC", "EZH2"]:
            self.assertTrue(_looks_like_gene(sym), f"{sym!r} should look like a gene")

    def test_non_genes(self):
        from long_tail_hunter.sources import _looks_like_gene
        for txt in [
            "",
            "brd4",                       # lowercase
            "bromodomain",                # too long, lowercase
            "BRD4 inhibitor",             # whitespace
            "BRCA1 VUS",                  # whitespace
            "AB",                         # too short
            "TOOLONGGENE",                # too long
            "123",                        # digits only / no leading letter
            "BRD4-X",                     # disallowed char
            "  ",                         # whitespace
            None,                         # not a str
        ]:
            self.assertFalse(_looks_like_gene(txt), f"{txt!r} should NOT look like a gene")


class TestChemblDispatch(unittest.TestCase):
    def _q(self, text, endpoint="target_search"):
        from long_tail_hunter.query import Query
        return Query(
            source="chembl", text=text, rationale="test",
            strategy="chemistry_side", params={"endpoint": endpoint},
        )

    def test_target_search_uses_gene_symbol_for_gene_like_text(self):
        from long_tail_hunter.sources import as_mcp
        mcp = as_mcp(self._q("BRD4"))
        self.assertEqual(mcp["tool"], "mcp__claude_ai_ChEMBL__target_search")
        self.assertEqual(mcp["args"].get("gene_symbol"), "BRD4")
        self.assertNotIn("target_name", mcp["args"])
        self.assertNotIn("query", mcp["args"])
        self.assertEqual(mcp["args"].get("limit"), 10)

    def test_target_search_uses_target_name_for_non_gene_text(self):
        from long_tail_hunter.sources import as_mcp
        mcp = as_mcp(self._q("BRD4 inhibitor cocrystal"))
        self.assertEqual(mcp["args"].get("target_name"), "BRD4 inhibitor cocrystal")
        self.assertNotIn("gene_symbol", mcp["args"])
        self.assertEqual(mcp["args"].get("limit"), 10)

    def test_other_endpoints_use_query_key(self):
        from long_tail_hunter.sources import as_mcp
        for ep in ("compound_search", "get_mechanism", "drug_search"):
            mcp = as_mcp(self._q("BRD4", endpoint=ep))
            self.assertIn("query", mcp["args"])
            self.assertEqual(mcp["args"]["limit"], 10)


class TestScoreLongTailness(unittest.TestCase):
    def _today_iso(self, days_ago=0):
        from datetime import date, timedelta
        return (date.today() - timedelta(days=days_ago)).isoformat()

    def test_returns_value_in_unit_interval(self):
        for r in [
            {},
            {"title": "x"},
            {"date": self._today_iso(0), "title": "Addgene RRID thesis dissertation limitations non-model"},
        ]:
            s = score_long_tailness(r)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_newer_outscores_older(self):
        fresh = {"title": "x", "abstract_preview": "",
                 "date": self._today_iso(10)}
        stale = {"title": "x", "abstract_preview": "",
                 "date": self._today_iso(300)}
        self.assertGreater(score_long_tailness(fresh), score_long_tailness(stale))

    def test_single_strategy_outscores_multi_strategy(self):
        r = {"title": "x", "abstract_preview": "",
             "date": self._today_iso(30)}
        single = score_long_tailness(r, meta={"strategies_matched": 1})
        many   = score_long_tailness(r, meta={"strategies_matched": 6})
        self.assertGreater(single, many,
                           "Multi-strategy hits should score lower (they're popular)")

    def test_niche_keywords_increase_score(self):
        bland = {"title": "A general method paper", "abstract_preview": "",
                 "date": self._today_iso(30)}
        niche = {"title": "A general method paper with limitations and Addgene plasmids",
                 "abstract_preview": "uses non-model organism; RRID provided",
                 "date": self._today_iso(30)}
        self.assertGreater(score_long_tailness(niche), score_long_tailness(bland))

    def test_missing_date_uses_default_recency(self):
        r = {"title": "x"}
        s = score_long_tailness(r, meta={"strategies_matched": 1})
        # default recency 0.3 -> 0.4*0.3 = 0.12; diversity = 0.85*0.3 = 0.255
        # niche = 0; total ~ 0.375
        self.assertGreater(s, 0.0)
        self.assertLess(s, 0.5)

    def test_garbage_date_falls_back_gracefully(self):
        r = {"date": "not-a-date", "title": "x"}
        # Should not raise.
        s = score_long_tailness(r)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_ordering_combines_all_signals(self):
        # A: fresh + single-strategy + niche-rich = high
        # B: older + many-strategy + bland = low
        a = {"date": self._today_iso(5),
             "title": "Replication failed in a non-model organism",
             "abstract_preview": "limitations; Addgene; RRID"}
        b = {"date": self._today_iso(300),
             "title": "A landmark popular paper",
             "abstract_preview": ""}
        sa = score_long_tailness(a, meta={"strategies_matched": 1})
        sb = score_long_tailness(b, meta={"strategies_matched": 6})
        self.assertGreater(sa, sb)


class TestFamousHitsFilter(unittest.TestCase):
    def _mk(self, **kw):
        return {"title": "", "abstract_preview": "", "category": "",
                "doi": "", **kw}

    def test_doi_match_drops_result(self):
        rs = [
            self._mk(title="A new CRISPR base editor", doi="10.1126/science.aad5227"),
            self._mk(title="An unrelated paper", doi="10.1038/something-else"),
        ]
        out = filter_results(rs, "biorxiv",
                             famous_hits=["10.1126/science.aad5227"])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["doi"], "10.1038/something-else")

    def test_doi_match_is_case_insensitive(self):
        rs = [self._mk(doi="10.1126/SCIENCE.AAD5227", title="X")]
        out = filter_results(rs, "biorxiv",
                             famous_hits=["10.1126/science.aad5227"])
        self.assertEqual(out, [])

    def test_title_substring_drops_result(self):
        rs = [
            self._mk(title="cell2location: A landmark deconvolution method"),
            self._mk(title="A different niche tool"),
        ]
        out = filter_results(rs, "biorxiv", famous_hits=["cell2location"])
        self.assertEqual(len(out), 1)
        self.assertIn("niche tool", out[0]["title"])

    def test_no_match_survives(self):
        rs = [self._mk(title="long-tail paper", doi="10.9999/foo")]
        out = filter_results(rs, "biorxiv",
                             famous_hits=["10.1126/science.aad5227",
                                          "cell2location"])
        self.assertEqual(len(out), 1)

    def test_empty_famous_hits_returns_all(self):
        rs = [self._mk(title="anything")]
        self.assertEqual(filter_results(rs, "biorxiv", []), rs)
        self.assertEqual(filter_results(rs, "biorxiv", [""]), rs)

    def test_filter_biorxiv_results_applies_both_filters(self):
        rs = [
            {"title": "A new base editor", "abstract_preview": "",
             "category": "", "doi": "10.1126/science.aad5227"},
            {"title": "Another base editor refinement", "abstract_preview": "",
             "category": "", "doi": "10.9999/long-tail"},
            {"title": "Unrelated zebrafish heart paper", "abstract_preview": "",
             "category": "", "doi": "10.0000/x"},
        ]
        out = filter_biorxiv_results(
            rs,
            client_filter=["base editor"],
            famous_hits=["10.1126/science.aad5227"],
        )
        # Keyword filter keeps the two base-editor papers; famous_hits drops
        # the one with the matching DOI; the unrelated paper was already
        # excluded by the keyword filter.
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["doi"], "10.9999/long-tail")

    def test_filter_paperclip_results_drops_famous_doi(self):
        rs = [
            {"title": "Famous tool paper", "doi": "10.1038/foo"},
            {"title": "Niche refinement", "doi": "10.9999/bar"},
        ]
        out = filter_paperclip_results(rs, famous_hits=["10.1038/foo"])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["doi"], "10.9999/bar")

    def test_filter_paperclip_results_no_hits_passes_all(self):
        rs = [{"title": "x"}, {"title": "y"}]
        self.assertEqual(filter_paperclip_results(rs, None), rs)
        self.assertEqual(filter_paperclip_results(rs, []), rs)


if __name__ == "__main__":
    unittest.main()
