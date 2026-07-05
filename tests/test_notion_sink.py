"""Tests for the Notion sink module."""
from __future__ import annotations
import unittest

from long_tail_hunter.topic import Topic, TopicKind
from long_tail_hunter.notion_sink import log_result, _NOTION_TOOL


def _topic(term: str = "CRISPR base editor") -> Topic:
    return Topic(term=term, kind=TopicKind.METHOD)


class TestLogResult(unittest.TestCase):
    """Acceptance tests for notion_sink.log_result (IMPROVEMENTS.md item 4)."""

    # --- acceptance criteria ---

    def test_tool_name_is_notion_update_page(self):
        out = log_result("page-abc", {"title": "Test paper", "doi": "10.x/y"}, _topic(), 0.8)
        self.assertEqual(out["tool"], _NOTION_TOOL)

    def test_result_contains_title(self):
        out = log_result("page-abc", {"title": "Niche paper on base editing", "doi": "10.x/y"}, _topic(), 0.75)
        self.assertIn("Niche paper on base editing", str(out["args"]))

    def test_result_contains_doi(self):
        doi = "10.1234/test.paper"
        out = log_result("page-abc", {"title": "x", "doi": doi}, _topic(), 0.5)
        self.assertIn(doi, str(out["args"]))

    # --- structural tests ---

    def test_returns_dict_with_tool_and_args(self):
        out = log_result("p", {"title": "x"}, _topic(), 0.0)
        self.assertIn("tool", out)
        self.assertIn("args", out)
        self.assertIsInstance(out["args"], dict)

    def test_page_id_in_args(self):
        out = log_result("my-page-id-123", {"title": "x"}, _topic(), 0.5)
        self.assertEqual(out["args"]["pageId"], "my-page-id-123")

    def test_append_block_is_bulleted_list_item(self):
        out = log_result("p", {"title": "x"}, _topic(), 0.5)
        block = out["args"]["append"][0]
        self.assertEqual(block["object"], "block")
        self.assertEqual(block["type"], "bulleted_list_item")
        self.assertIn("rich_text", block["bulleted_list_item"])

    def test_title_linked_to_url_when_present(self):
        url = "https://doi.org/10.1234/x"
        out = log_result("p", {"title": "Linked paper", "doi": "10.1234/x", "url": url}, _topic(), 0.8)
        rich_text = out["args"]["append"][0]["bulleted_list_item"]["rich_text"]
        title_part = rich_text[0]
        self.assertEqual(title_part["text"]["content"], "Linked paper")
        self.assertEqual(title_part["text"]["link"]["url"], url)

    def test_no_url_title_has_no_link(self):
        out = log_result("p", {"title": "Plain paper"}, _topic(), 0.5)
        rich_text = out["args"]["append"][0]["bulleted_list_item"]["rich_text"]
        title_text = rich_text[0]["text"]
        self.assertNotIn("link", title_text)

    def test_rationale_contains_score(self):
        out = log_result("p", {"title": "x"}, _topic(), 0.73)
        self.assertIn("0.73", str(out["args"]))

    def test_rationale_contains_topic_term(self):
        out = log_result("p", {"title": "x"}, _topic(term="frataxin deficiency"), 0.6)
        self.assertIn("frataxin deficiency", str(out["args"]))

    def test_missing_doi_produces_two_rich_text_parts(self):
        out = log_result("p", {"title": "No DOI paper"}, _topic(), 0.5)
        rich_text = out["args"]["append"][0]["bulleted_list_item"]["rich_text"]
        # Only title + rationale; no DOI bracket block.
        self.assertEqual(len(rich_text), 2)

    def test_doi_present_produces_three_rich_text_parts(self):
        out = log_result("p", {"title": "Paper with DOI", "doi": "10.1234/x"}, _topic(), 0.6)
        rich_text = out["args"]["append"][0]["bulleted_list_item"]["rich_text"]
        self.assertEqual(len(rich_text), 3)

    def test_name_field_used_as_title_fallback(self):
        out = log_result("p", {"name": "my-repo"}, _topic(), 0.4)
        rich_text = out["args"]["append"][0]["bulleted_list_item"]["rich_text"]
        self.assertEqual(rich_text[0]["text"]["content"], "my-repo")

    def test_no_title_no_name_uses_untitled(self):
        out = log_result("p", {}, _topic(), 0.3)
        rich_text = out["args"]["append"][0]["bulleted_list_item"]["rich_text"]
        self.assertEqual(rich_text[0]["text"]["content"], "(untitled)")

    def test_id_field_used_as_doi_fallback(self):
        out = log_result("p", {"title": "ChEMBL target", "id": "CHEMBL12345"}, _topic(), 0.5)
        self.assertIn("CHEMBL12345", str(out["args"]))
