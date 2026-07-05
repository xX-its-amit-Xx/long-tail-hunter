"""Notion sink — build the MCP call dict to log a long-tail result to a Notion page.

The Python package cannot execute MCP tool calls directly; that is the agent's
job. This module constructs and returns the exact dict the agent should pass to
the Notion MCP tool, keeping the call-shape logic testable in plain Python.
"""
from __future__ import annotations
from typing import Any

from .topic import Topic

_NOTION_TOOL = "mcp__Notion__notion-update-page"


def log_result(
    notion_page_id: str,
    result: dict[str, Any],
    topic: Topic,
    score: float,
) -> dict[str, Any]:
    """Return the MCP call dict for appending a long-tail paper to a Notion page.

    The returned dict has two keys: ``"tool"`` (the Notion MCP tool name) and
    ``"args"`` (the arguments to pass to it). The caller — typically the
    agent — executes the actual MCP call; nothing is sent from here.

    The appended block is a bulleted list item containing:
      - The paper/repo title, hyperlinked to its URL when available
      - A one-line rationale (topic term + long-tail score)
      - The DOI or identifier in brackets when available
    """
    title = str(result.get("title") or result.get("name") or "(untitled)")
    url = str(result.get("url") or "")
    doi = str(result.get("doi") or result.get("id") or "")
    rationale = f"Long-tail find for '{topic.term}'; score={score:.2f}"

    rich_text: list[dict[str, Any]] = []

    # Title segment — linked when a URL is available.
    title_text: dict[str, Any] = {"content": title}
    if url:
        title_text["link"] = {"url": url}
    rich_text.append({"type": "text", "text": title_text})

    # Rationale segment.
    rich_text.append({"type": "text", "text": {"content": f" — {rationale}"}})

    # DOI/identifier segment — omitted when not present.
    if doi:
        rich_text.append({"type": "text", "text": {"content": f" [{doi}]"}})

    return {
        "tool": _NOTION_TOOL,
        "args": {
            "pageId": notion_page_id,
            "append": [
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": rich_text},
                }
            ],
        },
    }
