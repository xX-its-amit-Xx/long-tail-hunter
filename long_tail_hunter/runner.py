"""Build *dispatchable* MCP calls from a SearchPlan.

The Python package can't directly call MCP tools — those live in the agent
(Claude). What it *can* do is emit a precise list of MCP invocations the
agent should run, batched where possible, with a per-batch description of
how to interpret the response.

Two helpers also live here:
  * `batch_paperclip(plan)` — collapses many paperclip queries into a single
    `searches "q1" "q2" ..."` call (faster, runs in parallel server-side).
  * `filter_biorxiv_results(results, client_filter)` — applies the
    client-side keyword filter to a bioRxiv MCP response, since the MCP
    layer can't filter by text itself.
"""
from __future__ import annotations
from typing import Any, Iterable
from dataclasses import dataclass

from .query import Query, SearchPlan
from .sources import as_mcp


@dataclass
class Dispatch:
    """A single ready-to-run MCP invocation, with provenance back to its
    originating Query (or list of Queries, for batched calls)."""
    tool: str
    args: dict[str, Any]
    origin: list[Query]
    client_filter: list[str] | None = None
    note: str = ""

    def describe(self) -> str:
        sources = ", ".join(q.source for q in self.origin)
        strategies = ", ".join(sorted({q.strategy for q in self.origin}))
        return f"{self.tool} | sources={sources} | strategies={strategies}"


def build_dispatches(plan: SearchPlan) -> list[Dispatch]:
    """Translate a SearchPlan into a list of MCP invocations.

    Paperclip queries are batched into a single `searches "q1" "q2" ...` call.
    Other queries become one Dispatch each.
    """
    out: list[Dispatch] = []

    paperclip_qs = [q for q in plan.queries if q.source == "paperclip"]
    if paperclip_qs:
        terms = [q.text.replace('"', '\\"') for q in paperclip_qs]
        # `searches` accepts space-separated quoted queries; 10 per call is a
        # safe chunk to keep response sizes manageable.
        for chunk_start in range(0, len(terms), 10):
            chunk_terms = terms[chunk_start:chunk_start + 10]
            chunk_qs = paperclip_qs[chunk_start:chunk_start + 10]
            quoted = " ".join(f'"{t}"' for t in chunk_terms)
            out.append(Dispatch(
                tool="mcp__claude_ai_Paperclip_for_literature_search__paperclip",
                args={"command": f"searches {quoted}"},
                origin=list(chunk_qs),
                note="Parallel paperclip search; one result block per query, saved under /session_files/searches/.",
            ))

    for q in plan.queries:
        if q.source == "paperclip":
            continue
        mcp = as_mcp(q)
        if mcp is None:
            continue
        out.append(Dispatch(
            tool=mcp["tool"],
            args=mcp["args"],
            origin=[q],
            client_filter=mcp.get("client_filter"),
            note=mcp.get("note", ""),
        ))
    return out


def filter_biorxiv_results(
    results: list[dict[str, Any]],
    client_filter: Iterable[str],
) -> list[dict[str, Any]]:
    """Keep results whose title/abstract_preview contains any filter term.

    Case-insensitive substring match — naive on purpose. bioRxiv's API
    returns full title + ~250-char abstract_preview, which is enough surface
    for keyword hits without false positives from a long abstract body.
    """
    terms_lc = [t.lower() for t in client_filter if t]
    if not terms_lc:
        return list(results)
    out = []
    for r in results:
        blob = " ".join([
            r.get("title", ""),
            r.get("abstract_preview", ""),
            r.get("category", ""),
        ]).lower()
        if any(t in blob for t in terms_lc):
            out.append(r)
    return out


def summarise_dispatches(dispatches: list[Dispatch]) -> dict[str, Any]:
    by_tool: dict[str, int] = {}
    total_origin = 0
    for d in dispatches:
        by_tool[d.tool] = by_tool.get(d.tool, 0) + 1
        total_origin += len(d.origin)
    return {
        "dispatch_count": len(dispatches),
        "queries_covered": total_origin,
        "by_tool": by_tool,
    }
