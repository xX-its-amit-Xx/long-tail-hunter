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
from datetime import date

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
    famous_hits: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Keep results whose title/abstract_preview contains any filter term.

    Case-insensitive substring match — naive on purpose. bioRxiv's API
    returns full title + ~250-char abstract_preview, which is enough surface
    for keyword hits without false positives from a long abstract body.

    If `famous_hits` is provided, also drop results that match one of those
    known-popular references (by DOI exact match or title substring).
    """
    terms_lc = [t.lower() for t in client_filter if t]
    if terms_lc:
        kept = []
        for r in results:
            blob = " ".join([
                r.get("title", ""),
                r.get("abstract_preview", ""),
                r.get("category", ""),
            ]).lower()
            if any(t in blob for t in terms_lc):
                kept.append(r)
    else:
        kept = list(results)
    if famous_hits:
        kept = filter_results(kept, "biorxiv", famous_hits)
    return kept


def filter_paperclip_results(
    results: list[dict[str, Any]],
    famous_hits: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Drop paperclip results that match a famous_hits entry.

    Paperclip results carry full text, but match surface is the same:
    DOI exact match or title substring. No client_filter keyword step —
    paperclip already does semantic search server-side.
    """
    if not famous_hits:
        return list(results)
    return filter_results(results, "paperclip", famous_hits)


def filter_results(
    results: list[dict[str, Any]],
    source: str,
    famous_hits: Iterable[str],
) -> list[dict[str, Any]]:
    """Drop results that match a famous_hits entry.

    Matching rules:
      * DOI: if the hit string looks like a DOI (contains '10.' prefix) and
        the result has a 'doi' field, require exact (case-insensitive) match.
      * Title / repo URL substring: case-insensitive substring match against
        the result's 'title', 'name', or 'repo' field (whichever exists).

    `source` is accepted for future per-source tweaks but currently only
    used to pick the right title-ish field (paperclip vs biorxiv vs github).
    """
    hits = [h for h in famous_hits if h]
    if not hits:
        return list(results)
    # Pre-split: DOIs vs. title/repo substrings.
    dois_lc = {h.lower() for h in hits if h.lower().startswith("10.") or "/10." in h.lower()}
    substrs_lc = [h.lower() for h in hits if h.lower() not in dois_lc]

    # Pick which title-ish field to match by source.
    if source == "github":
        title_fields = ("name", "full_name", "repo", "title")
    else:
        title_fields = ("title", "name")

    out = []
    for r in results:
        doi_lc = (r.get("doi") or "").lower()
        if doi_lc and doi_lc in dois_lc:
            continue
        title_blob_lc = " ".join(
            str(r.get(f, "")) for f in title_fields
        ).lower()
        if any(s and s in title_blob_lc for s in substrs_lc):
            continue
        out.append(r)
    return out


_NICHE_KEYWORDS = (
    "limitations", "non-model", "thesis", "dissertation", "addgene",
    "rrid", "geo accession", "replication failed", "off-target",
    "contradicts", "edge case", "benchmark",
)


def score_long_tailness(
    result: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> float:
    """Score a single result for how likely it is to be a long-tail find.

    Returns a value in [0, 1]; higher = more long-tail. Combines three
    signals as a weighted mean:

      * Recency (weight 0.4): score = max(0, 1 - days_since_today/365)
        when a YYYY-MM-DD `date` field is present. Default 0.3 if missing.
      * Source diversity penalty (weight 0.3): if `meta` carries
        `strategies_matched`, penalise results found by many strategies —
        those are the popular ones. score = max(0, 1 - matched * 0.15).
      * Niche-keyword density (weight 0.3): count occurrences of niche
        markers (limitations, thesis, Addgene, RRID, ...) in
        title + abstract_preview; each match adds 0.08, capped at 1.0.
    """
    meta = meta or {}

    # --- Recency ---
    raw_date = result.get("date") or ""
    if raw_date:
        try:
            d = date.fromisoformat(raw_date[:10])
            days = (date.today() - d).days
            recency = max(0.0, 1.0 - days / 365.0)
        except (ValueError, TypeError):
            recency = 0.3
    else:
        recency = 0.3

    # --- Source diversity penalty ---
    matched = meta.get("strategies_matched", 1)
    try:
        matched = int(matched)
    except (TypeError, ValueError):
        matched = 1
    diversity = max(0.0, 1.0 - matched * 0.15)

    # --- Niche-keyword density ---
    blob = " ".join([
        str(result.get("title", "")),
        str(result.get("abstract_preview", "")),
    ]).lower()
    niche_score = 0.0
    for kw in _NICHE_KEYWORDS:
        if kw in blob:
            niche_score += 0.08
    niche_score = min(1.0, niche_score)

    final = 0.4 * recency + 0.3 * diversity + 0.3 * niche_score
    # Guard against floating-point drift outside [0, 1].
    if final < 0.0:
        return 0.0
    if final > 1.0:
        return 1.0
    return final


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
