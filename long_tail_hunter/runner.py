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
  * `aggregate_results(raw_by_dispatch)` — normalises + deduplicates results
    from multiple dispatches into a unified list of `Result` objects.
"""
from __future__ import annotations
from typing import Any, Iterable
from dataclasses import dataclass, field
from datetime import date
import re

from .query import Query, SearchPlan
from .sources import as_mcp


_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi.org/",
    "doi:",
)
# Tight DOI shape: 10. + 4-9 digits + slash. Rejects '10.0' / '10.x release'.
_DOI_SHAPE_RE = re.compile(r"^10\.\d{4,9}/")


def _norm_doi(s: str) -> str:
    """Strip common DOI URL prefixes and lowercase, leaving a bare DOI."""
    s = s.strip().lower()
    for p in _DOI_PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
            break
    return s


def _looks_like_doi(s: str) -> bool:
    return bool(_DOI_SHAPE_RE.match(_norm_doi(s)))


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


@dataclass
class Result:
    """Normalised, deduplicated result from any source.

    `id` is the canonical identifier: bare DOI for biomed sources (biorxiv,
    paperclip), `full_name` for GitHub repos, ChEMBL ID otherwise.
    `strategies_matched` accumulates across all dispatches that returned this
    same result, so a large set signals a popular, widely-found item.
    """
    source: str
    id: str
    title: str
    url: str
    date: str
    abstract_preview: str
    raw: dict[str, Any]
    strategies_matched: set[str] = field(default_factory=set)


_BIOMED_SOURCES = frozenset({"biorxiv", "paperclip"})


def _extract_id(raw: dict[str, Any], source: str) -> str:
    """Return a canonical dedup key for a raw result dict."""
    if source in _BIOMED_SOURCES:
        doi = str(raw.get("doi") or "").strip()
        if doi:
            norm = _norm_doi(doi)
            if _looks_like_doi(norm):
                return norm
        title = str(raw.get("title") or "").strip().lower()
        return f"title:{title[:120]}" if title else ""
    if source == "github":
        for f in ("full_name", "name", "repo"):
            v = raw.get(f)
            if v:
                return str(v)
        return ""
    # ChEMBL and other sources
    for f in ("molecule_chembl_id", "target_chembl_id", "chembl_id", "id"):
        v = raw.get(f)
        if v:
            return str(v)
    title = str(raw.get("title") or raw.get("pref_name") or "").strip().lower()
    return f"title:{title[:120]}" if title else ""


def _extract_url(raw: dict[str, Any], source: str, id_str: str) -> str:
    """Construct or extract a URL for a result."""
    if source in _BIOMED_SOURCES:
        if id_str and not id_str.startswith("title:"):
            return f"https://doi.org/{id_str}"
        return str(raw.get("url") or "")
    if source == "github":
        url = raw.get("html_url") or raw.get("url") or ""
        if not url and id_str and not id_str.startswith("title:"):
            url = f"https://github.com/{id_str}"
        return str(url)
    return str(raw.get("url") or "")


def aggregate_results(
    raw_by_dispatch: "Iterable[tuple[Dispatch, list[dict[str, Any]]]]",
) -> list[Result]:
    """Normalise and deduplicate results across dispatches.

    Takes an iterable of ``(Dispatch, raw_results_list)`` pairs — the shape
    the agent produces after running MCP calls — and returns a flat list of
    ``Result`` objects keyed by canonical id.  When the same id appears from
    multiple dispatches the ``strategies_matched`` sets are unioned, letting
    ``score_long_tailness`` apply the correct diversity penalty: items found
    by many strategies are the popular ones.

    Pass ``my_dict.items()`` if you built a ``{Dispatch: [...]}`` mapping.
    """
    by_id: dict[str, Result] = {}

    for dispatch, raw_list in raw_by_dispatch:
        source = dispatch.origin[0].source if dispatch.origin else "unknown"
        dispatch_strategies = {q.strategy for q in dispatch.origin}

        for raw in raw_list:
            id_str = _extract_id(raw, source)
            if not id_str:
                continue

            if id_str in by_id:
                by_id[id_str].strategies_matched.update(dispatch_strategies)
            else:
                by_id[id_str] = Result(
                    source=source,
                    id=id_str,
                    title=str(
                        raw.get("title") or raw.get("pref_name")
                        or raw.get("name") or ""
                    ),
                    url=_extract_url(raw, source, id_str),
                    date=str(raw.get("date") or raw.get("date_revised") or ""),
                    abstract_preview=str(
                        raw.get("abstract_preview") or raw.get("abstract") or ""
                    ),
                    raw=raw,
                    strategies_matched=set(dispatch_strategies),
                )

    return list(by_id.values())


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
    # Pre-split: real DOIs vs. title/repo substrings. The DOI shape test
    # is tight (10. + 4-9 digits + slash) and prefix-normalised so URL-form
    # DOIs ('https://doi.org/10.x') match bare-DOI fields and vice versa.
    dois_norm: set[str] = set()
    substrs_lc: list[str] = []
    for h in hits:
        if _looks_like_doi(h):
            dois_norm.add(_norm_doi(h))
        else:
            substrs_lc.append(h.lower())

    # Pick which title-ish field to match by source.
    if source == "github":
        title_fields = ("name", "full_name", "repo", "title")
    else:
        title_fields = ("title", "name")

    out = []
    for r in results:
        result_doi = _norm_doi(str(r.get("doi") or ""))
        if result_doi and result_doi in dois_norm:
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
# Word-boundary patterns so "thesis" doesn't match "synthesis" /
# "hypothesis" / "photosynthesis", "benchmark" doesn't match
# "benchmarking", etc.
_NICHE_PATTERNS = tuple(
    re.compile(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", re.IGNORECASE)
    for kw in _NICHE_KEYWORDS
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

    # --- Source diversity penalty (renamed: anti-popularity rarity score) ---
    # Higher 'matched' = found by many strategies = more popular = lower
    # score. Floor matched at 0 to keep the score sane against negative
    # caller input.
    matched_raw = meta.get("strategies_matched", 1)
    if isinstance(matched_raw, (set, frozenset, list, tuple)):
        matched = len(matched_raw)
    elif isinstance(matched_raw, bool):
        # bool is a subclass of int — treat as 1 (matched) / 0 (not).
        matched = int(matched_raw)
    else:
        try:
            matched = int(matched_raw)
        except (TypeError, ValueError):
            matched = 1
    matched = max(0, matched)
    rarity = max(0.0, min(1.0, 1.0 - matched * 0.15))
    diversity = rarity  # legacy local name, kept for the weighted-mean line

    # --- Niche-keyword density ---
    blob = " ".join([
        str(result.get("title", "")),
        str(result.get("abstract_preview", "")),
    ])
    niche_score = 0.0
    for pat in _NICHE_PATTERNS:
        if pat.search(blob):
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
