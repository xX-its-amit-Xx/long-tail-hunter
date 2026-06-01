"""Source adapters.

Each adapter knows how to translate a generic Query into source-specific
call shape, and (optionally) how to execute it against a public HTTP API.

Two layers:
  * `as_http(query)` -> dict describing url/method/params/headers (or None if
    the source has no public HTTP API the tool wants to hit directly).
  * `as_mcp(query)`  -> dict describing the MCP tool name + arguments. Used
    by Claude (the agent) to run queries via connected MCP servers.
"""
from __future__ import annotations
from typing import Any, Optional
from .query import Query


# ---------- HTTP shape (for standalone use) ----------

def _biorxiv_http(q: Query) -> dict[str, Any]:
    # bioRxiv official API uses path-style URLs and lacks free-text search;
    # the long_tail_hunter MCP layer should be preferred for keyword work.
    # Here we expose the *details* endpoint shape and date-window form for
    # the runner to use when the MCP path is unavailable.
    df = q.params.get("date_from")
    dt = q.params.get("date_to")
    if df and dt:
        return {
            "method": "GET",
            "url": f"https://api.biorxiv.org/details/biorxiv/{df}/{dt}",
            "params": {},
            "note": "bioRxiv has no native free-text search; filter results client-side by `q.text` substring.",
        }
    return {
        "method": "GET",
        "url": "https://api.biorxiv.org/details/biorxiv",
        "params": {},
        "note": "Date window absent; consider using mcp__claude_ai_bioRxiv__search_preprints instead.",
    }


def _github_http(q: Query) -> dict[str, Any]:
    return {
        "method": "GET",
        "url": "https://api.github.com/search/repositories",
        "params": {
            "q": q.text,
            "sort": q.params.get("sort", "updated"),
            "order": q.params.get("order", "desc"),
            "per_page": 25,
        },
        "headers": {"Accept": "application/vnd.github+json"},
    }


def _bioconductor_http(q: Query) -> dict[str, Any]:
    return {
        "method": "GET",
        "url": "https://bioconductor.org/packages/release/BiocViews.html",
        "params": {},
        "note": "Bioconductor lacks a search API; this returns the BiocViews tree for client-side grep against `q.text`.",
    }


def _biostars_http(q: Query) -> dict[str, Any]:
    return {
        "method": "GET",
        "url": "https://www.biostars.org/api/posts/",
        "params": {"q": q.text, "ord": q.params.get("sort", "creation_date_desc")},
    }


def _bioc_support_http(q: Query) -> dict[str, Any]:
    # Bioconductor support runs the same Biostar engine.
    return {
        "method": "GET",
        "url": "https://support.bioconductor.org/api/posts/",
        "params": {"q": q.text, "ord": q.params.get("sort", "creation_date_desc")},
    }


def _stackoverflow_http(q: Query) -> dict[str, Any]:
    return {
        "method": "GET",
        "url": "https://api.stackexchange.com/2.3/search/advanced",
        "params": {
            "q": q.text,
            "site": "stackoverflow",
            "tagged": "bioinformatics",
            "order": "desc",
            "sort": q.params.get("sort", "creation"),
        },
    }


HTTP_ADAPTERS = {
    "biorxiv": _biorxiv_http,
    "github": _github_http,
    "bioconductor": _bioconductor_http,
    "biostars": _biostars_http,
    "bioc_support": _bioc_support_http,
    "stackoverflow": _stackoverflow_http,
}


def as_http(query: Query) -> Optional[dict[str, Any]]:
    fn = HTTP_ADAPTERS.get(query.source)
    return fn(query) if fn else None


# ---------- MCP shape (for the agent to dispatch) ----------

def _biorxiv_mcp(q: Query) -> dict[str, Any]:
    args: dict[str, Any] = {"limit": 50}
    if "category" in q.params:
        args["category"] = q.params["category"]
    if "date_from" in q.params and "date_to" in q.params:
        args["date_from"] = q.params["date_from"]
        args["date_to"] = q.params["date_to"]
    else:
        # No window means fall back to a sensible recent window — the MCP
        # tool requires SOME time bound.
        args["recent_days"] = 60
    note = ("bioRxiv MCP has no keyword search; the runner must client-side "
            "filter results against q.params['client_filter'] (list of terms).")
    return {
        "tool": "mcp__claude_ai_bioRxiv__search_preprints",
        "args": args,
        "client_filter": q.params.get("client_filter", [q.text]),
        "note": note,
    }


def _paperclip_mcp(q: Query) -> dict[str, Any]:
    # Paperclip's vsh command vocabulary: `search "<query>" -n N` is the
    # default semantic-search form. Single-quote escaping in vsh is brittle;
    # double quotes are the documented form.
    safe = q.text.replace('"', '\\"')
    return {
        "tool": "mcp__claude_ai_Paperclip_for_literature_search__paperclip",
        "args": {"command": f'search "{safe}" -n 10'},
        "note": "For batch dispatch, group multiple paperclip queries via the `searches` command.",
    }


_GENE_RE = __import__("re").compile(r"^[A-Z]+[A-Z0-9]*$")


def _looks_like_gene(text: str) -> bool:
    """Heuristic: does `text` look like a human gene symbol?

    Rules:
      * Length 3..7
      * No whitespace
      * Must match uppercase letters followed by optional letters/digits,
        starting with at least one uppercase letter.

    Examples that match: BRD4, BRCA1, TP53, KRAS, MTOR, ALK, MYC, EZH2.
    Examples that DON'T match: lowercase, multi-word phrases, "BRCA1 VUS".
    """
    if not text or not isinstance(text, str):
        return False
    if " " in text or "\t" in text:
        return False
    n = len(text)
    if n < 3 or n > 7:
        return False
    return bool(_GENE_RE.match(text))


def _chembl_mcp(q: Query) -> dict[str, Any]:
    endpoint = q.params.get("endpoint", "target_search")
    tool_map = {
        "compound_search": "mcp__claude_ai_ChEMBL__compound_search",
        "target_search":   "mcp__claude_ai_ChEMBL__target_search",
        "get_bioactivity": "mcp__claude_ai_ChEMBL__get_bioactivity",
        "get_mechanism":   "mcp__claude_ai_ChEMBL__get_mechanism",
        "drug_search":     "mcp__claude_ai_ChEMBL__drug_search",
        "get_admet":       "mcp__claude_ai_ChEMBL__get_admet",
    }
    tool = tool_map.get(endpoint, "mcp__claude_ai_ChEMBL__target_search")
    # target_search returns 200k-character payloads when called with `query=`
    # on a broad term. Use gene_symbol= when the text looks like a gene
    # symbol (much more selective), target_name= otherwise. Always cap with
    # limit=10 so we don't spill to tool-result files.
    if endpoint == "target_search":
        if _looks_like_gene(q.text):
            args: dict[str, Any] = {"gene_symbol": q.text, "limit": 10}
        else:
            args = {"target_name": q.text, "limit": 10}
    else:
        args = {"query": q.text, "limit": 10}
    return {
        "tool": tool,
        "args": args,
    }


MCP_ADAPTERS = {
    "biorxiv": _biorxiv_mcp,
    "paperclip": _paperclip_mcp,
    "chembl": _chembl_mcp,
}


def as_mcp(query: Query) -> Optional[dict[str, Any]]:
    fn = MCP_ADAPTERS.get(query.source)
    return fn(query) if fn else None


def coverage_report(queries: list[Query]) -> dict[str, dict[str, int]]:
    """How many of these queries can we actually run, and how?"""
    http_runnable = sum(1 for q in queries if q.source in HTTP_ADAPTERS)
    mcp_runnable = sum(1 for q in queries if q.source in MCP_ADAPTERS)
    by_source = {}
    for q in queries:
        by_source.setdefault(q.source, {"count": 0, "http": False, "mcp": False})
        by_source[q.source]["count"] += 1
        by_source[q.source]["http"] = q.source in HTTP_ADAPTERS
        by_source[q.source]["mcp"] = q.source in MCP_ADAPTERS
    return {
        "total": len(queries),
        "http_runnable": http_runnable,
        "mcp_runnable": mcp_runnable,
        "by_source": by_source,
    }
