"""Top-level orchestrator: take a topic, return a SearchPlan."""
from __future__ import annotations
from typing import Optional
from .topic import Topic, TopicKind
from .query import Query, SearchPlan
from . import strategies


def plan(
    topic: Topic | str,
    kind: TopicKind = TopicKind.MIXED,
    only: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
    dedupe: bool = True,
) -> SearchPlan:
    """Generate a search plan for `topic`.

    `topic` may be a string (treated as Topic.from_string) or a Topic.
    `only`/`exclude` filter the strategy set by name.
    `dedupe` collapses (source, text) duplicates.
    """
    if isinstance(topic, str):
        topic = Topic.from_string(topic, kind=kind)

    queries = strategies.apply_all(topic, only=only, exclude=exclude)
    if dedupe:
        queries = _dedupe(queries)

    notes = _notes_for(topic, queries)
    return SearchPlan(
        topic_term=topic.term,
        topic_kind=topic.kind.value,
        queries=queries,
        notes=notes,
    )


def _dedupe(queries: list[Query]) -> list[Query]:
    """Collapse exact-duplicate queries. The key includes every dispatchable
    parameter that changes the resulting MCP call (category, endpoint, ...)
    so we don't accidentally drop legitimately distinct queries."""
    seen: set[tuple[str, ...]] = set()
    out = []
    for q in queries:
        key = (
            q.source,
            q.text,
            q.params.get("endpoint", ""),
            q.params.get("category", ""),
            q.params.get("date_from", ""),
            q.params.get("date_to", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def _notes_for(topic: Topic, queries: list[Query]) -> list[str]:
    notes = []
    if not topic.obscure_synonyms:
        notes.append(
            "No obscure synonyms supplied — the strategy is weaker. Provide "
            "Topic.obscure_synonyms (older terms, alternative spellings, "
            "discipline-adjacent jargon) for better long-tail recall."
        )
    if not topic.famous_hits:
        notes.append(
            "No famous_hits supplied — cannot post-filter against the popular "
            "result set. Provide Topic.famous_hits (DOIs / repo names / paper "
            "titles to suppress) to make the counterweight sharper."
        )
    by_source = {}
    for q in queries:
        by_source[q.source] = by_source.get(q.source, 0) + 1
    notes.append(
        "Queries by source: " + ", ".join(f"{k}={v}" for k, v in sorted(by_source.items()))
    )
    return notes
