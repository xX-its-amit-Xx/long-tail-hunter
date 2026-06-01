"""Query and SearchPlan dataclasses — the structured output of a hunt."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import json


@dataclass
class Query:
    """A single query for a single source.

    `source` is a short identifier (e.g. 'biorxiv', 'github', 'bioconductor').
    `text` is the literal query string. `params` carries source-specific knobs
    (date range, sort order, language filter, ...). `rationale` explains *why*
    this query was generated — the bridge from strategy to result.
    `strategy` names the strategy that produced it. `tags` are free-form
    labels useful for downstream dedup / ranking ('recent', 'methodology',
    'negative-space').
    """
    source: str
    text: str
    rationale: str
    strategy: str
    params: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    expected_yield: str = "long-tail"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchPlan:
    topic_term: str
    topic_kind: str
    queries: list[Query] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_term": self.topic_term,
            "topic_kind": self.topic_kind,
            "queries": [q.to_dict() for q in self.queries],
            "notes": self.notes,
            "query_count": len(self.queries),
            "sources": sorted({q.source for q in self.queries}),
            "strategies": sorted({q.strategy for q in self.queries}),
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def by_source(self, source: str) -> list[Query]:
        return [q for q in self.queries if q.source == source]

    def by_strategy(self, strategy: str) -> list[Query]:
        return [q for q in self.queries if q.strategy == strategy]
