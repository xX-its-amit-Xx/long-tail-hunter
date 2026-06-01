"""Structured input — what we're hunting for."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TopicKind(str, Enum):
    PHENOTYPE = "phenotype"
    METHOD = "method"
    TARGET = "target"
    MIXED = "mixed"


@dataclass
class Topic:
    """A topic to hunt.

    `term` is the canonical user-supplied phrase. Everything else is optional
    enrichment — when absent, strategies fall back to heuristics on `term`.
    """
    term: str
    kind: TopicKind = TopicKind.MIXED
    synonyms: list[str] = field(default_factory=list)
    obscure_synonyms: list[str] = field(default_factory=list)
    organism: Optional[str] = None
    adjacent_methods: list[str] = field(default_factory=list)
    famous_hits: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)

    def all_synonyms(self) -> list[str]:
        return [self.term, *self.synonyms, *self.obscure_synonyms]

    @classmethod
    def from_string(cls, s: str, kind: TopicKind = TopicKind.MIXED) -> "Topic":
        return cls(term=s.strip(), kind=kind)
