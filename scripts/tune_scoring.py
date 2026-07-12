"""Grid-search the three scoring weights in score_long_tailness.

Usage:
    python scripts/tune_scoring.py

Reads `examples/scoring_corpus.json`, tries all weight triples
(w_recency, w_rarity, w_niche) from 0.0 to 1.0 step 0.1, and prints
the combination that maximises pairwise AUC separating "long-tail" from
"popular" labeled entries.

AUC here is the Wilcoxon-Mann-Whitney statistic: fraction of
(long-tail, popular) pairs where the long-tail entry scores strictly
higher. A value of 1.0 means perfect separation; 0.5 is random.

Tie-breaking among equal-AUC weight triples: prefer the triple whose
sum is closest to 1.0 (avoids degenerate near-zero solutions), then
closest to the current defaults (0.4, 0.3, 0.3) by Euclidean distance.
"""
from __future__ import annotations
import json
import re
import sys
from datetime import date
from pathlib import Path

_CORPUS_PATH = Path(__file__).parent.parent / "examples" / "scoring_corpus.json"

_NICHE_KEYWORDS = (
    "limitations", "non-model", "thesis", "dissertation", "addgene",
    "rrid", "geo accession", "replication failed", "off-target",
    "contradicts", "edge case", "benchmark",
)
_NICHE_PATTERNS = tuple(
    re.compile(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", re.IGNORECASE)
    for kw in _NICHE_KEYWORDS
)

_DEFAULTS = (0.4, 0.3, 0.3)


def _recency(entry: dict) -> float:
    raw = entry.get("date") or ""
    if raw:
        try:
            d = date.fromisoformat(raw[:10])
            days = (date.today() - d).days
            return max(0.0, 1.0 - days / 365.0)
        except (ValueError, TypeError):
            pass
    return 0.3


def _rarity(entry: dict) -> float:
    matched_raw = entry.get("strategies_matched", 1)
    if isinstance(matched_raw, (set, frozenset, list, tuple)):
        matched = len(matched_raw)
    elif isinstance(matched_raw, bool):
        matched = int(matched_raw)
    else:
        try:
            matched = int(matched_raw)
        except (TypeError, ValueError):
            matched = 1
    matched = max(0, matched)
    return max(0.0, min(1.0, 1.0 - matched * 0.15))


def _niche(entry: dict) -> float:
    blob = " ".join([
        str(entry.get("title", "")),
        str(entry.get("abstract_preview", "")),
    ])
    score = 0.0
    for pat in _NICHE_PATTERNS:
        if pat.search(blob):
            score += 0.08
    return min(1.0, score)


def score_entry(entry: dict, w_recency: float, w_rarity: float, w_niche: float) -> float:
    raw = w_recency * _recency(entry) + w_rarity * _rarity(entry) + w_niche * _niche(entry)
    return max(0.0, min(1.0, raw))


def pairwise_auc(lt_scores: list, pop_scores: list) -> float:
    """Fraction of (lt, pop) pairs where lt scores strictly higher."""
    if not lt_scores or not pop_scores:
        return 0.0
    wins = sum(1 for lt in lt_scores for pop in pop_scores if lt > pop)
    return wins / (len(lt_scores) * len(pop_scores))


def _tiebreak_key(triple: tuple) -> tuple:
    """Lower is better. Prefer sum near 1.0, then close to defaults."""
    w1, w2, w3 = triple
    sum_dist = abs(w1 + w2 + w3 - 1.0)
    euclidean = sum((a - b) ** 2 for a, b in zip(triple, _DEFAULTS)) ** 0.5
    return (sum_dist, euclidean)


def main() -> None:
    corpus = json.loads(_CORPUS_PATH.read_text())
    lt_entries = [e for e in corpus if e.get("label") == "long-tail"]
    pop_entries = [e for e in corpus if e.get("label") == "popular"]

    if not lt_entries or not pop_entries:
        print("ERROR: corpus must contain both 'long-tail' and 'popular' entries",
              file=sys.stderr)
        sys.exit(1)

    best_auc = -1.0
    best_triple = _DEFAULTS
    steps = [round(i * 0.1, 1) for i in range(11)]

    for w1 in steps:
        for w2 in steps:
            for w3 in steps:
                lt_scores = [score_entry(e, w1, w2, w3) for e in lt_entries]
                pop_scores = [score_entry(e, w1, w2, w3) for e in pop_entries]
                auc = pairwise_auc(lt_scores, pop_scores)
                if auc > best_auc or (
                    auc == best_auc
                    and _tiebreak_key((w1, w2, w3)) < _tiebreak_key(best_triple)
                ):
                    best_auc = auc
                    best_triple = (w1, w2, w3)

    w1, w2, w3 = best_triple
    default_lt = [score_entry(e, *_DEFAULTS) for e in lt_entries]
    default_pop = [score_entry(e, *_DEFAULTS) for e in pop_entries]
    default_auc = pairwise_auc(default_lt, default_pop)

    print(f"Corpus: {len(lt_entries)} long-tail, {len(pop_entries)} popular entries")
    print()
    print(f"Best weights: w_recency={w1:.1f}  w_rarity={w2:.1f}  w_niche={w3:.1f}")
    print(f"Pairwise AUC: {best_auc:.4f}")
    print()
    dw1, dw2, dw3 = _DEFAULTS
    print(f"Defaults:     w_recency={dw1:.1f}  w_rarity={dw2:.1f}  w_niche={dw3:.1f}")
    print(f"Default AUC:  {default_auc:.4f}")


if __name__ == "__main__":
    main()
