# long-tail-hunter

A small, reusable tool that **generates the search strategy, not the answer**.
Given a phenotype / method / target, it emits a structured query plan
designed to route around the popularity prior — past famous papers,
top-cited tools, and top-starred repos — and out into the long tail:
Bioconductor packages, recent bioRxiv preprints, niche mailing lists,
GitHub repos sorted by recent activity instead of stars.

## Quick start

```bash
python -m long_tail_hunter "neoantigen prediction" \
  --kind method \
  --syn neoepitope \
  --syn "MHC binding prediction" \
  --obscure "personalized cancer vaccine epitope" \
  --method "MHC binding prediction"
```

Output is a JSON `SearchPlan` listing every generated query, the strategy
that produced it, and a rationale.

```bash
python -m unittest discover -s tests       # 48 tests, ~6ms
python -m long_tail_hunter --list-strategies
```

## Why

Naive search inherits the field's popularity prior: the famous paper is
indexed everywhere, cited by everything, and shows up first. That's fine
when you want the obvious answer. It's a problem when you want what other
people miss — recent preprints not yet citation-weighted, niche packages,
applications in adjacent organisms, negative-result papers, theses, the
practical knowledge that lives in mailing-list threads.

This tool is the deliberate counterweight, in code.

## Architecture

- `long_tail_hunter/topic.py` — `Topic` dataclass (term, synonyms,
  obscure_synonyms, organism, adjacent_methods, famous_hits).
- `long_tail_hunter/query.py` — `Query` / `SearchPlan` dataclasses.
- `long_tail_hunter/strategies.py` — 10 strategies, each a pure
  `(Topic) -> list[Query]`:
  - `recent_preprints` — date-windowed bioRxiv by category
  - `methodology_focus` — method + qualifier (pipeline, benchmark, limitations…)
  - `negative_space` — failed to replicate, contradicts, artifact…
  - `software_first` — GitHub sorted by activity, not stars; Bioconductor
  - `niche_forums` — BioStars, Bioconductor support, SO bioinformatics tag
  - `obscure_synonyms` — non-canonical vocabulary
  - `cross_domain_transfer` — same method in plant/yeast/drosophila/...
  - `reagent_and_accession` — Addgene/GEO/RRID mentions
  - `thesis_and_preprint_floor` — dissertation-flavored full-text
  - `chemistry_side` — ChEMBL for target/phenotype topics
- `long_tail_hunter/categories.py` — keyword → bioRxiv category map.
- `long_tail_hunter/sources.py` — HTTP and MCP adapters per source.
- `long_tail_hunter/runner.py` — batches paperclip queries via `searches`,
  filters bioRxiv results client-side.

## Anti-popularity invariants (enforced by tests)

- No GitHub query sorts by stars.
- Every plan includes at least one `negative_space` query.
- Every plan includes at least one date-windowed query.
- Every plan hits at least 3 distinct sources.

## Live-validated

The Paperclip negative-space query `"spatial transcriptomics deconvolution
limitations"` returned NLSDeconv, Mahamune 2025, STged, and a masked-adversarial
NN — none of which are the famous cell2location / RCTD / SPOTlight hits.

## Open improvements

See `IMPROVEMENTS.md`.
