# Open improvements

A scheduled remote agent picks one of these each week. Each item names a
single coherent change with clear acceptance criteria.

## 1. `famous_hits` suppression filter

`Topic.famous_hits` is accepted but never applied. Implement a filter that
drops results whose DOI, repo URL, or title matches one of `famous_hits`,
and apply it inside `runner.filter_biorxiv_results` *and* a new
`filter_paperclip_results`. Update tests in `tests/test_runner.py` to cover.

**Acceptance:**
- New `filter_results(results, source, famous_hits)` helper in
  `runner.py`.
- Tests demonstrating a DOI in `famous_hits` causes the matching result to
  be dropped; non-matches survive.
- All existing 48 tests still pass.

## 2. Expand `categories.py` keyword map

Today's map has ~50 keywords. Add at least 30 more, particularly for:
- Immunology subterms (TCR, antibody, MHC, complement, ...)
- Developmental biology (organoid, gastruloid, somite, ...)
- Microscopy / imaging (cryoEM, lattice-light-sheet, expansion microscopy)
- Multi-omics (CITE-seq, Perturb-seq, mass cytometry, ...)
- Disease classes (Alzheimer, Parkinson, ALS, IBD, ...)

**Acceptance:**
- `_KEYWORD_TO_CAT` grows by ≥ 30 entries.
- New unit tests show specific new keywords map to expected categories.
- `test_falls_back_to_bioinformatics` still works for genuinely unknown
  jargon.

## 3. ChEMBL gene-symbol routing

`sources._chembl_mcp` currently passes `{"query": q.text}` for both endpoints.
`target_search` works better with `gene_symbol=` when the term matches a
gene pattern (uppercase letters + optional digits, 3–7 chars, e.g. `BRD4`,
`BRCA1`, `TP53`).

**Acceptance:**
- New helper `_looks_like_gene(text: str) -> bool` in `sources.py`.
- `_chembl_mcp` uses `gene_symbol=` for target_search when the helper
  returns True; otherwise `target_name=`.
- `limit` capped at 10 by default to avoid the 200k-character payload
  problem (see issue notes in code).
- Unit tests for both the gene-vs-non-gene routing and the limit cap.

## 4. `score_long_tailness(result)` heuristic

A small scorer the runner can use to rank results. Combine:
- Recency (newer → higher; exponential decay, half-life ~6 months)
- Source diversity (results found by multiple strategies → lower score —
  they're the popular ones)
- Niche-keyword density (presence of words like "limitations",
  "non-model organism", "thesis", "Addgene" → higher score)

**Acceptance:**
- `runner.score_long_tailness(result, meta) -> float` returning 0..1.
- Higher score = more long-tail.
- Unit tests with hand-crafted result dicts and expected ordering.

## 5. More test cases (especially phenotype-kind)

`examples/test_cases.json` has only one phenotype case (cytokine storm).
Add at least 4 more, spanning:
- A psychiatric phenotype (e.g. anhedonia, treatment-resistant depression)
- A metabolic phenotype (e.g. insulin resistance, NAFLD)
- A developmental phenotype (e.g. neural tube closure defect)
- A rare-disease phenotype

For each: term, kind, synonyms, obscure_synonyms, rationale. Update
`tests/test_integration.py` to keep its ">= 5 cases" assertion honest.

## Out of scope for the weekly routine

- GitHub MCP adapter (no GitHub MCP available yet).
- Result aggregator + dedup across sources (needs design discussion).
- Notion / Google Drive integration for tracking found papers (needs design).
