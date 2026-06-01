# Open improvements

A scheduled remote agent picks one of these each week. Each item names a
single coherent change with clear acceptance criteria.

## 1. Cross-source result aggregator + dedupe

Each source returns results in its own shape (bioRxiv has `doi`/`title`/
`abstract_preview`; paperclip returns excerpts; GitHub returns repo metadata;
ChEMBL returns target rows). The runner currently has no notion of a
unified result object, so downstream ranking and the new
`score_long_tailness` heuristic only work per-source.

**Acceptance:**
- New `runner.Result` dataclass with normalized fields: `source`, `id` (DOI
  if biomed, repo full_name for GitHub, ChEMBL ID otherwise), `title`,
  `url`, `date`, `abstract_preview`, `raw` (the original dict),
  `strategies_matched: set[str]`.
- New `runner.aggregate_results(raw_by_dispatch)` taking
  `{Dispatch: list[dict]}` and returning `list[Result]` deduplicated by
  normalized id (DOI preferred). Multi-source hits accumulate
  `strategies_matched`.
- Unit tests showing: (a) duplicate DOI across sources collapses to one
  Result, (b) `strategies_matched` contains both strategy names, (c)
  `score_long_tailness` consumed the diversity penalty correctly.

## 2. OpenAlex source adapter

OpenAlex (`api.openalex.org`) has a rich free-text search API (unlike
bioRxiv) and indexes most biomed venues. It's a perfect long-tail source
because the API exposes citation count — we deliberately sort by date,
not citations.

**Acceptance:**
- `sources._openalex_http(q)` and HTTP-only — no MCP adapter yet.
- Query shape: `https://api.openalex.org/works?search=<term>&sort=publication_date:desc&per-page=25&filter=type:article`.
- New strategy `openalex_recent(topic)` that emits OpenAlex queries
  parallel to `recent_preprints` (one per obscure_synonym; falls back to
  term).
- Strategy registered in `ALL_STRATEGIES`.
- Tests: `as_http` returns a dict with the right URL, the strategy
  produces >= 1 query, and the integration test still passes.

## 3. GitHub recent-commits scraping (no GitHub MCP)

The GitHub HTTP adapter today searches the `repositories` endpoint sorted
by `updated`, but that returns repos sorted by last-commit-time which can
be a README touch-up. The long-tail signal we actually want is *recent
substantive commits* in repos that match a niche term.

**Acceptance:**
- `sources._github_commits_http(q)` hitting
  `https://api.github.com/search/commits` with `q.text` and
  `Accept: application/vnd.github.cloak-preview+json`.
- New strategy `software_first_commits(topic)` that emits these queries
  alongside the existing repo-search queries.
- Tests covering: URL shape, sort=author-date, the strategy fires for
  every synonym.

## 4. Notion integration for logging found long-tail papers

The user (researcher) wants to keep a Notion page of long-tail papers
found per topic, with rationale. The Notion MCP is already connected.

**Acceptance:**
- New module `long_tail_hunter/notion_sink.py` with a single function
  `log_result(notion_page_id: str, result: dict, topic: Topic, score: float)`
  returning a dict describing the MCP call to make (we don't execute it
  inside the Python package — the agent does).
- The MCP call shape uses
  `mcp__claude_ai_Notion__notion-update-page` with an `append` payload
  containing a bullet: title (linked to URL), one-line rationale, score.
- Unit test asserting the produced dict has the right tool name and
  contains the result's title and DOI.

## 5. Scoring heuristic tuning with a labeled corpus

The current `score_long_tailness` heuristic is plausible but untuned.
Build a small labeled corpus (10-20 known long-tail papers, 10-20 known
popular papers from the same fields) and tune the three weights to
maximize ranking AUC on this corpus.

**Acceptance:**
- `examples/scoring_corpus.json` with 20+ entries: each has the result
  fields the scorer reads plus a `label` of `"long-tail"` or `"popular"`.
- New CLI subcommand or script `scripts/tune_scoring.py` that grid-
  searches the three weights (0.0..1.0 step 0.1) and prints the best
  (weights, AUC).
- Unit test that loads the corpus, runs `score_long_tailness` on every
  row, and asserts the median long-tail score > median popular score.

## Out of scope for the weekly routine

- Direct LLM-rerank of results (needs design discussion, model picking).
- Full integration test against live MCP servers (flaky; need fixtures).

## Shipped

### 2026-05-31

- `famous_hits` suppression filter — `runner.filter_results`,
  `filter_paperclip_results`, integrated into `filter_biorxiv_results`,
  with DOI exact and title-substring matching.
- ChEMBL gene-symbol routing — `sources._looks_like_gene` plus
  `_chembl_mcp` dispatch to `gene_symbol=` / `target_name=` with
  `limit=10` capped on every endpoint.
- Expanded `categories.py` keyword map — added 60+ entries covering
  immunology subterms (TCR/MHC/Treg/dendritic cell), developmental
  biology (organoid/gastruloid/somite/neural crest), microscopy
  (cryo-EM/lattice-light-sheet/expansion microscopy), multi-omics
  (CITE-seq/Perturb-seq/MERFISH/CyTOF), disease classes (Alzheimer/
  Parkinson/ALS/IBD/MS), and structural biology (AlphaFold/Rosetta).
- `score_long_tailness(result, meta)` heuristic — recency (0.4) +
  source-diversity penalty (0.3) + niche-keyword density (0.3),
  bounded to [0, 1].
- Four new phenotype test cases — treatment-resistant depression,
  insulin resistance / NAFLD, neural tube closure defect, Friedreich
  ataxia. Integration test now asserts >= 11 cases and >= 5
  phenotype cases.
