"""Query-generation strategies.

Each strategy is a function (Topic) -> list[Query]. The strategies are the
*deliberate counterweight to the popularity prior*: they route around famous
papers, top-cited tools, and the first-page Google result by construction.

Design rules:
  - A strategy produces queries that, *by their shape*, will not return the
    same hits a naive search would. (e.g. sort by date-not-citations, search
    for a method's failure mode, query a niche forum rather than PubMed.)
  - Every query carries a `rationale` explaining the trick.
  - Strategies are pure: same Topic in -> same queries out. Side effects
    (HTTP) live in source adapters / runners.
  - Strategies are cheap to add. Keep them small and orthogonal.
"""
from __future__ import annotations
from datetime import date, timedelta
from typing import Callable
from .topic import Topic, TopicKind
from .query import Query
from .categories import categories_for


Strategy = Callable[[Topic], list[Query]]


# ---------- helpers ----------

def _today() -> date:
    return date.today()


def _date_window(months_back: int) -> tuple[str, str]:
    end = _today()
    start = end - timedelta(days=months_back * 30)
    return start.isoformat(), end.isoformat()


def _synonyms_or_term(topic: Topic) -> list[str]:
    """Use obscure synonyms first when available — that's where the long tail lives."""
    if topic.obscure_synonyms:
        return topic.obscure_synonyms
    if topic.synonyms:
        return topic.synonyms
    return [topic.term]


# ---------- strategies ----------

def recent_preprints(topic: Topic) -> list[Query]:
    """bioRxiv preprints from the last 6 months — citations have not yet accumulated.

    The popularity prior takes years to form via citations. Recent preprints
    are the time before that prior kicks in.

    NB: bioRxiv has no native free-text search — we pull each likely category's
    recent window and rely on a downstream `client_filter` (term + synonyms)
    to keep on-topic results. `text` is the canonical filter phrase.
    """
    start, end = _date_window(6)
    cats = categories_for(topic)
    out = []
    for cat in cats:
        out.append(Query(
            source="biorxiv",
            text=topic.term,
            rationale=f"Recent preprints in '{cat}' — citation-based ranking has not yet biased the field.",
            strategy="recent_preprints",
            params={
                "date_from": start, "date_to": end, "sort": "date_desc",
                "category": cat,
                "client_filter": [topic.term, *topic.synonyms, *topic.obscure_synonyms],
            },
            tags=["recent", "preprint", "pre-citation", f"cat:{cat}"],
        ))
    return out


def methodology_focus(topic: Topic) -> list[Query]:
    """Search for the *method*, not the famous *result*.

    Method-centric queries surface papers that *use* a technique rather than
    introduce it — the famous paper introduced it once; thousands of papers
    apply it under the radar.

    Routed through Paperclip (full-text semantic search) because bioRxiv's
    API can't do free-text matching.
    """
    if topic.kind == TopicKind.METHOD or topic.adjacent_methods:
        anchors = topic.adjacent_methods or [topic.term]
    else:
        anchors = [topic.term]
    qualifiers = ["pipeline", "benchmark", "limitations", "reproducibility", "edge cases"]
    out = []
    for anchor in anchors:
        for q in qualifiers:
            out.append(Query(
                source="paperclip",
                text=f"{anchor} {q}",
                rationale=f"Full-text method+qualifier ({q}) — surfaces application/critique papers, not the seminal one.",
                strategy="methodology_focus",
                tags=["methodology", "fulltext"],
            ))
    return out


def negative_space(topic: Topic) -> list[Query]:
    """Search for what *failed* — limitations, contradictions, retractions.

    Popularity prior loves positive results. The long tail of negative,
    null, and contradictory findings is systematically under-cited but
    epistemically valuable.
    """
    qualifiers = [
        "failed to replicate",
        "limitations",
        "negative result",
        "contradicts",
        "off-target effects",
        "false positive",
        "artifact",
    ]
    out = []
    for q in qualifiers:
        out.append(Query(
            source="paperclip",
            text=f"{topic.term} {q}",
            rationale=f"Full-text negative-space probe ({q}) — surfaces under-cited counter-evidence.",
            strategy="negative_space",
            tags=["negative", "fulltext"],
        ))
    return out


def software_first(topic: Topic) -> list[Query]:
    """Search for packages, tools, repos, function names.

    Software is where methods actually run — the README of a niche tool
    contains the kind of practical knowledge that papers omit.
    """
    out = []
    for term in _synonyms_or_term(topic):
        out.append(Query(
            source="github",
            text=term,
            rationale="GitHub sorted by recent activity — surfaces tools that are alive, not famous.",
            strategy="software_first",
            params={"sort": "updated", "order": "desc", "lang_filter": None},
            tags=["software", "activity"],
        ))
        out.append(Query(
            source="github",
            text=f"{term} language:R",
            rationale="R-language filter — Bioconductor/CRAN tooling lives here, often unsearched on PubMed.",
            strategy="software_first",
            params={"sort": "updated", "order": "desc"},
            tags=["software", "R"],
        ))
        out.append(Query(
            source="github",
            text=f"{term} language:Python",
            rationale="Python-language filter for the scientific Python ecosystem.",
            strategy="software_first",
            params={"sort": "updated", "order": "desc"},
            tags=["software", "python"],
        ))
        out.append(Query(
            source="bioconductor",
            text=term,
            rationale="Bioconductor package search — curated, but indexed by Google far less than PubMed.",
            strategy="software_first",
            tags=["software", "bioconductor"],
        ))
    return out


def niche_forums(topic: Topic) -> list[Query]:
    """Mailing-list-style Q&A: BioStars, Bioc-support, SO bioinformatics.

    Niche forums capture the conversational knowledge that never makes it
    into a paper — the actual debugging, parameter choices, and gotchas.
    """
    out = []
    for term in _synonyms_or_term(topic):
        out.append(Query(
            source="biostars",
            text=term,
            rationale="BioStars Q&A — practical workflow knowledge absent from formal literature.",
            strategy="niche_forums",
            params={"sort": "creation_date_desc"},
            tags=["forum", "practical"],
        ))
        out.append(Query(
            source="bioc_support",
            text=term,
            rationale="Bioconductor support forum — error messages and edge cases.",
            strategy="niche_forums",
            tags=["forum", "bioconductor"],
        ))
        out.append(Query(
            source="stackoverflow",
            text=f"{term} [bioinformatics]",
            rationale="Tag-restricted SO query for low-vote, niche answers.",
            strategy="niche_forums",
            params={"sort": "newest"},
            tags=["forum", "stackoverflow"],
        ))
    return out


def obscure_synonyms(topic: Topic) -> list[Query]:
    """Run obscure synonyms through Paperclip.

    The famous term is heavily indexed; older or alternative terminology
    matches papers from adjacent fields or earlier decades that the
    canonical query misses.
    """
    if not topic.obscure_synonyms:
        return []
    out = []
    for syn in topic.obscure_synonyms:
        out.append(Query(
            source="paperclip",
            text=syn,
            rationale=f"Full-text search on obscure synonym '{syn}' — non-canonical vocabulary reaches papers the famous term misses.",
            strategy="obscure_synonyms",
            tags=["synonym", "fulltext"],
        ))
    return out


def cross_domain_transfer(topic: Topic) -> list[Query]:
    """Same method/target, different organism or disease.

    A technique developed in mouse-cancer is often republished, mostly
    unnoticed, in plant biology or microbial work. Cross-domain papers
    cite each other rarely but contain transferable insight.
    """
    out = []
    bridges = [
        "plant", "yeast", "drosophila", "zebrafish", "C. elegans",
        "non-model organism", "veterinary", "agricultural",
    ]
    if topic.organism:
        bridges = [b for b in bridges if topic.organism.lower() not in b.lower()]
    for bridge in bridges:
        out.append(Query(
            source="paperclip",
            text=f"{topic.term} {bridge}",
            rationale=f"Cross-domain probe into '{bridge}' — same method/concept in an adjacent field.",
            strategy="cross_domain_transfer",
            tags=["cross-domain", bridge.lower().replace(" ", "-")],
        ))
    return out


def reagent_and_accession(topic: Topic) -> list[Query]:
    """Search by reagent or accession patterns.

    Specific catalog numbers (Addgene plasmids, RRID, GEO GSE accessions)
    appear in methods sections of long-tail papers but not in titles or
    abstracts of famous ones.
    """
    out = []
    out.append(Query(
        source="paperclip",
        text=f"{topic.term} Addgene",
        rationale="Reagent-anchored search: papers that mention a specific Addgene deposit.",
        strategy="reagent_and_accession",
        tags=["reagent", "addgene"],
    ))
    out.append(Query(
        source="paperclip",
        text=f"{topic.term} GEO accession",
        rationale="GEO dataset mention — surfaces papers that reanalyzed a dataset, often under-cited.",
        strategy="reagent_and_accession",
        tags=["reagent", "geo"],
    ))
    out.append(Query(
        source="paperclip",
        text=f"{topic.term} RRID",
        rationale="RRID mention — papers that follow reproducibility identifier practice, often well-documented but low-profile.",
        strategy="reagent_and_accession",
        tags=["reagent", "rrid"],
    ))
    return out


def thesis_and_preprint_floor(topic: Topic) -> list[Query]:
    """Theses, conference posters, technical reports.

    These are the bedrock of long-tail science: full method detail,
    rarely cited by the field at large because they don't have a journal
    venue.
    """
    out = []
    out.append(Query(
        source="paperclip",
        text=f"{topic.term} thesis",
        rationale="Catches preprints that re-publish thesis chapters — full method detail.",
        strategy="thesis_and_preprint_floor",
        tags=["thesis", "depth"],
    ))
    out.append(Query(
        source="paperclip",
        text=f"{topic.term} dissertation",
        rationale="Dissertation-flavored full-text query for deep methodological writeups.",
        strategy="thesis_and_preprint_floor",
        tags=["thesis", "fulltext"],
    ))
    return out


# Target/phenotype-specific: bring in chemistry-side data via ChEMBL.
def chemistry_side(topic: Topic) -> list[Query]:
    """For targets/phenotypes: pull the chemical-biology angle.

    Bioinformatics-trained searchers often miss compound/mechanism data
    that ChEMBL indexes. This anchors the hunt in chemistry-of-the-target.
    """
    if topic.kind not in (TopicKind.TARGET, TopicKind.PHENOTYPE, TopicKind.MIXED):
        return []
    out = []
    out.append(Query(
        source="chembl",
        text=topic.term,
        rationale="ChEMBL target search — the chemical-biology long tail of any biological target.",
        strategy="chemistry_side",
        params={"endpoint": "target_search"},
        tags=["chemistry", "target"],
    ))
    out.append(Query(
        source="chembl",
        text=topic.term,
        rationale="ChEMBL mechanism — drugs known to act via this target/phenotype.",
        strategy="chemistry_side",
        params={"endpoint": "get_mechanism"},
        tags=["chemistry", "mechanism"],
    ))
    return out


# ---------- registry ----------

ALL_STRATEGIES: dict[str, Strategy] = {
    "recent_preprints": recent_preprints,
    "methodology_focus": methodology_focus,
    "negative_space": negative_space,
    "software_first": software_first,
    "niche_forums": niche_forums,
    "obscure_synonyms": obscure_synonyms,
    "cross_domain_transfer": cross_domain_transfer,
    "reagent_and_accession": reagent_and_accession,
    "thesis_and_preprint_floor": thesis_and_preprint_floor,
    "chemistry_side": chemistry_side,
}


def apply_all(topic: Topic, only: list[str] | None = None,
              exclude: list[str] | None = None) -> list[Query]:
    out: list[Query] = []
    for name, fn in ALL_STRATEGIES.items():
        if only is not None and name not in only:
            continue
        if exclude is not None and name in exclude:
            continue
        out.extend(fn(topic))
    return out
