"""Map a Topic to one or more bioRxiv subject categories.

bioRxiv's API has *no* free-text search — only category + date window. So to
search bioRxiv at all, we need a mapping from a topic's words to its likely
categories. Best-effort, keyword-based, deliberately conservative.

The 27 official bioRxiv categories are listed in `BIORXIV_CATEGORIES`.
"""
from __future__ import annotations
from .topic import Topic

BIORXIV_CATEGORIES = [
    "animal behavior and cognition", "biochemistry", "bioengineering",
    "bioinformatics", "biophysics", "cancer biology", "cell biology",
    "clinical trials", "developmental biology", "ecology", "epidemiology",
    "evolutionary biology", "genetics", "genomics", "immunology",
    "microbiology", "molecular biology", "neuroscience", "paleontology",
    "pathology", "pharmacology and toxicology", "physiology", "plant biology",
    "scientific communication and education", "synthetic biology",
    "systems biology", "zoology",
]


# Keyword -> category mapping. Lowercase, substring match against the
# topic text + all synonyms. Order doesn't matter; all matching cats fire.
_KEYWORD_TO_CAT: dict[str, list[str]] = {
    # genomics / bioinformatics cluster
    "crispr":             ["genetics", "genomics", "bioinformatics", "molecular biology"],
    "base editor":        ["genetics", "molecular biology"],
    "base editing":       ["genetics", "molecular biology"],
    "rna-seq":            ["genomics", "bioinformatics"],
    "scrna":              ["genomics", "bioinformatics", "cell biology"],
    "scrna-seq":          ["genomics", "bioinformatics", "cell biology"],
    "scatac":             ["genomics", "bioinformatics", "cell biology"],
    "atac-seq":           ["genomics", "bioinformatics"],
    "spatial":            ["genomics", "bioinformatics", "cell biology"],
    "transcriptom":       ["genomics", "bioinformatics"],
    "proteom":            ["biochemistry", "bioinformatics"],
    "metabolom":          ["biochemistry", "systems biology"],
    "deconvolution":      ["bioinformatics", "genomics"],
    "integration":        ["bioinformatics", "systems biology"],
    "ribosome profil":    ["molecular biology", "genomics"],
    "ribo-seq":           ["molecular biology", "genomics"],

    # cancer / immunology
    "cancer":             ["cancer biology"],
    "tumor":              ["cancer biology"],
    "tumour":             ["cancer biology"],
    "oncolog":            ["cancer biology"],
    "neoantigen":         ["cancer biology", "immunology"],
    "neoepitope":         ["cancer biology", "immunology"],
    "vaccine":            ["immunology"],
    "immune":             ["immunology"],
    "cytokine":           ["immunology"],
    "macrophage":         ["immunology", "cell biology"],
    "t cell":             ["immunology"],
    "b cell":             ["immunology"],

    # neuro
    "neuro":              ["neuroscience"],
    "synap":              ["neuroscience", "cell biology"],
    "axon":               ["neuroscience", "cell biology"],

    # target/protein
    "brca1":              ["cancer biology", "genetics", "molecular biology"],
    "brca2":              ["cancer biology", "genetics", "molecular biology"],
    "brd4":               ["cancer biology", "biochemistry", "pharmacology and toxicology"],
    "kinase":             ["biochemistry", "pharmacology and toxicology"],
    "bromodomain":        ["biochemistry", "molecular biology"],
    "protac":             ["biochemistry", "pharmacology and toxicology"],
    "drug":               ["pharmacology and toxicology"],
    "pharma":             ["pharmacology and toxicology"],

    # microbial / plant
    "plant":              ["plant biology"],
    "microb":             ["microbiology"],
    "bacteri":            ["microbiology"],
    "virus":              ["microbiology"],
    "fungi":              ["microbiology"],
    "yeast":              ["microbiology", "cell biology"],

    # systems / synthetic
    "systems biol":       ["systems biology"],
    "synthetic biol":     ["synthetic biology"],
    "network":            ["systems biology", "bioinformatics"],
    "model":              ["systems biology"],
}


def categories_for(topic: Topic) -> list[str]:
    """Return the bioRxiv categories likely to contain papers about `topic`.

    Falls back to ['bioinformatics'] when nothing matches — bioinformatics is
    the broadest catch-all for the cross-disciplinary methodology work that
    long-tail-hunter is most often used for.
    """
    haystack = " ".join([topic.term, *topic.synonyms, *topic.obscure_synonyms,
                         *topic.adjacent_methods]).lower()
    hits: set[str] = set()
    for kw, cats in _KEYWORD_TO_CAT.items():
        if kw in haystack:
            hits.update(cats)
    if not hits:
        return ["bioinformatics"]
    # Stable, sorted for determinism.
    return sorted(hits)
