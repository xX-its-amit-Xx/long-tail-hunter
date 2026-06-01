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

    # ---- Immunology subterms ----
    "tcr":                ["immunology"],
    "t-cell receptor":    ["immunology"],
    "antibody":           ["immunology"],
    "immunoglobulin":     ["immunology"],
    "mhc":                ["immunology"],
    "hla":                ["immunology", "genetics"],
    "complement":         ["immunology"],
    "regulatory t cell":  ["immunology"],
    "treg":               ["immunology"],
    "dendritic cell":     ["immunology", "cell biology"],
    "nk cell":            ["immunology"],
    "natural killer":     ["immunology"],
    "innate immun":       ["immunology"],
    "adaptive immun":     ["immunology"],

    # ---- Developmental biology ----
    "organoid":           ["developmental biology", "cell biology"],
    "gastruloid":         ["developmental biology"],
    "somite":             ["developmental biology"],
    "neural crest":       ["developmental biology", "neuroscience"],
    "neural tube":        ["developmental biology"],
    "embryo":             ["developmental biology"],
    "morphogen":          ["developmental biology"],
    "stem cell":          ["developmental biology", "cell biology"],
    "ipsc":               ["developmental biology", "cell biology"],

    # ---- Microscopy / imaging ----
    "cryo-em":            ["biophysics", "biochemistry"],
    "cryoem":             ["biophysics", "biochemistry"],
    "cryo em":            ["biophysics", "biochemistry"],
    "lattice light":      ["cell biology", "biophysics"],
    "light-sheet":        ["cell biology", "biophysics"],
    "expansion microsc":  ["cell biology", "biophysics"],
    "super-resolution":   ["cell biology", "biophysics"],
    "super resolution":   ["cell biology", "biophysics"],
    "stochastic optical": ["cell biology", "biophysics"],
    "single-molecule":    ["biophysics", "molecular biology"],

    # ---- Multi-omics ----
    "cite-seq":           ["genomics", "bioinformatics", "immunology"],
    "perturb-seq":        ["genomics", "bioinformatics", "genetics"],
    "mass cytometry":     ["immunology", "cell biology"],
    "cytof":              ["immunology", "cell biology"],
    "merfish":            ["genomics", "bioinformatics", "cell biology"],
    "seqfish":            ["genomics", "bioinformatics", "cell biology"],
    "slide-seq":          ["genomics", "bioinformatics"],
    "visium":             ["genomics", "bioinformatics"],
    "multi-omic":         ["bioinformatics", "systems biology"],
    "multiome":           ["bioinformatics", "systems biology"],

    # ---- Disease classes ----
    "alzheimer":          ["neuroscience", "pathology"],
    "parkinson":          ["neuroscience", "pathology"],
    "als":                ["neuroscience", "pathology"],
    "amyotrophic":        ["neuroscience", "pathology"],
    "huntington":         ["neuroscience", "pathology"],
    "ibd":                ["immunology", "pathology"],
    "inflammatory bowel": ["immunology", "pathology"],
    "multiple sclerosis": ["immunology", "neuroscience"],
    "diabetes":           ["physiology", "pathology"],
    "obesity":            ["physiology", "pathology"],
    "cardiovascular":     ["physiology", "pathology"],
    "fibrosis":           ["pathology", "cell biology"],
    "lupus":              ["immunology", "pathology"],
    "rheumatoid":         ["immunology", "pathology"],

    # ---- Structural biology / prediction ----
    "alphafold":          ["bioinformatics", "biophysics"],
    "structure predict":  ["bioinformatics", "biophysics"],
    "rosetta":            ["bioinformatics", "biophysics"],
    "molecular dynamics": ["biophysics", "biochemistry"],
    "x-ray crystal":      ["biophysics", "biochemistry"],
    "nmr":                ["biophysics", "biochemistry"],
}

# Sanity guard: silently drop any non-bioRxiv categories that may have been
# typed in above by accident. `BIORXIV_CATEGORIES` is the source of truth.
_KEYWORD_TO_CAT = {
    kw: [c for c in cats if c in BIORXIV_CATEGORIES]
    for kw, cats in _KEYWORD_TO_CAT.items()
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
