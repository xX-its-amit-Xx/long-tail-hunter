"""Map a Topic to one or more bioRxiv subject categories.

bioRxiv's API has *no* free-text search — only category + date window. So to
search bioRxiv at all, we need a mapping from a topic's words to its likely
categories. Best-effort, keyword-based, deliberately conservative.

The 27 official bioRxiv categories are listed in `BIORXIV_CATEGORIES`.
"""
from __future__ import annotations
import re
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
    # 'spatial' alone misfires on 'spatial memory', 'spatial resolution'.
    "spatial transcript": ["genomics", "bioinformatics", "cell biology"],
    "spatial omics":      ["genomics", "bioinformatics", "cell biology"],
    "spatial proteom":    ["genomics", "bioinformatics", "cell biology"],
    "spatial genom":      ["genomics", "bioinformatics", "cell biology"],
    "transcriptom":       ["genomics", "bioinformatics"],
    "proteom":            ["biochemistry", "bioinformatics"],
    "metabolom":          ["biochemistry", "systems biology"],
    "deconvolution":      ["bioinformatics", "genomics"],
    # 'integration' alone misfires on 'viral integration site' / 'social
    # integration' — restrict to the omics contexts we actually mean.
    "multi-omic integration": ["bioinformatics", "systems biology"],
    "scatac integration": ["bioinformatics", "genomics"],
    "data integration":   ["bioinformatics", "systems biology"],
    "omics integration":  ["bioinformatics", "systems biology"],
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

    # neuro — 'neur' is a stem that covers neuro*, neural*, neuron*.
    "neur":               ["neuroscience"],
    "synap":              ["neuroscience", "cell biology"],
    "axon":               ["neuroscience", "cell biology"],
    "glia":               ["neuroscience", "cell biology"],
    "brain":              ["neuroscience"],

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

    # systems / synthetic — bare 'network' and 'model' were too greedy.
    "systems biol":       ["systems biology"],
    "synthetic biol":     ["synthetic biology"],
    "regulatory network": ["systems biology", "bioinformatics"],
    "gene network":       ["systems biology", "bioinformatics"],
    "metabolic network":  ["systems biology", "biochemistry"],
    "boolean network":    ["systems biology", "bioinformatics"],
    "model organism":     ["systems biology"],

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

    # ---- Evolutionary biology / ecology / pop-gen ----
    "evolution":          ["evolutionary biology"],
    "population genetic": ["evolutionary biology", "genetics"],
    "phylogen":           ["evolutionary biology"],
    "selection scan":     ["evolutionary biology", "genomics"],
    "fst":                ["evolutionary biology", "genomics"],
    "pool-seq":           ["evolutionary biology", "genomics", "bioinformatics"],
    "popoolation":        ["evolutionary biology", "genomics", "bioinformatics"],
    "melanism":           ["evolutionary biology", "ecology"],
    "polymorphism":       ["evolutionary biology", "genetics"],
    "cryptic species":    ["evolutionary biology", "zoology"],
    "species complex":    ["evolutionary biology", "zoology"],
    "barcoding":          ["evolutionary biology", "ecology"],
    "supergene":          ["evolutionary biology", "genetics"],
    "chromosomal inversion": ["evolutionary biology", "genetics"],
    "balancing selection":["evolutionary biology", "genetics"],
    "introgression":      ["evolutionary biology", "genetics"],
    "ancient dna":        ["evolutionary biology", "paleontology"],
    "edna":               ["ecology"],
    "environmental dna":  ["ecology"],
    "metabarcoding":      ["ecology"],
    "biodiversity":       ["ecology"],
    "thermal tolerance":  ["ecology", "physiology"],
    "diapause":           ["ecology", "evolutionary biology"],
    "ephippia":           ["ecology", "evolutionary biology"],
    "resurrection ecolog":["ecology", "evolutionary biology"],
    "rapid adaptation":   ["evolutionary biology", "ecology"],

    # ---- Plant / symbiosis / microbe-host ----
    "mycorrhiz":          ["plant biology", "microbiology"],
    "symbio":             ["microbiology", "ecology"],
    "rhizosphere":        ["plant biology", "microbiology"],
    "phyllosphere":       ["plant biology", "microbiology"],
    "syncom":             ["microbiology", "ecology"],
    "synthetic community":["microbiology", "ecology"],
    "syncomm":            ["microbiology", "ecology"],
    "strigolactone":      ["plant biology"],
    "salicylic acid":     ["plant biology"],
    "pipecolic acid":     ["plant biology", "immunology"],
    "systemic acquired":  ["plant biology", "immunology"],
    "nlr":                ["plant biology", "immunology"],
    "r-gene":             ["plant biology", "immunology"],
    "effector":           ["microbiology", "immunology"],
    "rubisco":            ["plant biology", "biochemistry"],
    "pangenome":          ["genomics", "evolutionary biology"],
    "persister":          ["microbiology"],
    "biofilm":            ["microbiology"],
    "antibiotic toleranc":["microbiology"],
    "toxin-antitoxin":    ["microbiology", "molecular biology"],
    "pppgpp":             ["microbiology", "molecular biology"],
    "(p)ppgpp":           ["microbiology", "molecular biology"],
    "bacteriophage":      ["microbiology"],
    "phage":              ["microbiology"],

    # ---- Synthetic biology gaps ----
    "toehold":            ["synthetic biology", "bioinformatics"],
    "riboregulator":      ["synthetic biology", "molecular biology"],
    "genetic circuit":    ["synthetic biology"],
    # 'orthogonal' alone misfires (orthogonal axis, etc) — tighten to
    # the synthetic-biology contexts that actually use it.
    "orthogonal receptor":    ["synthetic biology", "biochemistry"],
    "orthogonal translation": ["synthetic biology", "biochemistry"],
    "orthogonal ribosom":     ["synthetic biology", "biochemistry"],
    "orthogonal trna":        ["synthetic biology", "biochemistry"],
    "genetic code expansion": ["synthetic biology", "biochemistry"],
    "unnatural amino":    ["synthetic biology", "biochemistry"],
    "directed evolution": ["synthetic biology", "biochemistry"],
    "nupack":             ["synthetic biology", "bioinformatics"],

    # ---- Phenotype-shaped routing (gap caught by 2026-05-31 review) ----
    "depress":            ["neuroscience", "pathology"],
    "psychiat":           ["neuroscience", "pathology"],
    "mood":               ["neuroscience"],
    "anhedonia":          ["neuroscience", "pathology"],
    "anxiety":            ["neuroscience", "pathology"],
    "schizophren":        ["neuroscience", "pathology"],
    "bipolar":            ["neuroscience", "pathology"],
    "insulin resist":     ["physiology", "pathology"],
    "nafld":              ["physiology", "pathology"],
    "masld":              ["physiology", "pathology"],
    "fatty liver":        ["physiology", "pathology"],
    "hepatic":            ["physiology", "pathology"],
    "ataxia":             ["neuroscience", "pathology"],
    "friedreich":         ["neuroscience", "pathology"],
    "frataxin":           ["biochemistry", "neuroscience"],
    "iron-sulfur":        ["biochemistry"],
    "rare disease":       ["pathology", "genetics"],
    "neural tube defect": ["developmental biology", "pathology"],
    "planar cell polar":  ["developmental biology", "cell biology"],
}

# Sanity guard: silently drop any non-bioRxiv categories that may have been
# typed in above by accident. `BIORXIV_CATEGORIES` is the source of truth.
_KEYWORD_TO_CAT = {
    kw: [c for c in cats if c in BIORXIV_CATEGORIES]
    for kw, cats in _KEYWORD_TO_CAT.items()
}

# Leading-boundary-only matching: the keyword must START at a non-word
# boundary, but may continue into a word (so stem-style keys like
# "depress", "transcriptom", "neuro" still match "depression",
# "transcriptomic", "neuroscience"). This avoids the substring-inside-word
# false-positive trap where short acronyms like "als" matched "trials" or
# "ibd" matched "rabidly", while preserving the prefix-stem usage that the
# original substring matcher relied on.
_KEYWORD_PATTERNS = {
    kw: re.compile(r"(?<!\w)" + re.escape(kw), re.IGNORECASE)
    for kw in _KEYWORD_TO_CAT
}


def categories_for(topic: Topic) -> list[str]:
    """Return the bioRxiv categories likely to contain papers about `topic`.

    Falls back to ['bioinformatics'] when nothing matches — bioinformatics is
    the broadest catch-all for the cross-disciplinary methodology work that
    long-tail-hunter is most often used for.
    """
    haystack = " ".join([topic.term, *topic.synonyms, *topic.obscure_synonyms,
                         *topic.adjacent_methods])
    hits: set[str] = set()
    for kw, cats in _KEYWORD_TO_CAT.items():
        if _KEYWORD_PATTERNS[kw].search(haystack):
            hits.update(cats)
    if not hits:
        return ["bioinformatics"]
    # Stable, sorted for determinism.
    return sorted(hits)
