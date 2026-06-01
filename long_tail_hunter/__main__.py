"""CLI entrypoint:

    python -m long_tail_hunter "neoantigen prediction" --kind method \\
        --syn neoepitope --syn "MHC binding prediction" \\
        --obscure "personalized cancer vaccine epitope"

Emits the SearchPlan as JSON on stdout.
"""
from __future__ import annotations
import argparse
import json
import sys
from .topic import Topic, TopicKind
from .hunter import plan
from .strategies import ALL_STRATEGIES


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="long_tail_hunter",
        description="Generate a long-tail search plan for a phenotype / method / target.",
    )
    p.add_argument("term", help="The thing to hunt (e.g. 'CRISPR base editor off-target').")
    p.add_argument("--kind", choices=[k.value for k in TopicKind],
                   default=TopicKind.MIXED.value)
    p.add_argument("--syn", action="append", default=[],
                   help="Standard synonym. Repeatable.")
    p.add_argument("--obscure", action="append", default=[],
                   help="Obscure / archaic / cross-discipline synonym. Repeatable.")
    p.add_argument("--organism", default=None)
    p.add_argument("--method", action="append", default=[],
                   help="Adjacent method. Repeatable.")
    p.add_argument("--famous", action="append", default=[],
                   help="Famous hit to track for suppression (DOI / repo / title).")
    p.add_argument("--exclude-term", action="append", default=[])
    p.add_argument("--only", action="append", default=None,
                   choices=list(ALL_STRATEGIES.keys()),
                   help="Restrict to these strategies. Repeatable.")
    p.add_argument("--exclude", action="append", default=None,
                   choices=list(ALL_STRATEGIES.keys()),
                   help="Drop these strategies. Repeatable.")
    p.add_argument("--list-strategies", action="store_true")
    p.add_argument("--no-dedupe", action="store_true")
    p.add_argument("--compact", action="store_true",
                   help="One-line JSON instead of pretty.")
    args = p.parse_args(argv)

    if args.list_strategies:
        for name, fn in ALL_STRATEGIES.items():
            doc = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
            print(f"{name}\t{doc}")
        return 0

    topic = Topic(
        term=args.term,
        kind=TopicKind(args.kind),
        synonyms=args.syn,
        obscure_synonyms=args.obscure,
        organism=args.organism,
        adjacent_methods=args.method,
        famous_hits=args.famous,
        exclude_terms=args.exclude_term,
    )

    sp = plan(
        topic,
        only=args.only,
        exclude=args.exclude,
        dedupe=not args.no_dedupe,
    )
    print(sp.to_json(indent=None if args.compact else 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
