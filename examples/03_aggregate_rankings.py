#!/usr/bin/env python3
"""Rank aggregation.

Combines the UMAP-projection ranking with a rank-based re-ranking result
(LHRR, CPRR, or RFE) using Borda Count -- the main aggregation strategy
used by this framework. RRF and CombSUM (alternative baselines) are also
available via ``--method``.

Only needs numpy -- no pyUDLF install required. The command below runs
directly on the bundled data/corel5k/ sample (Corel5k, Swin Transformer
features), combining its precomputed UMAP ranking and CPRR re-ranking.

Example:
    python 03_aggregate_rankings.py \\
        --umap-ranking data/corel5k/corel5k_swintf_umap.txt \\
        --rerank-ranking data/corel5k/corel5k_swintf_CPRR.txt \\
        --output output/borda_corel5k_swintf_CPRR.txt \\
        --method borda
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from manifold_rank_fusion.aggregation import aggregate_batch
from manifold_rank_fusion.io_utils import read_ranked_lists, write_ranked_lists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--umap-ranking", required=True, help="Ranked-list file from the UMAP projection.")
    parser.add_argument("--rerank-ranking", required=True, help="Ranked-list file from LHRR/CPRR/RFE re-ranking.")
    parser.add_argument("--output", required=True, help="Path to write the aggregated ranked-list file to.")
    parser.add_argument("--method", choices=["borda", "rrf", "combsum"], default="borda")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--rrf-k", type=int, default=60, help="k constant for RRF (only used with --method rrf).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    umap_ranking = read_ranked_lists(args.umap_ranking)
    rerank_ranking = read_ranked_lists(args.rerank_ranking)

    method_kwargs = {"k": args.rrf_k} if args.method == "rrf" else {}
    aggregated = aggregate_batch(
        umap_ranking,
        rerank_ranking,
        method=args.method,
        top_k=args.top_k,
        **method_kwargs,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_ranked_lists(aggregated, args.output)
    print(f"Wrote {len(aggregated)} aggregated ({args.method}) ranked lists to {args.output}")


if __name__ == "__main__":
    main()
