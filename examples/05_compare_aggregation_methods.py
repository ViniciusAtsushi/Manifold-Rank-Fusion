#!/usr/bin/env python3
"""Compare Borda Count, RRF, and CombSUM on the same rankings.

Aggregates the same pair of ranked lists (UMAP projection + a re-ranking
method's output) with all three fusion strategies and reports MAP /
Precision@k for each, for a direct side-by-side comparison. The
aggregation itself needs only numpy, but evaluating each result requires
pyUDLF (see the repository README).

Example:
    python 05_compare_aggregation_methods.py \\
        --umap-ranking data/corel5k/corel5k_swintf_umap.txt \\
        --rerank-ranking data/corel5k/corel5k_swintf_CPRR.txt \\
        --size-dataset 5000 \\
        --lists-file data/corel5k/corel5k_lists.txt \\
        --classes-file data/corel5k/corel5k_classes.txt \\
        --output-dir output/comparison_corel5k_swintf_CPRR
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from manifold_rank_fusion.aggregation import aggregate_batch
from manifold_rank_fusion.io_utils import read_ranked_lists, write_ranked_lists
from manifold_rank_fusion.udlf_rerank import configure_udlf, evaluate_ranking_file

METHODS = ["borda", "rrf", "combsum"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--umap-ranking", required=True)
    parser.add_argument("--rerank-ranking", required=True)
    parser.add_argument("--size-dataset", type=int, required=True)
    parser.add_argument("--lists-file", required=True)
    parser.add_argument("--classes-file", required=True)
    parser.add_argument("--output-dir", required=True, help="Directory to write the aggregated ranked lists to.")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--binary-path", help="Path to a local UDLF binary (optional).")
    parser.add_argument("--config-path", help="Path to a local UDLF config.ini (optional).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_udlf(binary_path=args.binary_path, config_path=args.config_path)

    umap_ranking = read_ranked_lists(args.umap_ranking)
    rerank_ranking = read_ranked_lists(args.rerank_ranking)

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    for method in METHODS:
        method_kwargs = {"k": args.rrf_k} if method == "rrf" else {}
        aggregated = aggregate_batch(
            umap_ranking,
            rerank_ranking,
            method=method,
            top_k=args.top_k,
            **method_kwargs,
        )

        output_path = os.path.join(args.output_dir, f"{method}_aggregated.txt")
        write_ranked_lists(aggregated, output_path)

        print(f"\n=== {method.upper()} ===")
        log = evaluate_ranking_file(
            ranked_list_file=output_path,
            size_dataset=args.size_dataset,
            lists_file=args.lists_file,
            classes_file=args.classes_file,
            top_k=args.top_k,
            output_log_file_path=os.path.join(args.output_dir, f"{method}_log.txt"),
        )
        results[method] = log

    print("\n=== Summary ===")
    for method, log in results.items():
        print(f"{method}: {log}")


if __name__ == "__main__":
    main()
