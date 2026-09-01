#!/usr/bin/env python3
"""Post re-ranking and effectiveness evaluation.

Applies an additional LHRR/CPRR/RFE re-ranking pass to an already-aggregated
ranked list (output of ``03_aggregate_rankings.py``) for a final refinement
step, then reports MAP / Precision@k / Recall@k through pyUDLF. Pass
``--no-post-rerank`` to only evaluate the aggregated list as-is, i.e. a
"rank aggregation only" configuration with no further refinement.

Requires pyUDLF to be installed (see the repository README). The command
below picks up where ``03_aggregate_rankings.py``'s example left off,
post-re-ranking output/borda_corel5k_swintf_CPRR.txt with CPRR.

Example:
    python 04_post_rerank_and_evaluate.py \\
        --input output/borda_corel5k_swintf_CPRR.txt \\
        --post-method CPRR \\
        --size-dataset 5000 \\
        --lists-file data/corel5k/corel5k_lists.txt \\
        --classes-file data/corel5k/corel5k_classes.txt \\
        --dataset corel5k
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from manifold_rank_fusion.udlf_rerank import (
    DATASET_K,
    RERANK_METHODS,
    configure_udlf,
    evaluate_ranking_file,
    run_rerank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Aggregated ranked-list file to refine/evaluate.")
    parser.add_argument("--post-method", choices=RERANK_METHODS, help="Re-ranking method for the post re-rank stage.")
    parser.add_argument("--no-post-rerank", action="store_true", help="Skip post re-ranking; evaluate --input directly.")
    parser.add_argument("--size-dataset", type=int, required=True)
    parser.add_argument("--lists-file", required=True)
    parser.add_argument("--classes-file", required=True)
    parser.add_argument("--dataset", choices=sorted(DATASET_K), help="Used to look up the paper's K value.")
    parser.add_argument("--k", type=int, help="Explicit K value (overrides --dataset lookup).")
    parser.add_argument("--l", type=int, default=1000)
    parser.add_argument("--t", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--log-file", help="Where to write the pyUDLF evaluation log.")
    parser.add_argument("--binary-path", help="Path to a local UDLF binary (optional).")
    parser.add_argument("--config-path", help="Path to a local UDLF config.ini (optional).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_udlf(binary_path=args.binary_path, config_path=args.config_path)

    if not args.no_post_rerank:
        if args.post_method is None:
            raise SystemExit("--post-method is required unless --no-post-rerank is set.")
        if args.k is not None:
            k = args.k
        elif args.dataset is not None:
            k = DATASET_K[args.dataset]
        else:
            raise SystemExit("Either --k or --dataset must be provided for post re-ranking.")

        run_rerank(
            method=args.post_method,
            input_file=args.input,
            size_dataset=args.size_dataset,
            lists_file=args.lists_file,
            classes_file=args.classes_file,
            k=k,
            l=args.l,
            t=args.t,
            output_file=False,
            output_log_file_path=args.log_file,
        )
    else:
        evaluate_ranking_file(
            ranked_list_file=args.input,
            size_dataset=args.size_dataset,
            lists_file=args.lists_file,
            classes_file=args.classes_file,
            top_k=args.top_k,
            output_log_file_path=args.log_file,
        )


if __name__ == "__main__":
    main()
