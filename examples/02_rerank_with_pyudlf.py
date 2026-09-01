#!/usr/bin/env python3
"""Rank-based manifold learning re-ranking (LHRR / CPRR / RFE).

Applies one of LHRR, CPRR, or RFE to an initial ranked-list file through
pyUDLF. Typically run on the original-feature ranking, but the same script
can also be pointed at a UMAP-projection ranking if a UMAP-space
re-ranking variant is needed.

Requires pyUDLF to be installed (see the repository README). The command
below re-ranks the bundled data/corel5k/corel5k_swintf.txt sample with
CPRR; data/corel5k/corel5k_swintf_CPRR.txt is the precomputed output of
this exact command, provided so you can skip this step if you don't have
pyUDLF installed.

Example:
    python 02_rerank_with_pyudlf.py \\
        --method CPRR \\
        --input data/corel5k/corel5k_swintf.txt \\
        --output output/corel5k_swintf_CPRR \\
        --size-dataset 5000 \\
        --lists-file data/corel5k/corel5k_lists.txt \\
        --classes-file data/corel5k/corel5k_classes.txt \\
        --dataset corel5k
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from manifold_rank_fusion.udlf_rerank import DATASET_K, RERANK_METHODS, configure_udlf, run_rerank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=RERANK_METHODS)
    parser.add_argument("--input", required=True, help="Ranked-list .txt file to re-rank.")
    parser.add_argument("--output", required=True, help="Output path (without extension).")
    parser.add_argument("--size-dataset", type=int, required=True)
    parser.add_argument("--lists-file", required=True)
    parser.add_argument("--classes-file", required=True)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_K),
        help="Dataset name, used to look up the paper's K value. Overridden by --k if given.",
    )
    parser.add_argument("--k", type=int, help="Explicit K value (overrides --dataset lookup).")
    parser.add_argument("--l", type=int, default=1000, help="Ranked list size (default: 1000).")
    parser.add_argument("--t", type=int, default=2, help="Number of iterations (default: 2).")
    parser.add_argument("--log-file", help="Where to write the pyUDLF evaluation log.")
    parser.add_argument("--binary-path", help="Path to a local UDLF binary (optional).")
    parser.add_argument("--config-path", help="Path to a local UDLF config.ini (optional).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.k is not None:
        k = args.k
    elif args.dataset is not None:
        k = DATASET_K[args.dataset]
    else:
        raise SystemExit("Either --k or --dataset must be provided.")

    configure_udlf(binary_path=args.binary_path, config_path=args.config_path)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)

    run_rerank(
        method=args.method,
        input_file=args.input,
        size_dataset=args.size_dataset,
        lists_file=args.lists_file,
        classes_file=args.classes_file,
        k=k,
        l=args.l,
        t=args.t,
        output_file=True,
        output_file_path=args.output,
        output_log_file_path=args.log_file,
    )


if __name__ == "__main__":
    main()
