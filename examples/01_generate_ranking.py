#!/usr/bin/env python3
"""Load deep features and generate an initial ranked list.

Features are loaded from a pre-extracted ``.npy`` file (feature extraction
itself is out of scope for this repository) and a ranked list is generated
by ordering Euclidean distances with a Ball Tree, either directly on the
original features or on a UMAP projection of them.

This repository bundles a ready-to-use sample under data/corel5k/
(Corel5k dataset, Swin Transformer features) so the commands below work
right out of a fresh clone, with no dataset download required. The bundled
corel5k_swintf.txt / corel5k_swintf_umap.txt are the precomputed outputs of
these exact commands (default random_state=42) -- diff against them to
check reproducibility.

Example:
    # Original-feature ranking (no projection)
    python 01_generate_ranking.py \\
        --features data/corel5k/features_swintf_corel5k.npy \\
        --output output/corel5k_swintf.txt

    # UMAP-projection ranking
    python 01_generate_ranking.py \\
        --features data/corel5k/features_swintf_corel5k.npy \\
        --output output/corel5k_swintf_umap.txt \\
        --umap
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from manifold_rank_fusion.io_utils import write_ranked_lists
from manifold_rank_fusion.projection import build_ranked_lists, project_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Path to a features_<descriptor>_<dataset>.npy file.")
    parser.add_argument("--output", required=True, help="Path to write the ranked-list .txt file to.")
    parser.add_argument("--top-k", type=int, default=1000, help="Ranked list size L (default: 1000).")
    parser.add_argument("--umap", action="store_true", help="Project features with UMAP before ranking.")
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    features = np.load(args.features)
    print(f"Loaded features: {features.shape}")

    if args.umap:
        features = project_features(
            features,
            n_components=args.n_components,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.random_state,
        )
        print(f"UMAP projection: {features.shape}")

    _, ranked_lists = build_ranked_lists(features, top_k=args.top_k)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_ranked_lists(ranked_lists, args.output)
    print(f"Wrote ranked lists to {args.output}")


if __name__ == "__main__":
    main()
