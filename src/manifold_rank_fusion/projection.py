"""Neighbor embedding projection (UMAP) and ranked-list generation.

UMAP is used to project high-dimensional deep features into a low-dimensional
space, and a ranked list is then generated for every query by ordering the
Euclidean distances between projected points, using a Ball Tree as the
indexing structure.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import umap
from sklearn.neighbors import BallTree

# Default UMAP hyperparameters: default umap-learn values (n_neighbors=15,
# min_dist=0.1) with n_components=2, Euclidean metric, and a fixed
# random_state for reproducibility.
DEFAULT_UMAP_PARAMS = dict(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=42,
)


def project_features(features: np.ndarray, **umap_params) -> np.ndarray:
    """Project high-dimensional features into a low-dimensional space with UMAP.

    Args:
        features: array of shape ``(n_items, n_dims)`` with the original
            deep-feature representations (e.g. loaded from a
            ``features_<descriptor>_<dataset>.npy`` file).
        **umap_params: overrides for ``DEFAULT_UMAP_PARAMS`` (e.g.
            ``n_neighbors=30`` to explore a different neighborhood size).

    Returns:
        The projected features, shape ``(n_items, n_components)``.
    """
    params = {**DEFAULT_UMAP_PARAMS, **umap_params}
    reducer = umap.UMAP(**params)
    return reducer.fit_transform(features)


def build_ranked_lists(
    features: np.ndarray, top_k: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ranked lists by ordering Euclidean distances with a Ball Tree.

    Args:
        features: array of shape ``(n_items, n_dims)`` to index (either the
            original features or a UMAP projection).
        top_k: size of the ranked list to keep for every query.

    Returns:
        A tuple ``(distances, ranked_lists)``, each of shape
        ``(n_items, top_k)``. ``ranked_lists[i]`` is the ranked list for
        query ``i``: the ids of the ``top_k`` nearest items, ordered from
        closest to farthest (``ranked_lists[i][0] == i`` for the query
        itself).
    """
    tree = BallTree(features)
    distances, ranked_lists = tree.query(features, k=top_k)
    return distances, ranked_lists


def project_and_rank(
    features: np.ndarray, top_k: int = 1000, **umap_params
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: UMAP projection followed by ranked-list generation."""
    projected = project_features(features, **umap_params)
    return build_ranked_lists(projected, top_k=top_k)
