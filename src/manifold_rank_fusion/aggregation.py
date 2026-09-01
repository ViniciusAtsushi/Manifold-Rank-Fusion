"""Unsupervised rank aggregation strategies.

Three rank aggregation methods are implemented here, all operating
directly on ranked lists (sequences of item ids ordered from most to least
relevant), not on raw distances or similarity scores:

- ``borda_count``: the main aggregation strategy used by this framework.
  Each item receives, per list, a score equal to the number of candidates
  ranked below it; scores are summed across lists and the final ranking is
  sorted by descending score.
- ``reciprocal_rank_fusion``: an alternative aggregation baseline. Each
  item receives a score of ``1 / (k + r)``, where ``r`` is its 1-indexed
  rank position in a list and ``k = 60`` by default (Cormack et al., 2009).
- ``combsum``: another alternative aggregation baseline. CombSUM classically
  operates on normalized similarity/distance scores, but the re-ranking
  methods this framework combines (LHRR, CPRR, RFE) do not directly expose
  geometric distances that could be normalized and combined with
  projection-based ranking distances. This implementation instead uses a
  positional score transformation: each item receives a score of
  ``1 / (r + 1)`` according to its rank position ``r`` in a list, summed
  across lists.

All three functions accept an arbitrary number of ranked lists (not just
two), since the aggregation formulas generalize naturally to combining more
than two rankings.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Optional, Sequence

RankedList = Sequence[int]


def _scores_to_ranking(scores: dict, top_k: Optional[int]) -> List[int]:
    ranking = [item for item, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
    return ranking[:top_k] if top_k is not None else ranking


def borda_count(*ranked_lists: RankedList, top_k: Optional[int] = None) -> List[int]:
    """Aggregate ranked lists using the Borda Count method.

    For each list, an item at 0-indexed position ``i`` receives a score
    equal to the number of candidates ranked below it, i.e.
    ``len(list) - i - 1``. Scores are summed across all input lists and the
    final ranking is produced by sorting items by descending total score.

    All input lists are expected to have the same length, since the score
    of a position depends on the size of the list it came from.
    """
    if not ranked_lists:
        return []

    lengths = {len(lst) for lst in ranked_lists}
    if len(lengths) > 1:
        raise ValueError(
            f"borda_count expects all ranked lists to have the same length, got lengths {sorted(lengths)}"
        )

    scores: dict = defaultdict(float)
    for ranked_list in ranked_lists:
        num_candidates = len(ranked_list)
        for position, item in enumerate(ranked_list):
            scores[int(item)] += num_candidates - position - 1

    return _scores_to_ranking(scores, top_k)


def reciprocal_rank_fusion(
    *ranked_lists: RankedList, k: int = 60, top_k: Optional[int] = None
) -> List[int]:
    """Aggregate ranked lists using Reciprocal Rank Fusion (RRF).

    Each item receives a contribution of ``1 / (k + r)`` from every list it
    appears in, where ``r`` is its 1-indexed rank position. Contributions
    are summed across lists and the final ranking is sorted by descending
    score. Follows Cormack et al. (2009), with the standard default ``k = 60``.
    """
    if not ranked_lists:
        return []

    scores: dict = defaultdict(float)
    for ranked_list in ranked_lists:
        for position, item in enumerate(ranked_list):
            rank = position + 1
            scores[int(item)] += 1.0 / (k + rank)

    return _scores_to_ranking(scores, top_k)


def combsum(*ranked_lists: RankedList, top_k: Optional[int] = None) -> List[int]:
    """Aggregate ranked lists using the CombSUM strategy.

    CombSUM classically operates on normalized similarity/distance scores.
    Because the re-ranking methods used in this work (LHRR, CPRR, RFE) do
    not expose geometric distances that can be normalized and combined with
    UMAP-projection distances, this implementation instead uses a positional
    score transformation: each item receives a score of ``1 / (r + 1)``,
    where ``r`` is its 0-indexed rank position in a list. Scores are summed
    across lists.
    """
    if not ranked_lists:
        return []

    scores: dict = defaultdict(float)
    for ranked_list in ranked_lists:
        for position, item in enumerate(ranked_list):
            scores[int(item)] += 1.0 / (position + 1)

    return _scores_to_ranking(scores, top_k)


_METHODS = {
    "borda": borda_count,
    "rrf": reciprocal_rank_fusion,
    "combsum": combsum,
}


def aggregate_batch(
    *ranking_batches: Iterable[RankedList],
    method: str = "borda",
    top_k: Optional[int] = 1000,
    **method_kwargs,
) -> List[List[int]]:
    """Aggregate rankings for a batch of queries.

    Each element of ``ranking_batches`` is a sequence of per-query ranked
    lists (e.g. the rows of a ranked-lists file read via
    ``pyUDLF.utils.readData.read_ranked_lists_file_numeric``). All batches
    must contain the same number of queries and are combined query by
    query, in order.

    Args:
        *ranking_batches: two or more collections of per-query ranked
            lists to combine (e.g. the UMAP-projection ranking and a
            re-ranked list, both shaped ``(n_queries, list_size)``).
        method: one of ``"borda"``, ``"rrf"``, ``"combsum"``.
        top_k: number of items to keep per aggregated ranked list.
        **method_kwargs: extra keyword arguments forwarded to the
            underlying aggregation function (e.g. ``k=60`` for RRF).

    Returns:
        A list with one aggregated ranked list per query.
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown aggregation method '{method}'. Choose from {list(_METHODS)}.")

    if len(ranking_batches) < 2:
        raise ValueError("aggregate_batch needs at least two ranking batches to combine.")

    n_queries = {len(batch) for batch in ranking_batches}
    if len(n_queries) > 1:
        raise ValueError(f"All ranking batches must have the same number of queries, got {sorted(n_queries)}.")

    aggregate_fn = _METHODS[method]
    n = n_queries.pop()

    aggregated = []
    for query_idx in range(n):
        lists_for_query = [batch[query_idx] for batch in ranking_batches]
        aggregated.append(aggregate_fn(*lists_for_query, top_k=top_k, **method_kwargs))

    return aggregated
