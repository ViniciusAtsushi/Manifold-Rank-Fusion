"""I/O helpers for the ranked-list file format used throughout this project.

Ranked-list files store one line per query, with whitespace-separated item
ids ordered from most to least relevant -- the same plain-text format
pyUDLF's ``readData``/``writeData`` produce and consume. The read/write
helpers here are a pure-Python reimplementation of that format (no pyUDLF
import), so generating rankings and aggregating them (``projection.py``,
``aggregation.py``) never requires installing pyUDLF -- only re-ranking and
its evaluation (``udlf_rerank.py``) do.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def read_ranked_lists(path: str, top_k: int = -1) -> np.ndarray:
    """Read a numeric ranked-list file into an ``(n_queries, list_size)`` array.

    Args:
        path: path to a ranked-list file (one line per query,
            whitespace-separated item ids).
        top_k: if positive, truncate every ranked list to its first
            ``top_k`` items; ``-1`` (default) keeps the full list.
    """
    with open(path, "r", encoding="utf-8") as f:
        rows = [line.split() for line in f if line.strip()]

    if top_k is not None and top_k != -1:
        rows = [row[:top_k] for row in rows]

    return np.array([[int(item) for item in row] for row in rows])


def write_ranked_lists(ranked_lists: Sequence[Sequence[int]], path: str) -> None:
    """Write ranked lists (one sequence of item ids per query) to a ranked-list file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in ranked_lists:
            f.write(" ".join(str(int(item)) for item in row))
            f.write(" \n")


def read_classes(lists_path: str, classes_path: str) -> Dict[int, int]:
    """Read the ``<item filename>:<class id>`` classes file into ``{item_id: class_id}``.

    Item ids are the 0-indexed line positions in the corresponding
    ``*_lists.txt`` file, matching the ids used in ranked-list files.
    """
    with open(lists_path, "r", encoding="utf-8") as f:
        item_names = [line.strip() for line in f if line.strip()]

    name_to_class: Dict[str, int] = {}
    with open(classes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, class_id = line.rsplit(":", 1)
            name_to_class[name] = int(class_id)

    return {idx: name_to_class[name] for idx, name in enumerate(item_names)}


def classes_as_list(lists_path: str, classes_path: str) -> List[int]:
    """Read classes as a 0-indexed list, suitable for ``pyUDLF.utils.evaluation``."""
    classes = read_classes(lists_path, classes_path)
    return [classes[idx] for idx in range(len(classes))]
