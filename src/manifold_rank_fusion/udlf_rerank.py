"""pyUDLF wrapper for rank-based manifold learning re-ranking.

Wraps three rank-based manifold learning / re-ranking methods -- LHRR,
CPRR, and RFE -- through pyUDLF (https://github.com/UDLF/pyUDLF), plus a
helper to evaluate an already-computed ranked-list file (e.g. after
aggregation) without applying any further re-ranking.

pyUDLF is not published on PyPI: install it from a local clone of
https://github.com/UDLF/pyUDLF (see the repository README).
"""

from __future__ import annotations

from typing import Optional

from pyUDLF import run_calls as udlf
from pyUDLF.utils import inputType

# K parameter per dataset, adjusted according to dataset size (L = 1000 and
# T = 2 are kept fixed for all datasets and methods).
DATASET_K = {
    "flowers": 80,
    "corel5k": 100,
    "pets": 100,
    "cub200": 120,
    "dogs": 120,
}

RERANK_METHODS = ("CPRR", "LHRR", "RFE")


def configure_udlf(binary_path: Optional[str] = None, config_path: Optional[str] = None) -> None:
    """Point pyUDLF at a local UDLF binary/config install.

    Both arguments are optional: if omitted, pyUDLF auto-downloads and
    manages its own UDLF binary under ``~/.pyudlf``.

    Note: these must be *called* (``udlf.setConfigPath(path)``), not
    assigned (``udlf.setConfigPath = path``) -- the latter silently
    replaces the function object with a string and never actually changes
    the config path used by later calls.
    """
    if binary_path is not None:
        udlf.setBinaryPath(binary_path)
    if config_path is not None:
        udlf.setConfigPath(config_path)


def build_rerank_input(
    method: str,
    input_file: str,
    size_dataset: int,
    lists_file: str,
    classes_file: str,
    k: int,
    l: int = 1000,
    t: int = 2,
    output_file: bool = False,
    output_file_path: Optional[str] = None,
    output_log_file_path: Optional[str] = None,
) -> inputType.InputType:
    """Build the pyUDLF input configuration for CPRR, LHRR, or RFE.

    Uses the default ranked-list size (``L = 1000``) and iteration count
    (``T = 2``); ``K`` should come from ``DATASET_K`` for the dataset being
    processed.
    """
    if method not in RERANK_METHODS:
        raise ValueError(f"method must be one of {RERANK_METHODS}, got {method!r}")

    input_data = inputType.InputType()
    input_data.set_dataset_size(size_dataset)
    input_data.set_task("UDL")
    input_data.set_lists_file(lists_file)
    input_data.set_classes_file(classes_file)
    input_data.set_input_file(input_file)
    input_data.set_method_name(method)

    if method == "CPRR":
        input_data.set_method_parameters("CPRR", l=l, k=k, t=t)
    elif method == "LHRR":
        input_data.set_method_parameters("LHRR", k=k, l=l, t=t)
    elif method == "RFE":
        input_data.set_method_parameters(
            "RFE",
            k=k,
            t=t,
            l=l,
            pa=0.1,
            th_cc=0,
            rerank_by_emb=False,
            export_embeddings=False,
            perform_ccs=False,
        )

    input_data.set_output_file(output_file)
    if output_file:
        # pyUDLF's own InputType defaults to OUTPUT_FILE_FORMAT=MATRIX /
        # OUTPUT_RK_FORMAT=ALL; without forcing these, output_file=True
        # silently writes a full distance matrix instead of a ranked-list
        # file, which then reads back as garbage everywhere downstream.
        input_data.set_output_file_format("RK")
        input_data.set_output_rk_format("NUM")
    if output_file_path is not None:
        input_data.set_output_file_path(output_file_path)
    if output_log_file_path is not None:
        input_data.set_output_log_file_path(output_log_file_path)

    return input_data


def run_rerank(
    method: str,
    input_file: str,
    size_dataset: int,
    lists_file: str,
    classes_file: str,
    k: int,
    l: int = 1000,
    t: int = 2,
    output_file: bool = False,
    output_file_path: Optional[str] = None,
    output_log_file_path: Optional[str] = None,
):
    """Run one of CPRR/LHRR/RFE and return the pyUDLF output object."""
    input_data = build_rerank_input(
        method=method,
        input_file=input_file,
        size_dataset=size_dataset,
        lists_file=lists_file,
        classes_file=classes_file,
        k=k,
        l=l,
        t=t,
        output_file=output_file,
        output_file_path=output_file_path,
        output_log_file_path=output_log_file_path,
    )
    output = udlf.run(input_data, get_output=True)
    output.print_log()
    return output


def evaluate_ranking_file(
    ranked_list_file: str,
    size_dataset: int,
    lists_file: str,
    classes_file: str,
    top_k: int = 1000,
    output_log_file_path: Optional[str] = None,
):
    """Compute MAP / Precision@k / Recall@k for an existing ranked-list file.

    Applies ``UDL_METHOD = NONE``, i.e. no further re-ranking is performed;
    use this to evaluate aggregated (Borda Count / RRF / CombSUM) or
    post-re-ranked lists as-is.
    """
    input_data = inputType.InputType()
    input_data.set_dataset_size(size_dataset)
    input_data.set_task("UDL")
    input_data.set_lists_file(lists_file)
    input_data.set_classes_file(classes_file)
    input_data.set_input_file(ranked_list_file)
    input_data.set_method_name("NONE")
    input_data.set_param("PARAM_NONE_L", top_k)
    input_data.set_output_file(False)
    if output_log_file_path is not None:
        input_data.set_output_log_file_path(output_log_file_path)

    output = udlf.run(input_data, get_output=True)
    output.print_log()
    return output.get_log()
