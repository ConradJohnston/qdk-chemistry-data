"""Benchmark different sparse state preparation methods on F2 molecule.

Loads the full F2 wavefunction from ``data/input_wavefunctions.json``, creates
subsets (num_dets = 2, 3, ..., N), runs resource estimation for four
state preparation methods(``gf2x``, ``gf2x_binary_encoding``,
``Rupprecht2026``,``Ramacciotti2024``), and produces:

  - ``f2_matrix_results.json`` — raw resource-estimate results.
  - ``f2_matrix_results.png`` — line plots of logical qubits, non-Clifford
    count, and Clifford count vs. number of configurations.
  - ``f2_matrix_results_stacked.png`` — stacked bar charts showing sparse
    vs. dense breakdown for three sample points per method.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from state_preparation_methods import (
    Ramacciotti2024,
    Rupprecht2026,
    combine_estimates,
    gf2x,
    gf2x_binary_encoding,
)


def scan_wfns(
    wfns: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Scan through a list of wavefunctions and return resource counts.

    Args:
        wfns: Ordered list of wavefunction dicts with keys ``"bitstrings"``
            and ``"coeffs"``, typically produced by :func:`_partition_wfn`.

    Returns:
        Dict keyed by method name (``"gf2x"``, ``"gf2x_binary_encoding"``,
        ``"Rupprecht2026"``, ``"Ramacciotti2024"``).  Each value is a list of
        result dicts with keys ``num_dets``, ``num_qubits``, ``sparse``
        (resource-estimate dict), and ``dense`` (resource-estimate dict).
    """

    results: dict[str, list[dict[str, Any]]] = {
        "Rupprecht2026": [],
        "Ramacciotti2024": [],
        "gf2x": [],
        "gf2x_binary_encoding": [],
    }
    for wfn in wfns:
        bitstrings = wfn["bitstrings"]
        coeffs = wfn["coeffs"]
        num_dets = len(bitstrings)
        num_qubits = len(bitstrings[0])

        base = {"num_dets": num_dets, "num_qubits": num_qubits}

        sparse_est, dense_est = gf2x(bitstrings, coeffs)
        results["gf2x"].append({**base, "sparse": sparse_est, "dense": dense_est})

        sparse_est, dense_est = gf2x_binary_encoding(bitstrings, coeffs)
        results["gf2x_binary_encoding"].append(
            {**base, "sparse": sparse_est, "dense": dense_est}
        )

        sparse_est, dense_est = Rupprecht2026(bitstrings, coeffs)
        results["Rupprecht2026"].append(
            {**base, "sparse": sparse_est, "dense": dense_est}
        )

        sparse_est, dense_est = Ramacciotti2024(bitstrings, coeffs)
        results["Ramacciotti2024"].append(
            {**base, "sparse": sparse_est, "dense": dense_est}
        )
    return results


def _partition_wfn(entry: dict) -> list[dict[str, Any]]:
    """Partition a wavefunction into prefix subsets of increasing configuration count.

    Starting from 2 configurations up to the full set, each subset uses the
    first ``n`` bitstrings and their re-normalised coefficients.

    Args:
        entry: Dict with keys ``"bitstrings"`` (list of bitstring strs) and
            ``"coeffs"`` (list of floats).

    Returns:
        List of wavefunction dicts with keys ``"bitstrings"`` and ``"coeffs"``,
        one per prefix size from 2 to ``len(entry["bitstrings"])`` inclusive.
    """
    bitstrings = entry["bitstrings"]
    coeffs = entry["coeffs"]
    n_total = len(bitstrings)
    partitions = []
    for n in range(2, n_total + 1):
        sub_coeffs = coeffs[:n]
        norm = np.linalg.norm(sub_coeffs)
        if norm > 0:
            sub_coeffs = [c / norm for c in sub_coeffs]
        partitions.append({"bitstrings": bitstrings[:n], "coeffs": sub_coeffs})
    return partitions


def run_molecule_benchmark(
    wfn_filepath: Path, molecule: str
) -> dict[str, list[dict[str, Any]]]:
    """Load a molecule's wavefunction from JSON and return resource counts.

    Args:
        wfn_filepath: Path to the JSON file containing per-molecule wavefunction
            data (e.g. ``data/input_wavefunctions.json``).
        molecule: Key identifying the molecule entry in the JSON file
            (e.g. ``"F2"``).

    Returns:
        Resource-count results as returned by :func:`scan_wfns`.

    Raises:
        KeyError: If ``molecule`` is not present in the JSON file.
    """
    with open(wfn_filepath, "r") as f:
        all_wfns = json.load(f)
    if molecule not in all_wfns:
        raise KeyError(f"Molecule {molecule!r} not found in {wfn_filepath}")
    wfns = _partition_wfn(all_wfns[molecule])
    return scan_wfns(wfns)


# Plotting helpers
METHOD_LABELS = {
    "gf2x": "Ours w/o binary encoding",
    "gf2x_binary_encoding": "Ours w/ binary encoding",
    "Rupprecht2026": "Rupprecht2026",
    "Ramacciotti2024": "Ramacciotti2024",
}
METHOD_COLORS = {
    "gf2x": "#D55E00",
    "gf2x_binary_encoding": "#0072B2",
    "Rupprecht2026": "#CC79A7",
    "Ramacciotti2024": "#E69F00",
}
METHOD_MARKERS = {
    "gf2x": "D",
    "gf2x_binary_encoding": "^",
    "Rupprecht2026": "s",
    "Ramacciotti2024": "o",
}
METHOD_ORDER = ["Ramacciotti2024", "gf2x", "gf2x_binary_encoding", "Rupprecht2026"]
STACKED_ORDER = ["Ramacciotti2024", "Rupprecht2026", "gf2x", "gf2x_binary_encoding"]
Y_LABELS = {
    "logical_qubits": "Logical Qubits",
    "non_clifford_count": "Non-Clifford Count",
    "clifford_count": "Clifford Count",
}
X_LABELS = {
    "num_dets": "Number of Configurations",
}


def _process_results(results: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Flatten per-method results into a flat list of row dicts for plotting.

    Args:
        results: Nested results dict as returned by :func:`scan_wfns`, keyed
            by method name with each value a list of result dicts.

    Returns:
        Flat list of row dicts, each containing ``method``, ``num_dets``,
        ``num_qubits``, ``logical_qubits``, ``non_clifford_count``,
        ``clifford_count``, ``sparse_non_clifford_count``,
        ``dense_non_clifford_count``, ``sparse_clifford_count``, and
        ``dense_clifford_count``.
    """
    processed: list[dict[str, Any]] = []
    for method, entries in results.items():
        for entry in entries:
            sparse = entry.get("sparse", {})
            dense = entry.get("dense", {})
            combined = combine_estimates(sparse, dense)

            row = {
                "method": method,
                "num_dets": entry["num_dets"],
                "num_qubits": entry["num_qubits"],
                **combined,
                "sparse_non_clifford_count": sparse.get("non_clifford_count", 0),
                "dense_non_clifford_count": dense.get("non_clifford_count", 0),
                "sparse_clifford_count": sparse.get("clifford_count", 0),
                "dense_clifford_count": dense.get("clifford_count", 0),
            }
            processed.append(row)
    return processed


def plot_performance_lines(
    results: dict[str, list[dict[str, Any]]],
    name: str,
    fig_dir: Path,
    x_key: str = "num_dets",
) -> None:
    """Plot logical-qubit, Clifford, and non-Clifford counts vs. number of configurations.

    Produces one subplot per active metric on a log-y scale and saves the
    figure to ``figures/{name}_matrix_results.png``.

    Args:
        results: Nested results dict as returned by :func:`scan_wfns`.
        name: Label appended to the output filename (e.g. ``"f2"``).
        fig_dir: Directory to save the figure.
        x_key: Column to use for the x-axis.  Defaults to ``"num_dets"``.
    """
    data = _process_results(results)
    # Determine valid metrics
    metrics = [k for k in Y_LABELS if any(row.get(k) for row in data)]
    if not metrics:
        return

    _, axes = plt.subplots(
        1, len(metrics), figsize=(6 * len(metrics), 5), squeeze=False
    )
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        methods = [m for m in METHOD_ORDER if m in set(d["method"] for d in data)]
        for method in methods:
            subset = [d for d in data if d["method"] == method]
            subset.sort(key=lambda x: x[x_key])
            xs = [d[x_key] for d in subset]
            ys = [d.get(metric, 0) for d in subset]

            ax.plot(
                xs,
                ys,
                linestyle="--",
                marker=METHOD_MARKERS.get(method, "o"),
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method),
                alpha=0.9,
            )

        ax.set_ylabel(Y_LABELS[metric], fontsize=16)
        ax.set_xlabel(X_LABELS[x_key], fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    axes[-1].legend(loc="best", fontsize=14)

    plt.tight_layout()
    if fig_dir is None:
        fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"{name}_matrix_results.png"
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_stacked_resources(
    results: dict[str, list[dict[str, Any]]],
    name: str,
    fig_dir: Path,
    metric: str = "non_clifford_count",
    y_label: str = "Non-Clifford Count",
) -> None:
    """Plot stacked bars showing sparse-isometry vs. dense-loading breakdown.

    Selects 3 evenly-spaced sample points per method and saves the figure to
    ``figures/{name}_matrix_results_stacked.png``.

    Args:
        results: Nested results dict as returned by :func:`scan_wfns`.
        name: Label appended to the output filename (e.g. ``"f2"``).
        fig_dir: Directory to save the figure.
        metric: Base metric name to stack.  The function expects keys
            ``sparse_{metric}`` and ``dense_{metric}`` in each row dict.
            Defaults to ``"non_clifford_count"``.
        y_label: Y-axis label string.  Defaults to ``"Non-Clifford Count"``.
    """
    data = _process_results(results)
    methods = [m for m in STACKED_ORDER if m in set(d["method"] for d in data)]

    # Setup layout
    group_gap = 1.0
    bar_width = 0.8
    x_positions: list[float] = []
    x_labels: list[str] = []
    tick_locs: list[tuple[float, str]] = []
    current_x: float = 0
    sparse_vals, dense_vals = [], []

    for method in methods:
        # Pick 3 evenly spaced points
        method_data = sorted(
            [d for d in data if d["method"] == method], key=lambda x: x["num_dets"]
        )
        indices = np.linspace(0, len(method_data) - 1, 3, dtype=int)
        subset = [method_data[i] for i in indices]
        group_start = current_x

        for entry in subset:
            sparse_vals.append(entry.get(f"sparse_{metric}", 0))
            dense_vals.append(entry.get(f"dense_{metric}", 0))
            x_positions.append(current_x)
            x_labels.append(f"{entry['num_dets']} configs")
            current_x += 1

        # Add label for method group
        tick_locs.append(
            ((group_start + current_x - 1) / 2, METHOD_LABELS.get(method, method))
        )
        current_x += group_gap

    if not x_positions:
        return

    _, ax = plt.subplots(figsize=(max(8, len(x_positions)), 6))

    ax.bar(
        x_positions, sparse_vals, bar_width, label="Sparse Isometry", color="#f0dc82"
    )
    ax.bar(
        x_positions,
        dense_vals,
        bar_width,
        bottom=sparse_vals,
        label="Dense Loading",
        color="#56B4E9",
    )

    # Double labeling x-axis (Wavefunction ID + Method Name)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=12, rotation=45)

    # Method labels
    for x_pos, label in tick_locs:
        ax.text(
            x_pos,
            -0.21,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            fontsize=12,
        )

    ax.set_ylabel(y_label, fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    # ax.set_yscale("log")
    ax.legend(fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"{name}_matrix_results_stacked.png"
    plt.savefig(fig_path, dpi=600)
    plt.close()


def main() -> None:
    """Parse command-line arguments and run the molecule benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark sparse state preparation methods on a molecule wavefunction."
    )
    parser.add_argument(
        "--wfn_path",
        default=Path(__file__).parent / "data" / "input_wavefunctions.json",
        type=Path,
        help="Path to the wavefunction JSON file (e.g. data/input_wavefunctions.json).",
    )
    parser.add_argument(
        "--molecule",
        type=str,
        default="F2",
        help="Molecule key in the JSON file (e.g. F2).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output",
        metavar="DIR",
        help="Directory for JSON results and figures. Default: %(default)s",
    )
    args = parser.parse_args()

    name = args.molecule.lower()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = run_molecule_benchmark(args.wfn_path, molecule=args.molecule)

    json_path = output_dir / f"{name}_matrix_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    plot_performance_lines(results, name, fig_dir=figures_dir)
    plot_stacked_resources(results, name, fig_dir=figures_dir)


if __name__ == "__main__":
    """The main entry point for the molecule benchmark script.

    The command-line to reproduce the F2 benchmark is:
    `python estimate_f2.py data/input_wavefunctions.json F2`.
    """
    main()
