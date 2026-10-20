"""
Experiment 2 runner — Example 2.9: Non-concavity of Indirect Utility.

Reproduces the analytic verification that the indirect utility ū(z) is not
concave, despite the primitive utility u(x) = ln(x) being concave.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


from src.example_29 import (
    run_example_29,
    print_report,
    branch_safe,
    branch_risky,
    indirect_utility,
    solve_crossing_symbolic,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def plot_indirect_utility(result, save_path: str) -> None:
    """Plot branches and upper envelope to visualize non-concavity."""
    z = result.z_grid
    bs = result.branch_safe_values
    br = result.branch_risky_values
    iu = result.indirect_utility_values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: all three curves
    ax = axes[0]
    ax.plot(z, bs, "b-", linewidth=1.5, label="φ=0: ln(z)")
    ax.plot(z, br, "r--", linewidth=1.5, label="φ=1: ½[ln(z+2)+ln(z-1)]")
    ax.plot(z, iu, "k-", linewidth=2.5, label="ū(z) = max{·,·}")
    ax.axvline(x=2.0, color="gray", linestyle=":", linewidth=1, label="z*=2 (crossing)")
    ax.scatter([2.0], [np.log(2)], color="purple", s=100, zorder=5, label=f"ū(2)=ln2≈{np.log(2):.4f}")
    ax.set_xlabel("z (initial capital)")
    ax.set_ylabel("Expected Utility")
    ax.set_title("Indirect Utility ū(z) = max{ln(z), ½[ln(z+2)+ln(z-1)]}\n(Non-concave upper envelope)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([result.parameters["z_start"], result.parameters["z_end"]])

    # Right: second differences (non-concavity diagnostic)
    ax2 = axes[1]
    z_mid = z[1:-1]
    sd = result.second_differences
    ax2.plot(z_mid, sd, "k-", linewidth=1.5)
    ax2.axhline(y=0, color="red", linestyle="--", linewidth=1, label="y=0 (concavity boundary)")
    ax2.fill_between(z_mid, sd, 0, where=sd > 0, color="red", alpha=0.3, label="Non-concave region")
    ax2.fill_between(z_mid, sd, 0, where=sd < 0, color="blue", alpha=0.2, label="Concave region")
    # Mark crossing point
    ax2.axvline(x=2.0, color="gray", linestyle=":", linewidth=1, label="z*=2")
    ax2.set_xlabel("z")
    ax2.set_ylabel("Second finite difference Δ²ū(z)")
    ax2.set_title("Shape Diagnostic: Second Differences of ū(z)\n(Positive → non-concave)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {save_path}")


def main() -> None:
    """Run experiment 2 and save results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("\nRunning Experiment 2 — Example 2.9: Non-concavity of Indirect Utility\n")

    result = run_example_29(z_start=1.01, z_end=6.0, n_points=500)
    print_report(result)

    # Plot
    plot_path = os.path.join(RESULTS_DIR, "exp2_indirect_utility.png")
    plot_indirect_utility(result, plot_path)

    sym = result.symbolic_analysis
    summary = {
        "experiment": "exp_2",
        "title": "Non-concavity of Indirect Utility (Example 2.9)",
        "parameters": result.parameters,
        "symbolic_analysis": {
            "crossing_z": str(sym["crossing_z"]),
            "crossing_value": str(sym["crossing_value"]),
            "crossing_value_float": sym["crossing_value_float"],
            "left_derivative": str(sym["left_derivative"]),
            "left_derivative_float": sym["left_derivative_float"],
            "right_derivative": str(sym["right_derivative"]),
            "right_derivative_float": sym["right_derivative_float"],
            "derivative_jumps_up": sym["derivative_jumps_up"],
            "non_concave": sym["non_concave"],
        },
        "numerical_shape": {
            "max_second_difference": result.max_second_difference,
            "non_concave_detected": result.max_second_difference > 0,
        },
        "non_concave_flag": result.non_concave_flag,
        "expected": {
            "crossing_z": 2.0,
            "left_derivative": 0.5,
            "right_derivative": 0.625,
        },
        "all_match": (
            abs(sym["crossing_z"] - 2) < 1e-12
            and abs(sym["left_derivative_float"] - 0.5) < 1e-10
            and abs(sym["right_derivative_float"] - 0.625) < 1e-10
        ),
    }

    json_path = os.path.join(RESULTS_DIR, "exp2_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Saved JSON: {json_path}")

    print(f"\n✓ Experiment 2 complete. Non-concavity verified: {result.non_concave_flag}")
    return result


if __name__ == "__main__":
    main()
