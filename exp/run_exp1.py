"""
Experiment 1 runner — Example 2.7: Non-convexity of Attainable Set.

Reproduces the paper's witness construction, evaluates the quadratic form,
verifies the contradiction, and saves results.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

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

from src.example_27 import (
    run_example_27,
    print_report,
    quadratic_form_numeric,
    derive_quadratic_form,
    DELTA,
    DELTA_SYM,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def plot_quadratic_form(save_path: str) -> None:
    """Plot the quadratic form Q(h1, h2) surface and the three evaluation points."""
    h1_grid = np.linspace(0, 2, 100)
    h2_grid = np.linspace(0, 1.5, 100)
    H1, H2 = np.meshgrid(h1_grid, h2_grid)
    Q = quadratic_form_numeric(H1, H2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: contour plot with witness points
    ax = axes[0]
    cs = ax.contourf(H1, H2, Q, levels=30, cmap="viridis")
    plt.colorbar(cs, ax=ax, label="Q(h₁, h₂)")
    pts = {"H=(1,1)": (1.0, 1.0, 4.7), "G=(1.5,0)": (1.5, 0.0, 4.725), "H̄=(1.25,0.5)": (1.25, 0.5, 4.79375)}
    colors = ["red", "blue", "green"]
    for (label, (x, y, _)), color in zip(pts.items(), colors):
        ax.scatter([x], [y], color=color, s=100, zorder=5, label=f"{label}: Q={_:.5f}")
    ax.set_xlabel("h₁")
    ax.set_ylabel("h₂")
    ax.set_title("Q(h₁,h₂) = (21/10)h₁² + (23/10)h₁h₂ + (3/10)h₂²\nContour with Witness Points")
    ax.legend(fontsize=8)

    # Right: bar chart of Q values
    ax2 = axes[1]
    labels = ["Q(H)\n(1,1)", "Q(G)\n(1.5,0)", "½(Q(H)+Q(G))\nmidpoint avg", "Q(H̄)\n(1.25,0.5)"]
    values = [4.7, 4.725, 4.7125, 4.79375]
    bar_colors = ["steelblue", "steelblue", "orange", "red"]
    bars = ax2.bar(labels, values, color=bar_colors)
    ax2.set_ylim(4.65, 4.82)
    ax2.axhline(y=4.7125, color="orange", linestyle="--", linewidth=1.5, label="Midpoint avg = 4.7125")
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{val}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Q value (impact cost)")
    ax2.set_title("Contradiction: Q(H̄) > ½(Q(H)+Q(G))\n⟹ S is not convex")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {save_path}")


def main() -> None:
    """Run experiment 1 and save results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\nRunning Experiment 1 — Example 2.7: Non-convexity of Attainable Set\n")

    # Run main computation
    result = run_example_27()
    print_report(result)

    # Derive quadratic form symbolically
    print("\nSymbolic derivation of Q(h1,h2):")
    Q_derived = derive_quadratic_form()
    print(f"  Q(h1,h2) = {Q_derived}")
    print(f"  Expanded = {sp.expand(Q_derived)}")

    # Plot
    plot_path = os.path.join(RESULTS_DIR, "exp1_quadratic_form.png")
    plot_quadratic_form(plot_path)

    # Save JSON summary
    summary = {
        "experiment": "exp_1",
        "title": "Non-convexity of Attainable Set S (Example 2.7)",
        "parameters": result.parameters,
        "strategies": result.strategies,
        "Q_H": result.Q_H,
        "Q_G": result.Q_G,
        "Q_Hbar": result.Q_Hbar,
        "midpoint_average": result.midpoint_average,
        "contradiction_gap": result.contradiction_gap,
        "is_non_convex": result.is_non_convex,
        "Q_H_exact": str(result.Q_H_sym),
        "Q_G_exact": str(result.Q_G_sym),
        "Q_Hbar_exact": str(result.Q_Hbar_sym),
        "liquidation_checks": result.liquidation_checks,
        "recursive_vs_explicit_checks": result.recursive_checks,
        "expected_values": {
            "Q_H": 4.7,
            "Q_G": 4.725,
            "Q_Hbar": 4.79375,
            "midpoint_average": 4.7125,
            "contradiction_gap": -0.08125,
        },
        "all_match": (
            abs(result.Q_H - 4.7) < 1e-12
            and abs(result.Q_G - 4.725) < 1e-12
            and abs(result.Q_Hbar - 4.79375) < 1e-12
        ),
    }

    json_path = os.path.join(RESULTS_DIR, "exp1_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Saved JSON: {json_path}")

    print(f"\n✓ Experiment 1 complete. Non-convexity reproduced: {result.is_non_convex}")
    return result


if __name__ == "__main__":
    main()
