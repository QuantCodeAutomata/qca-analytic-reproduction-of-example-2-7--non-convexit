"""
Experiment 3 runner — Example 2.10: Finite-State DP under Transient Price Impact.

Implements the backward recursion, validates against brute-force exhaustive
search, checks pathwise accounting, and assesses qualitative non-concavity.
"""

from __future__ import annotations

import json
import os
import sys
import time

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


from src.example_210 import (
    T,
    R_RATE,
    DELTA,
    PROB_PLUS,
    PROB_MINUS,
    utility,
    build_scenario_tree,
    get_all_leaf_paths,
    backward_dp,
    grid_refinement_study,
    pathwise_accounting_check,
    print_report,
    forward_terminal_cash,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def plot_dp_results(dp_result: dict, save_path: str) -> None:
    """Plot G̃_0(z), u(z), and no-trade benchmark."""
    z = dp_result["z_grid"]
    g0 = dp_result["G0_values"]
    nt = dp_result["no_trade_values"]
    u_z = utility(z)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(z, u_z, "b--", linewidth=1.5, label="u(z) primitive utility")
    ax.plot(z, nt, "g:", linewidth=1.5, label="No-trade G̃_0 (H=0)")
    ax.plot(z, g0, "k-", linewidth=2.5, label="Optimal G̃_0(z)")
    ax.set_xlabel("z (initial capital)")
    ax.set_ylabel("Value")
    ax.set_title(f"Example 2.10: G̃_0(z) vs u(z)\n"
                 f"(T=4, grid={dp_result['action_grid_points']} pts, "
                 f"r=[0.01], δ=[10,10,15,5])")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: second differences diagnostic
    ax2 = axes[1]
    sd = np.diff(g0, n=2)
    z_mid = z[1:-1]
    ax2.plot(z_mid, sd, "k-", linewidth=1.5)
    ax2.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="y=0")
    ax2.fill_between(z_mid, sd, 0, where=sd > 0, color="red", alpha=0.3, label="Non-concave")
    ax2.fill_between(z_mid, sd, 0, where=sd < 0, color="blue", alpha=0.2, label="Concave")
    ax2.set_xlabel("z")
    ax2.set_ylabel("Δ²G̃_0(z)")
    ax2.set_title("Second Differences of G̃_0(z)\n(Non-concavity diagnostic)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {save_path}")


def plot_grid_refinement(refinement: dict, save_path: str) -> None:
    """Plot G̃_0(z) for various action-grid sizes."""
    fig, ax = plt.subplots(figsize=(10, 5))
    z_test = next(iter(refinement.values()))  # same z for all
    # Actually need z_test from study
    z_test = np.linspace(-2, 2, 10)
    for gs, vals in refinement.items():
        ax.plot(z_test, vals, label=f"Grid size={gs}", linewidth=1.5)
    ax.set_xlabel("z")
    ax.set_ylabel("G̃_0(z)")
    ax.set_title("Grid-Refinement Sensitivity Study\nG̃_0(z) for Different Action-Grid Sizes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {save_path}")


def main() -> None:
    """Run experiment 3 and save results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("\nRunning Experiment 3 — Example 2.10: Finite-State DP\n")

    # DP with standard grid (17 points)
    z_grid = np.linspace(-5, 5, 50)
    t0 = time.time()
    dp_result = backward_dp(action_grid_points=17, z_grid=z_grid)
    elapsed = time.time() - t0
    print(f"  DP completed in {elapsed:.2f}s")
    print_report(dp_result)

    # Pathwise accounting check on best policy at middle z
    root, levels = build_scenario_tree()
    leaf_paths = get_all_leaf_paths(root)
    mid_iz = len(z_grid) // 2
    sample_pol = dp_result["best_policy"][mid_iz]
    if sample_pol is not None:
        h4 = -(sample_pol[0] + sample_pol[1] + sample_pol[2])
        H_check = list(sample_pol) + [h4]
        acct_ok = pathwise_accounting_check(H_check, leaf_paths)
        print(f"\n  Pathwise accounting (recursive==explicit): {acct_ok}")

    # Grid-refinement study
    print("\n  Running grid-refinement study...")
    z_test = np.linspace(-2, 2, 10)
    refinement = grid_refinement_study(grid_sizes=[5, 9, 13, 17], z_test=z_test)

    # Plot
    plot_dp_results(dp_result, os.path.join(RESULTS_DIR, "exp3_dp_value_function.png"))
    plot_grid_refinement(refinement, os.path.join(RESULTS_DIR, "exp3_grid_refinement.png"))

    # Non-concavity assessment
    g0 = dp_result["G0_values"]
    sd = np.diff(g0, n=2)
    max_sd = float(np.max(sd))
    non_concave = max_sd > 1e-6
    print(f"\n  Shape: max second-diff = {max_sd:.4e}, non-concavity = {non_concave}")

    # Save results
    summary = {
        "experiment": "exp_3",
        "title": "Finite-State DP Replication (Example 2.10)",
        "parameters": {
            "T": T,
            "r": R_RATE,
            "delta": DELTA,
            "prob_up": PROB_PLUS,
            "prob_down": PROB_MINUS,
        },
        "action_grid_points": dp_result["action_grid_points"],
        "z_grid_range": [float(z_grid[0]), float(z_grid[-1])],
        "n_z": len(z_grid),
        "forward_matches": bool(dp_result["forward_matches"]),
        "dp_matches_exhaustive": bool(dp_result["dp_matches_exhaustive"]),
        "pathwise_accounting_ok": bool(acct_ok) if sample_pol is not None else None,
        "non_concavity_detected": bool(non_concave),
        "max_second_difference": max_sd,
        "dp_elapsed_seconds": elapsed,
        "grid_refinement": {str(gs): vals.tolist() for gs, vals in refinement.items()},
        "sample_G0_values": {
            str(round(float(z_grid[i]), 3)): round(float(g0[i]), 6)
            for i in np.linspace(0, len(z_grid) - 1, 10, dtype=int)
        },
    }

    json_path = os.path.join(RESULTS_DIR, "exp3_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Saved JSON: {json_path}")

    print(f"\n✓ Experiment 3 complete. DP consistent: {dp_result['forward_matches']}")
    return dp_result, refinement


if __name__ == "__main__":
    main()
