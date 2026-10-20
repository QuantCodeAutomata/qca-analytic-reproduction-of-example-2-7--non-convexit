"""
Experiment 4 runner — Stress Tests for Non-Monotone Liquidity.

Runs four families of independent finite-state DP instances to verify
theorem-consistency of the backward recursion even in non-monotone
liquidity environments.
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


from src.stress_tests import (
    run_all_families,
    print_summary,
    StressTestResult,
    make_family_A_instances,
    make_family_B_instances,
    make_family_C_instances,
    make_family_D_instances,
    run_instance,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def plot_family_results(results: list[StressTestResult], save_path: str) -> None:
    """Plot G̃_0(z) for each stress-test instance, organized by family."""
    families = {"A": [], "B": [], "C": [], "D": []}
    for r in results:
        if r.family in families:
            families[r.family].append(r)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    family_titles = {
        "A": "Family A: Non-Monotone Liquidity",
        "B": "Family B: Monotone Liquidity (Control)",
        "C": "Family C: Horizon Sensitivity",
        "D": "Family D: Utility-Shape Sensitivity",
    }

    for ax, (fam, rlist) in zip(axes.flatten(), families.items()):
        for r in rlist:
            mono_label = "(mono)" if r.is_monotone_liquidity else "(non-mono)"
            ax.plot(r.z_grid, r.dp_values,
                    label=f"{r.instance_id} {mono_label}", linewidth=1.5)
            ax.plot(r.z_grid, r.no_trade_values,
                    "--", linewidth=0.7, alpha=0.5)
        ax.set_xlabel("z")
        ax.set_ylabel("G̃_0(z)")
        ax.set_title(family_titles[fam])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Exp 4: DP Optimal Values G̃_0(z) by Family\n"
                 "(dashed = no-trade benchmark)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {save_path}")


def plot_monotonicity_comparison(results: list[StressTestResult], save_path: str) -> None:
    """Plot consistency metrics: monotone vs non-monotone instances."""
    mono_ids = [r.instance_id for r in results if r.is_monotone_liquidity]
    non_mono_ids = [r.instance_id for r in results if not r.is_monotone_liquidity]
    mono_consistent = [r.dp_matches_brute_force and r.forward_matches and r.accounting_ok
                       for r in results if r.is_monotone_liquidity]
    non_mono_consistent = [r.dp_matches_brute_force and r.forward_matches and r.accounting_ok
                           for r in results if not r.is_monotone_liquidity]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: consistency by instance
    ax = axes[0]
    all_ids = [r.instance_id for r in results]
    consistency = [1 if (r.dp_matches_brute_force and r.forward_matches and r.accounting_ok) else 0
                   for r in results]
    colors = ["steelblue" if r.is_monotone_liquidity else "orange" for r in results]
    ax.bar(all_ids, consistency, color=colors)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("Consistent (1=Yes)")
    ax.set_title("DP Internal Consistency by Instance\n(Blue=Monotone, Orange=Non-Monotone)")
    ax.tick_params(axis="x", rotation=45)

    # Scatter: max second-diff per instance
    ax2 = axes[1]
    max_sds = [r.max_second_diff for r in results]
    ids_full = [r.instance_id for r in results]
    ax2.bar(ids_full, max_sds, color=colors)
    ax2.axhline(y=1e-6, color="red", linestyle="--", linewidth=1, label="Non-concavity threshold")
    ax2.set_ylabel("Max second-difference of G̃_0")
    ax2.set_yscale("symlog", linthresh=1e-8)
    ax2.set_title("Non-Concavity Magnitude by Instance\n(above red line = non-concave)")
    ax2.legend()
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {save_path}")


def main() -> None:
    """Run experiment 4 and save results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("\nRunning Experiment 4 — Stress Tests: Non-Monotone Liquidity\n")

    t0 = time.time()
    results = run_all_families(tol=1e-10)
    elapsed = time.time() - t0
    print(f"\n  All families completed in {elapsed:.2f}s")

    print_summary(results)

    # Plots
    plot_family_results(results, os.path.join(RESULTS_DIR, "exp4_family_results.png"))
    plot_monotonicity_comparison(results, os.path.join(RESULTS_DIR, "exp4_consistency.png"))

    # Aggregate statistics
    n_total = len(results)
    n_consistent = sum(
        1 for r in results if r.dp_matches_brute_force and r.forward_matches and r.accounting_ok
    )
    n_non_concave = sum(1 for r in results if r.non_concavity_detected)
    n_non_mono = sum(1 for r in results if not r.is_monotone_liquidity)
    n_mono = n_total - n_non_mono
    n_non_mono_consistent = sum(
        1 for r in results
        if not r.is_monotone_liquidity and r.dp_matches_brute_force and r.forward_matches
    )

    summary = {
        "experiment": "exp_4",
        "title": "Stress Tests: Non-Monotone Liquidity",
        "total_instances": n_total,
        "consistent_instances": n_consistent,
        "consistency_rate": n_consistent / n_total,
        "non_monotone_instances": n_non_mono,
        "monotone_instances": n_mono,
        "non_monotone_consistent": n_non_mono_consistent,
        "non_concavity_detected_count": n_non_concave,
        "elapsed_seconds": elapsed,
        "instances": [
            {
                "instance_id": r.instance_id,
                "family": r.family,
                "T": r.T,
                "is_monotone_liquidity": r.is_monotone_liquidity,
                "liquidity_sequence": r.liquidity_sequence,
                "dp_matches_brute_force": r.dp_matches_brute_force,
                "brute_force_value": r.brute_force_value,
                "dp_value_at_sample": r.dp_value_at_sample,
                "forward_matches": r.forward_matches,
                "accounting_ok": r.accounting_ok,
                "non_concavity_detected": r.non_concavity_detected,
                "max_second_diff": r.max_second_diff,
            }
            for r in results
        ],
    }

    json_path = os.path.join(RESULTS_DIR, "exp4_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Saved JSON: {json_path}")

    print(f"\n✓ Experiment 4 complete.")
    print(f"  Consistency rate: {n_consistent}/{n_total} instances")
    print(f"  Non-monotone consistent: {n_non_mono_consistent}/{n_non_mono}")
    return results


if __name__ == "__main__":
    main()
