"""
Tests for Experiment 3 — Example 2.10: Finite-State DP under Transient Price Impact.

Validates:
- Scenario tree construction
- Utility function properties
- Pathwise accounting (recursive == explicit)
- DP consistency (forward matches backward)
- Grid-refinement stability
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from src.model import (
    build_rho_table,
    verify_liquidation,
    verify_recursive_vs_explicit,
    utility_piecewise_logistic,
)
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
    pathwise_accounting_check,
    grid_refinement_study,
    forward_terminal_cash,
)

TOL = 1e-10


# ---------------------------------------------------------------------------
# Scenario tree tests
# ---------------------------------------------------------------------------

class TestScenarioTree:
    def test_tree_has_correct_leaf_count(self):
        root, levels = build_scenario_tree()
        leaves = get_all_leaf_paths(root)
        assert len(leaves) == 2**T  # 16 for T=4

    def test_leaf_probabilities_sum_to_one(self):
        root, levels = build_scenario_tree()
        leaves = get_all_leaf_paths(root)
        total_prob = sum(path[-1].prob for path in leaves)
        assert abs(total_prob - 1.0) < TOL, f"Total prob = {total_prob}"

    def test_root_probability_is_one(self):
        root, _ = build_scenario_tree()
        assert abs(root.prob - 1.0) < TOL

    def test_root_price_is_zero(self):
        root, _ = build_scenario_tree()
        assert abs(root.price - 0.0) < TOL

    def test_level_sizes(self):
        root, levels = build_scenario_tree()
        for t, nodes in enumerate(levels):
            assert len(nodes) == 2**t, f"Level {t}: expected {2**t}, got {len(nodes)}"

    def test_each_node_has_correct_children(self):
        root, levels = build_scenario_tree()
        for t in range(T):
            for node in levels[t]:
                assert len(node.children) == 2

    def test_leaf_prices_are_sums_of_shocks(self):
        """Each leaf price should be sum of shock sequence."""
        root, levels = build_scenario_tree()
        leaves = get_all_leaf_paths(root)
        for path in leaves:
            # path[0].price = 0, path[t].price = sum of first t shocks
            price = 0.0
            for t in range(1, T + 1):
                shock = path[t].price - path[t - 1].price
                assert shock in [1.0, -1.0], f"Unexpected shock: {shock}"
                price += shock
            assert abs(path[T].price - price) < TOL


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestUtilityFunction:
    def test_utility_non_decreasing(self):
        x = np.linspace(-5, 10, 200)
        u = utility(x)
        diffs = np.diff(u)
        assert np.all(diffs >= -1e-10), "Utility must be non-decreasing"

    def test_utility_bounded_above(self):
        """For x >> 0, utility approaches 1 (normalized)."""
        u_large = utility(100.0)
        assert u_large <= 1.0 + 1e-10, f"Utility not bounded above: {u_large}"

    def test_utility_linear_for_negative(self):
        """u(x) = x for x <= 0."""
        for x in [-5.0, -2.0, -0.5, 0.0]:
            assert abs(utility(x) - x) < TOL, f"utility({x}) = {utility(x)}, expected {x}"

    def test_utility_positive_range(self):
        """For x > 0, utility should be between 0 and 1."""
        for x in [0.01, 1.0, 5.0, 10.0]:
            u = utility(x)
            assert 0 <= u <= 1.0 + 1e-10, f"utility({x}) = {u} out of [0,1]"

    def test_utility_vectorized(self):
        x = np.linspace(-3, 3, 50)
        u = utility(x)
        assert len(u) == 50

    def test_utility_matches_piecewise_logistic(self):
        """utility() in example_210 should match utility_piecewise_logistic in model."""
        x = np.linspace(-5, 10, 50)
        u1 = utility(x)
        u2 = utility_piecewise_logistic(x)
        np.testing.assert_allclose(u1, u2, atol=1e-12)


# ---------------------------------------------------------------------------
# Pathwise accounting tests
# ---------------------------------------------------------------------------

class TestPathwiseAccounting:
    def test_no_trade_strategy(self):
        """H = [0, 0, 0, 0]: zero impact, zero cash change."""
        root, _ = build_scenario_tree()
        leaf_paths = get_all_leaf_paths(root)
        H_zero = [0.0, 0.0, 0.0, 0.0]
        ok = pathwise_accounting_check(H_zero, leaf_paths)
        assert ok, "Zero strategy should trivially satisfy accounting"

    def test_simple_round_trip_strategy(self):
        """H = [1, 0, 0, -1]: recursive == explicit."""
        root, _ = build_scenario_tree()
        leaf_paths = get_all_leaf_paths(root)
        H = [1.0, 0.0, 0.0, -1.0]
        ok = pathwise_accounting_check(H, leaf_paths)
        assert ok

    def test_all_positive_strategy(self):
        H = [0.5, 0.5, 0.5, -1.5]
        assert verify_liquidation(H, tol=1e-10)
        root, _ = build_scenario_tree()
        leaf_paths = get_all_leaf_paths(root)
        ok = pathwise_accounting_check(H, leaf_paths)
        assert ok


# ---------------------------------------------------------------------------
# DP consistency tests
# ---------------------------------------------------------------------------

class TestDPConsistency:
    @pytest.fixture(scope="class")
    def dp_small(self):
        """Small action grid for fast testing."""
        z_grid = np.linspace(-2, 2, 10)
        return backward_dp(action_grid_points=5, z_grid=z_grid)

    def test_forward_matches_backward(self, dp_small):
        """Forward-evaluated policy must match DP value."""
        assert dp_small["forward_matches"], "Forward policy values must match DP values"

    def test_dp_matches_exhaustive(self, dp_small):
        """DP should equal exhaustive search (same computation)."""
        assert dp_small["dp_matches_exhaustive"]

    def test_best_values_are_finite(self, dp_small):
        assert np.all(np.isfinite(dp_small["G0_values"]))

    def test_best_policy_not_none(self, dp_small):
        """Every z should have a non-None optimal policy."""
        for pol in dp_small["best_policy"]:
            assert pol is not None, "Policy should not be None"

    def test_policy_satisfies_liquidation(self, dp_small):
        """Every optimal policy must satisfy H_4 = -(H_1+H_2+H_3)."""
        for pol in dp_small["best_policy"]:
            if pol is not None:
                h1, h2, h3 = pol
                h4 = -(h1 + h2 + h3)
                H = [h1, h2, h3, h4]
                assert verify_liquidation(H, tol=1e-10)

    def test_optimal_beats_no_trade_for_large_z(self, dp_small):
        """For large z, optimal should match or beat no-trade in some regime."""
        g0 = dp_small["G0_values"]
        nt = dp_small["no_trade_values"]
        # Optimal is always >= no-trade (no-trade is a feasible strategy)
        np.testing.assert_array_less(nt - 1e-10, g0 + 1e-10)


# ---------------------------------------------------------------------------
# Resilience factor tests
# ---------------------------------------------------------------------------

class TestResilienceFactors:
    def test_rho_diagonal_is_one(self):
        """ρ_{j,j} = 1 for all j."""
        rho = build_rho_table(R_RATE, T)
        for j in range(T + 1):
            assert abs(rho[j, j] - 1.0) < TOL

    def test_rho_decreasing_in_t(self):
        """For fixed j, ρ_{j,t} is non-increasing as t increases (r≥0)."""
        rho = build_rho_table(R_RATE, T)
        for j in range(T):
            for t in range(j, T):
                assert rho[j, t] >= rho[j, t + 1] - TOL

    def test_rho_correct_for_constant_r(self):
        """With constant r=0.01, ρ_{0,4} = exp(-4*0.01)."""
        rho = build_rho_table(R_RATE, T)
        expected = math.exp(-4 * 0.01)
        assert abs(rho[0, 4] - expected) < TOL


# ---------------------------------------------------------------------------
# Grid refinement tests
# ---------------------------------------------------------------------------

class TestGridRefinement:
    def test_refinement_values_converge(self):
        """Finer grids should not decrease G̃_0 values."""
        z_test = np.linspace(-1, 1, 5)
        ref = grid_refinement_study(grid_sizes=[5, 9], z_test=z_test)
        # Finer grid can only find a better or equal optimum
        g_coarse = ref[5]
        g_fine = ref[9]
        np.testing.assert_array_less(g_coarse - 1e-8, g_fine + 1e-8)

    def test_all_grids_produce_finite_values(self):
        z_test = np.linspace(-1, 1, 5)
        ref = grid_refinement_study(grid_sizes=[5, 9], z_test=z_test)
        for gs, vals in ref.items():
            assert np.all(np.isfinite(vals)), f"Grid {gs} has non-finite values"
