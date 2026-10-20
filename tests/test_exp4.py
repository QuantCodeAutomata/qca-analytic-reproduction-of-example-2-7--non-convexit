"""
Tests for Experiment 4 — Stress Tests: Non-Monotone Liquidity.

Validates:
- Parameter validation logic
- Liquidity monotonicity detection
- Instance runner correctness (DP == brute force, forward == DP)
- Accounting consistency on each family
- Utility function properties for all utility types
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from src.model import (
    build_rho_table,
    verify_liquidation,
    utility_piecewise_logistic,
    utility_logistic_bounded,
    utility_linear_bounded,
)
from src.stress_tests import (
    validate_instance_params,
    compute_liquidity_monotonicity,
    build_generic_tree,
    get_leaf_paths_generic,
    forward_cash_generic,
    expected_utility_for_strategy,
    brute_force_optimum,
    dp_optimum,
    run_instance,
    make_family_A_instances,
    make_family_B_instances,
    make_family_C_instances,
    make_family_D_instances,
    StressTestInstance,
    _select_utility,
)

TOL = 1e-8


# ---------------------------------------------------------------------------
# Parameter validation tests
# ---------------------------------------------------------------------------

class TestParameterValidation:
    def test_valid_params_passes(self):
        validate_instance_params(3, [0.01]*3, [5.0]*3, 1.0, 0.6)  # must not raise

    def test_horizon_too_small_raises(self):
        with pytest.raises(AssertionError):
            validate_instance_params(1, [0.01], [5.0], 1.0, 0.6)

    def test_depth_below_min_raises(self):
        with pytest.raises(AssertionError):
            validate_instance_params(2, [0.01]*2, [0.05, 5.0], 1.0, 0.6)

    def test_negative_resilience_raises(self):
        with pytest.raises(AssertionError):
            validate_instance_params(2, [-0.01, 0.01], [5.0, 5.0], 0.5, 0.6)

    def test_prob_out_of_bounds_raises(self):
        with pytest.raises(AssertionError):
            validate_instance_params(2, [0.01]*2, [5.0]*2, 0.5, 1.0)
        with pytest.raises(AssertionError):
            validate_instance_params(2, [0.01]*2, [5.0]*2, 0.5, 0.0)


# ---------------------------------------------------------------------------
# Liquidity monotonicity tests
# ---------------------------------------------------------------------------

class TestLiquidityMonotonicity:
    def test_constant_depth_zero_r_is_monotone(self):
        """ρ_{0,t}^2 δ_t = 1*δ_t = constant => monotone."""
        rho = build_rho_table([0.0, 0.0, 0.0], 3)
        delta = [5.0, 5.0, 5.0]
        seq, is_mono = compute_liquidity_monotonicity(rho, delta)
        assert is_mono

    def test_decreasing_depth_zero_r_is_monotone(self):
        """Decreasing depths with r=0 => non-increasing ρ²δ => monotone."""
        rho = build_rho_table([0.0, 0.0, 0.0], 3)
        delta = [10.0, 5.0, 2.0]
        seq, is_mono = compute_liquidity_monotonicity(rho, delta)
        assert is_mono

    def test_increasing_depth_zero_r_is_non_monotone(self):
        """Increasing depths with r=0 => ρ²δ = δ increasing => non-monotone."""
        rho = build_rho_table([0.0, 0.0, 0.0], 3)
        delta = [1.0, 5.0, 10.0]
        seq, is_mono = compute_liquidity_monotonicity(rho, delta)
        assert not is_mono

    def test_family_A_has_non_monotone_instances(self):
        """At least some family A instances should be non-monotone."""
        instances = make_family_A_instances()
        non_mono_count = 0
        for inst in instances:
            rho = build_rho_table(inst.r, inst.T)
            seq, is_mono = compute_liquidity_monotonicity(rho, inst.delta)
            if not is_mono:
                non_mono_count += 1
        assert non_mono_count > 0, "Family A should have non-monotone instances"

    def test_family_B_has_monotone_instances(self):
        """Family B should have some monotone instances."""
        instances = make_family_B_instances()
        mono_count = 0
        for inst in instances:
            rho = build_rho_table(inst.r, inst.T)
            seq, is_mono = compute_liquidity_monotonicity(rho, inst.delta)
            if is_mono:
                mono_count += 1
        assert mono_count > 0, "Family B should have monotone instances"


# ---------------------------------------------------------------------------
# Scenario tree tests
# ---------------------------------------------------------------------------

class TestGenericTree:
    @pytest.mark.parametrize("T", [2, 3, 4])
    def test_leaf_count(self, T):
        root, _ = build_generic_tree(T, 1.0, -1.0, 0.6)
        leaves = get_leaf_paths_generic(root, T)
        assert len(leaves) == 2**T

    @pytest.mark.parametrize("T", [2, 3, 4])
    def test_leaf_probabilities_sum_to_one(self, T):
        root, _ = build_generic_tree(T, 1.0, -1.0, 0.6)
        leaves = get_leaf_paths_generic(root, T)
        total = sum(p[-1].prob for p in leaves)
        assert abs(total - 1.0) < TOL

    def test_prices_are_cumulative_sums(self):
        T = 3
        root, _ = build_generic_tree(T, 1.0, -1.0, 0.5)
        leaves = get_leaf_paths_generic(root, T)
        for path in leaves:
            for t in range(1, T + 1):
                shock = path[t].price - path[t - 1].price
                assert shock in [1.0, -1.0]


# ---------------------------------------------------------------------------
# Utility function tests for all families
# ---------------------------------------------------------------------------

class TestUtilityFunctions:
    @pytest.mark.parametrize("name", ["logistic_bounded", "linear_bounded", "piecewise_logistic"])
    def test_utility_non_decreasing(self, name):
        util_fn = _select_utility(name)
        x = np.linspace(-5, 10, 100)
        u = util_fn(x)
        diffs = np.diff(u)
        assert np.all(diffs >= -1e-10), f"{name} utility not non-decreasing"

    @pytest.mark.parametrize("name", ["logistic_bounded", "linear_bounded", "piecewise_logistic"])
    def test_utility_bounded_above(self, name):
        util_fn = _select_utility(name)
        x = np.array([100.0, 1000.0])
        u = util_fn(x)
        assert np.all(u <= 2.0 + 1e-6), f"{name} utility exceeds bound at large x: {u}"

    def test_unknown_utility_raises(self):
        with pytest.raises(ValueError):
            _select_utility("nonexistent_utility")


# ---------------------------------------------------------------------------
# DP vs brute-force cross-validation
# ---------------------------------------------------------------------------

class TestDPvsBruteForce:
    @pytest.fixture(scope="class")
    def simple_instance(self):
        return StressTestInstance(
            instance_id="test_simple",
            family="test",
            T=2,
            r=[0.0, 0.0],
            delta=[3.0, 3.0],
            prob_up=0.6,
            eps_plus=1.0,
            eps_minus=-1.0,
            z_grid_range=(-1.0, 1.0),
            n_z=5,
            action_grid_range=(-2.0, 2.0),
            n_action=5,
            utility_name="logistic_bounded",
            B=0.0,
            zeta_0=0.0,
            delta_min=0.5,
        )

    def test_dp_matches_brute_force(self, simple_instance):
        result = run_instance(simple_instance, tol=1e-10)
        assert result.dp_matches_brute_force, \
            f"DP={result.dp_value_at_sample}, BF={result.brute_force_value}"

    def test_forward_matches_dp(self, simple_instance):
        result = run_instance(simple_instance, tol=1e-10)
        assert result.forward_matches

    def test_accounting_ok(self, simple_instance):
        result = run_instance(simple_instance, tol=1e-10)
        assert result.accounting_ok

    def test_result_has_correct_family(self, simple_instance):
        result = run_instance(simple_instance)
        assert result.family == "test"

    def test_result_horizon_matches(self, simple_instance):
        result = run_instance(simple_instance)
        assert result.T == 2


# ---------------------------------------------------------------------------
# Family A: non-monotone liquidity
# ---------------------------------------------------------------------------

class TestFamilyA:
    @pytest.fixture(scope="class")
    def family_a_results(self):
        instances = make_family_A_instances()
        return [run_instance(inst, tol=1e-8) for inst in instances]

    def test_all_dp_match_brute_force(self, family_a_results):
        for r in family_a_results:
            assert r.dp_matches_brute_force, \
                f"Instance {r.instance_id}: DP={r.dp_value_at_sample}, BF={r.brute_force_value}"

    def test_all_forward_match_dp(self, family_a_results):
        for r in family_a_results:
            assert r.forward_matches, f"Instance {r.instance_id}: forward mismatch"

    def test_all_accounting_ok(self, family_a_results):
        for r in family_a_results:
            assert r.accounting_ok, f"Instance {r.instance_id}: accounting error"

    def test_dp_values_are_finite(self, family_a_results):
        for r in family_a_results:
            assert np.all(np.isfinite(r.dp_values)), f"Instance {r.instance_id}: non-finite DP"


# ---------------------------------------------------------------------------
# Family B: monotone liquidity control
# ---------------------------------------------------------------------------

class TestFamilyB:
    @pytest.fixture(scope="class")
    def family_b_results(self):
        instances = make_family_B_instances()
        return [run_instance(inst, tol=1e-8) for inst in instances]

    def test_all_dp_match_brute_force(self, family_b_results):
        for r in family_b_results:
            assert r.dp_matches_brute_force, f"Instance {r.instance_id}"

    def test_all_accounting_ok(self, family_b_results):
        for r in family_b_results:
            assert r.accounting_ok, f"Instance {r.instance_id}"


# ---------------------------------------------------------------------------
# Family C: horizon sensitivity
# ---------------------------------------------------------------------------

class TestFamilyC:
    def test_all_horizons_run_successfully(self):
        instances = make_family_C_instances()
        for inst in instances:
            result = run_instance(inst, tol=1e-8)
            assert result.dp_matches_brute_force, f"T={inst.T}: DP mismatch"
            assert result.accounting_ok, f"T={inst.T}: accounting error"

    def test_correct_horizons_tested(self):
        instances = make_family_C_instances()
        horizons = [inst.T for inst in instances]
        assert set(horizons) == {2, 3, 4, 5}


# ---------------------------------------------------------------------------
# Family D: utility shape sensitivity
# ---------------------------------------------------------------------------

class TestFamilyD:
    def test_all_utilities_run_successfully(self):
        instances = make_family_D_instances()
        for inst in instances:
            result = run_instance(inst, tol=1e-8)
            assert result.dp_matches_brute_force, f"Utility {inst.utility_name}: DP mismatch"

    def test_correct_utilities_tested(self):
        instances = make_family_D_instances()
        utility_names = [inst.utility_name for inst in instances]
        expected = {"logistic_bounded", "linear_bounded", "piecewise_logistic"}
        assert expected.issubset(set(utility_names))
