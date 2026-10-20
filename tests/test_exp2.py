"""
Tests for Experiment 2 — Example 2.9: Non-concavity of Indirect Utility.

Validates:
- Branch function correctness
- Crossing at z=2
- Derivative values 1/2 and 5/8
- Non-concavity flag
- Mathematical bounds (utility in range)
"""

from __future__ import annotations

import math
import pytest
import numpy as np
import sympy as sp

from src.example_29 import (
    branch_safe,
    branch_risky,
    indirect_utility,
    solve_crossing_symbolic,
    finite_difference_second_derivative,
    run_example_29,
)

TOL = 1e-10


# ---------------------------------------------------------------------------
# Branch function tests
# ---------------------------------------------------------------------------

class TestBranchFunctions:
    def test_branch_safe_at_z2(self):
        """branch_safe(2) = ln(2)."""
        assert abs(branch_safe(2.0) - math.log(2)) < TOL

    def test_branch_risky_at_z2(self):
        """branch_risky(2) = ½ ln(4) + ½ ln(1) = ½ ln 4 = ln 2."""
        val = branch_risky(2.0)
        expected = math.log(2)
        assert abs(val - expected) < TOL, f"branch_risky(2) = {val}, expected {expected}"

    def test_branches_equal_at_crossing(self):
        """Both branches must be equal at z=2."""
        assert abs(branch_safe(2.0) - branch_risky(2.0)) < TOL

    def test_branch_safe_below_risky_for_large_z(self):
        """For large z, risky branch dominates safe branch."""
        z_large = 5.0
        assert branch_risky(z_large) > branch_safe(z_large)

    def test_branch_safe_above_risky_near_1(self):
        """For z near 1+ε, safe branch dominates (risky → -∞)."""
        z_near = 1.01
        assert branch_safe(z_near) > branch_risky(z_near)

    def test_branch_safe_vectorized(self):
        z = np.array([2.0, 3.0, 4.0])
        vals = branch_safe(z)
        expected = np.log(z)
        np.testing.assert_allclose(vals, expected, atol=TOL)

    def test_branch_risky_vectorized(self):
        z = np.array([2.0, 3.0, 4.0])
        vals = branch_risky(z)
        expected = 0.5 * np.log(z + 2) + 0.5 * np.log(z - 1)
        np.testing.assert_allclose(vals, expected, atol=TOL)


# ---------------------------------------------------------------------------
# Indirect utility tests
# ---------------------------------------------------------------------------

class TestIndirectUtility:
    def test_indirect_utility_at_crossing(self):
        """ū(2) = ln(2)."""
        val = indirect_utility(2.0)
        assert abs(val - math.log(2)) < TOL

    def test_indirect_utility_is_max(self):
        """ū(z) ≥ both branches everywhere in domain."""
        z = np.linspace(1.01, 6.0, 100)
        iu = indirect_utility(z)
        np.testing.assert_array_less(branch_safe(z) - 1e-15, iu + 1e-12)
        np.testing.assert_array_less(branch_risky(z) - 1e-15, iu + 1e-12)

    def test_indirect_utility_equals_safe_near_1(self):
        """Near z=1, safe branch is active."""
        z_near = 1.05
        iu = indirect_utility(z_near)
        assert abs(iu - branch_safe(z_near)) < TOL

    def test_indirect_utility_equals_risky_for_large_z(self):
        """For z >> 2, risky branch is active."""
        z_large = 5.0
        iu = indirect_utility(z_large)
        assert abs(iu - branch_risky(z_large)) < TOL


# ---------------------------------------------------------------------------
# Symbolic analysis tests
# ---------------------------------------------------------------------------

class TestSymbolicAnalysis:
    def test_crossing_at_z2(self):
        """Crossing must be at z = 2."""
        sym = solve_crossing_symbolic()
        assert sym["crossing_z"] == 2, f"crossing_z = {sym['crossing_z']}"

    def test_crossing_value_is_ln2(self):
        """ū(2) = ln(2)."""
        sym = solve_crossing_symbolic()
        assert abs(sym["crossing_value_float"] - math.log(2)) < TOL

    def test_left_derivative_is_one_half(self):
        """Left derivative = 1/2."""
        sym = solve_crossing_symbolic()
        assert abs(sym["left_derivative_float"] - 0.5) < TOL, \
            f"left deriv = {sym['left_derivative_float']}"

    def test_right_derivative_is_five_eighths(self):
        """Right derivative = 5/8 = 0.625."""
        sym = solve_crossing_symbolic()
        assert abs(sym["right_derivative_float"] - 0.625) < TOL, \
            f"right deriv = {sym['right_derivative_float']}"

    def test_derivative_jumps_up(self):
        """Right derivative > left derivative (upward jump)."""
        sym = solve_crossing_symbolic()
        assert sym["derivative_jumps_up"] is True

    def test_non_concave_flag(self):
        """Non-concavity flag should be True."""
        sym = solve_crossing_symbolic()
        assert sym["non_concave"] is True

    def test_left_derivative_exact_rational(self):
        """d/dz ln(z)|_{z=2} = 1/2 as sympy rational."""
        sym = solve_crossing_symbolic()
        ld = sym["left_derivative"]
        assert ld == sp.Rational(1, 2), f"left_deriv = {ld}"

    def test_right_derivative_exact_rational(self):
        """½[(z+2)^{-1} + (z-1)^{-1}]|_{z=2} = ½(1/4 + 1) = 5/8."""
        sym = solve_crossing_symbolic()
        rd = sym["right_derivative"]
        assert rd == sp.Rational(5, 8), f"right_deriv = {rd}"


# ---------------------------------------------------------------------------
# Second-difference diagnostic tests
# ---------------------------------------------------------------------------

class TestSecondDifferences:
    def test_second_differences_positive_near_crossing(self):
        """Around z=2 the second differences should be positive (non-concave)."""
        z_grid = np.linspace(1.5, 2.5, 200)
        iu = indirect_utility(z_grid)
        sd = finite_difference_second_derivative(z_grid, iu)
        assert np.max(sd) > 0, "Expected positive second-differences near crossing"

    def test_length_of_second_differences(self):
        z_grid = np.linspace(1.01, 6.0, 100)
        iu = indirect_utility(z_grid)
        sd = finite_difference_second_derivative(z_grid, iu)
        assert len(sd) == len(z_grid) - 2


# ---------------------------------------------------------------------------
# Full run tests
# ---------------------------------------------------------------------------

class TestFullRun:
    def test_run_default_params(self):
        result = run_example_29()
        sym = result.symbolic_analysis
        assert abs(float(sym["crossing_z"]) - 2.0) < TOL
        assert abs(sym["left_derivative_float"] - 0.5) < TOL
        assert abs(sym["right_derivative_float"] - 0.625) < TOL
        assert result.non_concave_flag is True

    def test_run_max_second_diff_positive(self):
        result = run_example_29()
        assert result.max_second_difference > 0

    def test_run_grid_length(self):
        result = run_example_29(z_start=1.01, z_end=6.0, n_points=200)
        assert len(result.z_grid) == 200
        assert len(result.branch_safe_values) == 200
        assert len(result.indirect_utility_values) == 200

    def test_run_domain_enforcement(self):
        """z_start must be > 1."""
        with pytest.raises(AssertionError):
            run_example_29(z_start=0.5)

    def test_run_different_grid_same_conclusion(self):
        """Non-concavity should be robust to grid choice."""
        for n in [100, 300, 500]:
            result = run_example_29(z_start=1.01, z_end=6.0, n_points=n)
            assert result.non_concave_flag is True


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_z_value(self):
        """Evaluate at single z=3."""
        z = 3.0
        iu = indirect_utility(z)
        expected = max(branch_safe(z), branch_risky(z))
        assert abs(iu - expected) < TOL

    def test_branch_risky_requires_z_gt_1(self):
        """branch_risky(1) should produce -inf due to ln(0)."""
        val = branch_risky(1.0)
        assert not np.isfinite(val)

    def test_indirect_utility_monotone_increasing(self):
        """ū(z) should be non-decreasing on (1, ∞)."""
        z = np.linspace(1.01, 6.0, 300)
        iu = indirect_utility(z)
        diffs = np.diff(iu)
        assert np.all(diffs >= -1e-10), "Indirect utility should be non-decreasing"
