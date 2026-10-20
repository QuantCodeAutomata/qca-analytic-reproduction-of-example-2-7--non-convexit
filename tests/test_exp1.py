"""
Tests for Experiment 1 — Example 2.7: Non-convexity of Attainable Set.

Validates:
- Liquidation constraints
- Recursive == explicit terminal cash
- Exact quadratic form values
- Contradiction sign and non-convexity conclusion
- Edge cases and mathematical properties
"""

from __future__ import annotations

import pytest
import numpy as np
import sympy as sp

from src.model import (
    build_rho_table,
    terminal_cash_explicit,
    verify_liquidation,
    verify_recursive_vs_explicit,
    simulate_recursive,
)
from src.example_27 import (
    T,
    R,
    DELTA,
    H_STRATEGY,
    G_STRATEGY,
    HBAR_STRATEGY,
    DELTA_SYM,
    H_SYM,
    G_SYM,
    HBAR_SYM,
    impact_cost,
    impact_cost_symbolic,
    quadratic_form_numeric,
    quadratic_form_symbolic,
    derive_quadratic_form,
    run_example_27,
)

TOL = 1e-12


# ---------------------------------------------------------------------------
# Model parameter tests
# ---------------------------------------------------------------------------

class TestModelParameters:
    def test_horizon_is_3(self):
        assert T == 3

    def test_zero_resilience(self):
        assert R == [0.0, 0.0, 0.0]

    def test_depths_correct(self):
        assert DELTA == [1.0, 10.0, 10.0]

    def test_rho_all_ones_when_r_zero(self):
        """With r_t=0, all resilience factors ρ_{j,t} = 1."""
        rho = build_rho_table(R, T)
        for j in range(T + 1):
            for t in range(j, T + 1):
                assert abs(rho[j, t] - 1.0) < TOL, f"ρ[{j},{t}] = {rho[j,t]}, expected 1.0"


# ---------------------------------------------------------------------------
# Liquidation constraint tests
# ---------------------------------------------------------------------------

class TestLiquidation:
    def test_H_sums_to_zero(self):
        assert verify_liquidation(H_STRATEGY, tol=TOL)

    def test_G_sums_to_zero(self):
        assert verify_liquidation(G_STRATEGY, tol=TOL)

    def test_Hbar_sums_to_zero(self):
        assert verify_liquidation(HBAR_STRATEGY, tol=TOL)

    def test_Hbar_is_midpoint_of_H_and_G(self):
        for i in range(3):
            expected = 0.5 * (H_STRATEGY[i] + G_STRATEGY[i])
            assert abs(HBAR_STRATEGY[i] - expected) < TOL

    def test_H3_derived_from_H1_H2(self):
        """H_3 = -(H_1 + H_2)."""
        assert abs(H_STRATEGY[2] + H_STRATEGY[0] + H_STRATEGY[1]) < TOL

    def test_G3_derived_from_G1_G2(self):
        assert abs(G_STRATEGY[2] + G_STRATEGY[0] + G_STRATEGY[1]) < TOL

    def test_Hbar3_derived_from_Hbar1_Hbar2(self):
        assert abs(HBAR_STRATEGY[2] + HBAR_STRATEGY[0] + HBAR_STRATEGY[1]) < TOL

    def test_non_liquidating_strategy_fails(self):
        bad = [1.0, 1.0, 1.0]  # does not sum to 0
        assert not verify_liquidation(bad, tol=TOL)


# ---------------------------------------------------------------------------
# Recursive vs explicit cash tests
# ---------------------------------------------------------------------------

class TestCashFormulas:
    @pytest.mark.parametrize("strategy,name", [
        (H_STRATEGY, "H"), (G_STRATEGY, "G"), (HBAR_STRATEGY, "Hbar")
    ])
    def test_recursive_equals_explicit_zero_prices(self, strategy, name):
        """With zero prices, only impact cost contributes; both formulas agree."""
        prices = [0.0, 0.0, 0.0]
        xi_r, xi_e, ok = verify_recursive_vs_explicit(
            strategy, prices, R, DELTA, xi_0=0.0, zeta_0=0.0, tol=1e-10
        )
        assert ok, f"Strategy {name}: recursive={xi_r}, explicit={xi_e}"

    @pytest.mark.parametrize("strategy", [H_STRATEGY, G_STRATEGY, HBAR_STRATEGY])
    def test_recursive_equals_explicit_nonzero_prices(self, strategy):
        prices = [1.0, -0.5, 0.3]
        xi_r, xi_e, ok = verify_recursive_vs_explicit(
            strategy, prices, R, DELTA, xi_0=0.0, zeta_0=0.0, tol=1e-10
        )
        assert ok, f"recursive={xi_r}, explicit={xi_e}"

    def test_impact_cost_all_nonnegative(self):
        """Impact cost is always non-negative for any strategy."""
        for strategy in [H_STRATEGY, G_STRATEGY, HBAR_STRATEGY]:
            assert impact_cost(strategy) >= 0.0

    def test_zero_strategy_zero_impact_cost(self):
        assert impact_cost([0.0, 0.0, 0.0]) < TOL


# ---------------------------------------------------------------------------
# Quadratic form exact values
# ---------------------------------------------------------------------------

class TestQuadraticForm:
    def test_Q_H_numeric(self):
        """Q(H) = Q(1,1) = 4.7 exactly."""
        val = quadratic_form_numeric(1.0, 1.0)
        assert abs(val - 4.7) < TOL, f"Q(H) = {val}"

    def test_Q_G_numeric(self):
        """Q(G) = Q(1.5, 0) = 4.725 exactly."""
        val = quadratic_form_numeric(1.5, 0.0)
        assert abs(val - 4.725) < TOL, f"Q(G) = {val}"

    def test_Q_Hbar_numeric(self):
        """Q(H̄) = Q(1.25, 0.5) = 4.79375 exactly."""
        val = quadratic_form_numeric(1.25, 0.5)
        assert abs(val - 4.79375) < TOL, f"Q(H̄) = {val}"

    def test_midpoint_average(self):
        """½(Q(H) + Q(G)) = 4.7125."""
        avg = 0.5 * (quadratic_form_numeric(1.0, 1.0) + quadratic_form_numeric(1.5, 0.0))
        assert abs(avg - 4.7125) < TOL, f"midpoint avg = {avg}"

    def test_contradiction_gap(self):
        """Contradiction: midpoint_avg - Q(H̄) = -0.08125 < 0."""
        avg = 0.5 * (quadratic_form_numeric(1.0, 1.0) + quadratic_form_numeric(1.5, 0.0))
        q_hbar = quadratic_form_numeric(1.25, 0.5)
        gap = avg - q_hbar
        assert abs(gap - (-0.08125)) < TOL, f"gap = {gap}"
        assert gap < 0, "Gap must be negative for non-convexity"

    def test_Q_H_symbolic(self):
        """Exact rational arithmetic: Q(H) = 47/10."""
        q = quadratic_form_symbolic(sp.Rational(1), sp.Rational(1))
        assert q == sp.Rational(47, 10), f"Q(H) sym = {q}"

    def test_Q_G_symbolic(self):
        """Exact rational arithmetic: Q(G) = 189/40."""
        q = quadratic_form_symbolic(sp.Rational(3, 2), sp.Rational(0))
        assert q == sp.Rational(189, 40), f"Q(G) sym = {q}"
        assert abs(float(q) - 4.725) < TOL

    def test_Q_Hbar_symbolic(self):
        """Exact rational arithmetic: Q(H̄) = 4.79375."""
        q = quadratic_form_symbolic(sp.Rational(5, 4), sp.Rational(1, 2))
        assert abs(float(q) - 4.79375) < 1e-10, f"Q(H̄) sym = {q} = {float(q)}"

    def test_Q_zero_at_origin(self):
        assert abs(quadratic_form_numeric(0.0, 0.0)) < TOL

    def test_Q_positive_definite_sample(self):
        """Q should be positive for positive arguments."""
        for h1, h2 in [(0.5, 0.5), (1.0, 0.0), (0.0, 1.0), (2.0, 1.0)]:
            assert quadratic_form_numeric(h1, h2) > 0


# ---------------------------------------------------------------------------
# Symbolic derivation test
# ---------------------------------------------------------------------------

class TestSymbolicDerivation:
    def test_derive_Q_matches_explicit_at_H(self):
        """Derived symbolic Q, evaluated at h1=1, h2=1, matches 4.7."""
        Q_expr = derive_quadratic_form()
        h1, h2 = sp.symbols("h1 h2", positive=True)
        val_at_H = Q_expr.subs([(h1, 1), (h2, 1)])
        assert abs(float(val_at_H) - 4.7) < 1e-8, f"Derived Q at H = {float(val_at_H)}"

    def test_derive_Q_matches_explicit_at_G(self):
        Q_expr = derive_quadratic_form()
        h1, h2 = sp.symbols("h1 h2", positive=True)
        val_at_G = Q_expr.subs([(h1, sp.Rational(3, 2)), (h2, 0)])
        assert abs(float(val_at_G) - 4.725) < 1e-8


# ---------------------------------------------------------------------------
# Full run test
# ---------------------------------------------------------------------------

class TestFullRun:
    def test_run_produces_correct_values(self):
        result = run_example_27()
        assert abs(result.Q_H - 4.7) < TOL
        assert abs(result.Q_G - 4.725) < TOL
        assert abs(result.Q_Hbar - 4.79375) < TOL
        assert abs(result.midpoint_average - 4.7125) < TOL
        assert abs(result.contradiction_gap - (-0.08125)) < TOL
        assert result.is_non_convex is True

    def test_run_liquidation_all_pass(self):
        result = run_example_27()
        assert all(result.liquidation_checks.values())

    def test_run_recursive_checks_all_pass(self):
        result = run_example_27()
        assert all(result.recursive_checks.values())

    def test_run_symbolic_exact(self):
        result = run_example_27()
        assert result.Q_H_sym == sp.Rational(47, 10)
        assert abs(float(result.Q_G_sym) - 4.725) < 1e-10
        assert abs(float(result.Q_Hbar_sym) - 4.79375) < 1e-10


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_step_strategy(self):
        """Single step with trivial liquidation: H=[1,-1]. T=2."""
        H = [1.0, -1.0]
        P = [0.5, -0.5]
        r = [0.0, 0.0]
        delta = [2.0, 2.0]
        assert verify_liquidation(H)
        xi_r, xi_e, ok = verify_recursive_vs_explicit(H, P, r, delta, 0.0, 0.0, tol=1e-10)
        assert ok

    def test_large_trade_increments(self):
        """Large impact costs are still non-negative."""
        H = [100.0, 100.0, -200.0]
        ic = impact_cost(H)
        assert ic >= 0.0

    def test_zero_depth_raises_division(self):
        """Zero depth causes division by zero — should propagate as inf."""
        H = [1.0, 0.0, -1.0]
        delta_bad = [0.0, 10.0, 10.0]
        with pytest.raises((ZeroDivisionError, ValueError, FloatingPointError)):
            # This should either raise or produce inf; either is acceptable
            ic = impact_cost(H, delta_bad)
            if not np.isfinite(ic):
                raise ValueError("inf impact cost")
