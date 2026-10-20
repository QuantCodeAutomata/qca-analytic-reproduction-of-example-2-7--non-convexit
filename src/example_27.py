"""
Example 2.7 — Non-convexity of the attainable superhedgeable payoff set S.

Paper claim: Without the monotonicity condition on ρ_{0,t}^2 δ_t, the set
  S = {Y ∈ L^0 : ∃ X ∈ A_0 s.t. ξ_T^X ≥ Y a.s.}
need not be convex.

This module reproduces the witness construction exactly:
  - T = 3, r_t = 0, δ = (1, 10, 10), P_0 = ζ_0 = B = z = 0
  - H = (1, 1, -2)   — witness strategy 1
  - G = (1.5, 0, -1.5) — witness strategy 2
  - H̄ = (H+G)/2 = (1.25, 0.5, -1.75) — midpoint candidate

The reduced quadratic form governing the superhedging comparison is:
  Q(h_1, h_2) = (21/10) h_1^2 + (23/10) h_1 h_2 + (3/10) h_2^2

Custom — Context7 found no library equivalent for this algebraic witness construction.
Uses sympy for exact rational arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, Tuple

import numpy as np

# Using sympy for exact rational arithmetic — Context7 confirmed (/sympy/sympy)
import sympy as sp

from src.model import (
    build_rho_table,
    terminal_cash_explicit,
    verify_liquidation,
    verify_recursive_vs_explicit,
)


# ---------------------------------------------------------------------------
# Model parameters (Example 2.7)
# ---------------------------------------------------------------------------

T = 3
R = [0.0, 0.0, 0.0]        # resilience rates: r_0, r_1, r_2
DELTA = [1.0, 10.0, 10.0]  # market depths: δ_1, δ_2, δ_3

# Witness strategies (increments H_1, H_2, H_3)
H_STRATEGY = [1.0, 1.0, -2.0]
G_STRATEGY = [1.5, 0.0, -1.5]
HBAR_STRATEGY = [1.25, 0.5, -1.75]  # midpoint = (H + G) / 2

# Sympy rational versions
DELTA_SYM = [sp.Rational(1), sp.Rational(10), sp.Rational(10)]
H_SYM = [sp.Rational(1), sp.Rational(1), sp.Rational(-2)]
G_SYM = [sp.Rational(3, 2), sp.Rational(0), sp.Rational(-3, 2)]
HBAR_SYM = [sp.Rational(5, 4), sp.Rational(1, 2), sp.Rational(-7, 4)]


# ---------------------------------------------------------------------------
# Impact cost computation
# ---------------------------------------------------------------------------

def impact_cost(H: list, delta: list = DELTA) -> float:
    """Compute the total market impact cost for a liquidating strategy.

    Impact = ∑_{t=1}^T |H_t| ∑_{j=1}^t |H_j| / δ_j

    With r_t = 0, ρ_{j,t} = 1 for all j ≤ t, so resilience is trivial.

    Parameters
    ----------
    H     : trade increments H_1, ..., H_T
    delta : market depths δ_1, ..., δ_T

    Returns
    -------
    float : total impact cost
    """
    T_loc = len(H)
    cost = 0.0
    for t in range(T_loc):
        inner = sum(abs(H[j]) / delta[j] for j in range(t + 1))
        cost += abs(H[t]) * inner
    return cost


def impact_cost_symbolic(H: list, delta: list = None) -> sp.Expr:
    """Exact symbolic impact cost using sympy rationals.

    Parameters
    ----------
    H     : trade increments as sympy Rationals
    delta : market depths as sympy Rationals

    Returns
    -------
    sympy expression : exact impact cost
    """
    if delta is None:
        delta = DELTA_SYM
    T_loc = len(H)
    cost = sp.Integer(0)
    for t in range(T_loc):
        inner = sum(sp.Abs(H[j]) / delta[j] for j in range(t + 1))
        cost += sp.Abs(H[t]) * inner
    return sp.simplify(cost)


# ---------------------------------------------------------------------------
# Quadratic form Q(h_1, h_2)
# ---------------------------------------------------------------------------

def quadratic_form_numeric(h1: float, h2: float) -> float:
    """Evaluate the reduced quadratic form Q(h_1, h_2).

    Q(h_1, h_2) = (21/10) h_1^2 + (23/10) h_1 h_2 + (3/10) h_2^2

    This is the paper-reported form governing the superhedging comparison
    when H_3 = -(h_1 + h_2) and all entries are non-negative in absolute value.

    Custom — Context7 found no library equivalent (paper Section 2, Example 2.7).

    Parameters
    ----------
    h1, h2 : first and second trade increments

    Returns
    -------
    float : Q(h_1, h_2)
    """
    return (21 / 10) * h1**2 + (23 / 10) * h1 * h2 + (3 / 10) * h2**2


def quadratic_form_symbolic(h1: sp.Rational, h2: sp.Rational) -> sp.Rational:
    """Exact symbolic evaluation of Q(h_1, h_2).

    Parameters
    ----------
    h1, h2 : sympy Rational values

    Returns
    -------
    sympy Rational : exact Q value
    """
    return (sp.Rational(21, 10) * h1**2
            + sp.Rational(23, 10) * h1 * h2
            + sp.Rational(3, 10) * h2**2)


def derive_quadratic_form() -> sp.Expr:
    """Symbolically derive Q(h_1, h_2) from the impact cost formula.

    Substitutes H_3 = -(H_1 + H_2) and expands assuming H_1, H_2 > 0.

    Returns
    -------
    sympy Expr : symbolic Q(h_1, h_2)
    """
    h1, h2 = sp.symbols("h1 h2", positive=True)
    h3 = -(h1 + h2)

    # Build impact cost symbolically
    H = [h1, h2, h3]
    delta = DELTA_SYM

    cost = sp.Integer(0)
    for t in range(3):
        inner = sum(sp.Abs(H[j]) / delta[j] for j in range(t + 1))
        cost += sp.Abs(H[t]) * inner

    # Since h1, h2 > 0, |h3| = h1 + h2
    cost_simplified = sp.simplify(cost.rewrite(sp.Piecewise))
    # Manual substitution for clarity:
    cost_manual = (h1**2 / 1
                   + h2 * (h1 / 1 + h2 / 10)
                   + (h1 + h2) * (h1 / 1 + h2 / 10 + (h1 + h2) / 10))
    Q = sp.expand(cost_manual)
    return Q


# ---------------------------------------------------------------------------
# Main Example 2.7 computation
# ---------------------------------------------------------------------------

@dataclass
class Example27Result:
    """Container for all Example 2.7 outputs."""
    parameters: Dict
    strategies: Dict
    impact_costs_numeric: Dict[str, float]
    impact_costs_symbolic: Dict[str, sp.Rational]
    Q_H: float
    Q_G: float
    Q_Hbar: float
    midpoint_average: float
    contradiction_gap: float
    is_non_convex: bool
    Q_H_sym: sp.Rational = field(default=None)
    Q_G_sym: sp.Rational = field(default=None)
    Q_Hbar_sym: sp.Rational = field(default=None)
    liquidation_checks: Dict[str, bool] = field(default_factory=dict)
    recursive_checks: Dict[str, bool] = field(default_factory=dict)


def run_example_27(
    prices: list = None,
    tol: float = 1e-12,
) -> Example27Result:
    """Reproduce Example 2.7 exactly.

    Parameters
    ----------
    prices : price realization P_1, P_2, P_3 for recursive cross-check
             (default: [0, 0, 0] since Gaussian is used analytically)
    tol    : numerical tolerance for equality checks

    Returns
    -------
    Example27Result
    """
    if prices is None:
        prices = [0.0, 0.0, 0.0]  # deterministic cross-check prices

    # --- Parameter manifest ---
    params = {
        "T": T,
        "r": R,
        "delta": DELTA,
        "zeta_0": 0.0,
        "xi_0": 0.0,
        "B": 0.0,
        "model": "transient price impact, r_t=0 => rho_{j,t}=1",
    }
    strategies = {
        "H": H_STRATEGY,
        "G": G_STRATEGY,
        "Hbar": HBAR_STRATEGY,
    }

    # --- Liquidation checks ---
    liq = {
        "H": verify_liquidation(H_STRATEGY),
        "G": verify_liquidation(G_STRATEGY),
        "Hbar": verify_liquidation(HBAR_STRATEGY),
    }

    # --- Recursive vs explicit cross-check (P=[0,0,0], impact terms only) ---
    rec_checks = {}
    for name, strat in strategies.items():
        xi_r, xi_e, ok = verify_recursive_vs_explicit(
            strat, prices, R, DELTA, xi_0=0.0, zeta_0=0.0, tol=tol
        )
        rec_checks[name] = ok

    # --- Numeric impact costs (= Q values since P_t=0 analytic meaning) ---
    ic_num = {name: impact_cost(strat) for name, strat in strategies.items()}

    # --- Symbolic exact evaluation ---
    Q_H_sym = quadratic_form_symbolic(sp.Rational(1), sp.Rational(1))
    Q_G_sym = quadratic_form_symbolic(sp.Rational(3, 2), sp.Rational(0))
    Q_Hbar_sym = quadratic_form_symbolic(sp.Rational(5, 4), sp.Rational(1, 2))

    ic_sym = {
        "H": impact_cost_symbolic(H_SYM),
        "G": impact_cost_symbolic(G_SYM),
        "Hbar": impact_cost_symbolic(HBAR_SYM),
    }

    # --- Numeric quadratic form evaluations ---
    Q_H = quadratic_form_numeric(1.0, 1.0)
    Q_G = quadratic_form_numeric(1.5, 0.0)
    Q_Hbar = quadratic_form_numeric(1.25, 0.5)

    midpoint_average = 0.5 * (Q_H + Q_G)
    contradiction_gap = midpoint_average - Q_Hbar

    # Non-convexity: midpoint average < Q_Hbar means midpoint payoff less
    # attainable (higher impact cost = worse superhedging position)
    is_non_convex = contradiction_gap < 0

    return Example27Result(
        parameters=params,
        strategies=strategies,
        impact_costs_numeric=ic_num,
        impact_costs_symbolic=ic_sym,
        Q_H=Q_H,
        Q_G=Q_G,
        Q_Hbar=Q_Hbar,
        midpoint_average=midpoint_average,
        contradiction_gap=contradiction_gap,
        is_non_convex=is_non_convex,
        Q_H_sym=Q_H_sym,
        Q_G_sym=Q_G_sym,
        Q_Hbar_sym=Q_Hbar_sym,
        liquidation_checks=liq,
        recursive_checks=rec_checks,
    )


def print_report(result: Example27Result) -> None:
    """Print a human-readable report of Example 2.7 results."""
    print("=" * 70)
    print("EXAMPLE 2.7 — Non-convexity of Attainable Set S")
    print("=" * 70)
    print("\nParameter Manifest:")
    for k, v in result.parameters.items():
        print(f"  {k} = {v}")

    print("\nWitness Strategies:")
    for name, strat in result.strategies.items():
        print(f"  {name} = {strat}")

    print("\nLiquidation Checks (∑ H_t = 0):")
    for name, ok in result.liquidation_checks.items():
        print(f"  {name}: {ok}")

    print("\nRecursive == Explicit Cash Checks (P=[0,0,0]):")
    for name, ok in result.recursive_checks.items():
        print(f"  {name}: {ok}")

    print("\nImpact Costs (numeric):")
    for name, ic in result.impact_costs_numeric.items():
        print(f"  {name}: {ic}")

    print("\nImpact Costs (symbolic, exact):")
    for name, ic in result.impact_costs_symbolic.items():
        print(f"  {name}: {ic} = {float(ic):.8f}")

    print("\nQuadratic Form Q(h1, h2) = (21/10)h1² + (23/10)h1h2 + (3/10)h2²")
    print(f"  Q(H)  = Q(1,   1)    = {result.Q_H}  (expected 4.7)")
    print(f"  Q(G)  = Q(1.5, 0)    = {result.Q_G}  (expected 4.725)")
    print(f"  Q(H̄) = Q(1.25, 0.5) = {result.Q_Hbar}  (expected 4.79375)")

    print("\nSymbolic exact values:")
    print(f"  Q(H)  = {result.Q_H_sym}  = {float(result.Q_H_sym)}")
    print(f"  Q(G)  = {result.Q_G_sym}  = {float(result.Q_G_sym)}")
    print(f"  Q(H̄) = {result.Q_Hbar_sym} = {float(result.Q_Hbar_sym)}")

    print(f"\nMidpoint average  ½(Q(H) + Q(G)) = {result.midpoint_average}  (expected 4.7125)")
    print(f"Q(H̄) - midpoint avg             = {result.contradiction_gap}  (expected +0.08125)")
    print(f"Contradiction gap (avg - Q(H̄)) = {result.contradiction_gap}  (expected -0.08125)")

    print(f"\nConclusion: S is non-convex = {result.is_non_convex}")
    if result.is_non_convex:
        print("  The midpoint strategy's impact cost exceeds the average of H and G,")
        print("  meaning the midpoint payoff is strictly dominated in the superhedging")
        print("  sense — confirming non-convexity of S when monotonicity is dropped.")
    print("=" * 70)
