"""
Example 2.9 — Non-concavity of indirect utility on a non-convex feasible set.

Paper setup (one-step frictionless model):
  - Feasible strategies: Φ = {0, 1}          (non-convex discrete set)
  - Utility: u(x) = ln(x)                    (concave)
  - Return R: P(R=2)=1/2,  P(R=-1)=1/2
  - Initial capital z > 1

Indirect utility:
  ū(z) = max{ ln(z),  ½[ln(z+2) + ln(z-1)] }

Analytic results:
  - Crossing at z = 2  (both branches equal ln 2)
  - Left derivative at z=2:  (ln z)' = 1/z = 1/2
  - Right derivative at z=2: ½[(z+2)⁻¹ + (z-1)⁻¹] = ½(1/4 + 1) = 5/8
  - Upward jump in derivative => indirect utility is NOT concave

Custom — Context7 found no library equivalent for this specific example.
Uses sympy for symbolic crossing and derivative computation.
Uses numpy + matplotlib for numerical evaluation and plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

# Using sympy for symbolic math — Context7 confirmed (/sympy/sympy)
import sympy as sp


# ---------------------------------------------------------------------------
# Branch functions
# ---------------------------------------------------------------------------

def branch_safe(z: float | np.ndarray) -> float | np.ndarray:
    """Expected utility for strategy φ=0: E[u(z)] = ln(z).

    Parameters
    ----------
    z : initial capital, must satisfy z > 0

    Returns
    -------
    float or ndarray : ln(z)
    """
    return np.log(z)


def branch_risky(z: float | np.ndarray) -> float | np.ndarray:
    """Expected utility for strategy φ=1: ½ ln(z+2) + ½ ln(z-1).

    Valid for z > 1 (so that z-1 > 0).

    Parameters
    ----------
    z : initial capital, must satisfy z > 1

    Returns
    -------
    float or ndarray : ½ ln(z+2) + ½ ln(z-1)
    """
    return 0.5 * np.log(z + 2) + 0.5 * np.log(z - 1)


def indirect_utility(z: float | np.ndarray) -> float | np.ndarray:
    """Indirect utility function ū(z) = max{ln(z), ½[ln(z+2) + ln(z-1)]}.

    Parameters
    ----------
    z : initial capital, must satisfy z > 1

    Returns
    -------
    float or ndarray : ū(z)
    """
    return np.maximum(branch_safe(z), branch_risky(z))


# ---------------------------------------------------------------------------
# Symbolic analysis
# ---------------------------------------------------------------------------

def solve_crossing_symbolic() -> Dict:
    """Solve analytically for the crossing point of the two branches.

    Algebraic reduction (paper's exact derivation):
      ln(z) = ½(ln(z+2) + ln(z-1))
      => ln(z²) = ln((z+2)(z-1))      [exponentiate both sides × 2]
      => z² = (z+2)(z-1) = z² + z - 2
      => 0 = z - 2  => z = 2

    SymPy handles the polynomial step; the log equation itself has multiple
    generators that SymPy's solver cannot directly handle, so we do the
    algebraic pre-simplification symbolically and let SymPy solve the
    residual polynomial.

    Returns
    -------
    dict with crossing_z, crossing_value, left_deriv, right_deriv
    """
    z = sp.Symbol("z", positive=True)

    # Branch functions symbolically
    f_safe = sp.ln(z)
    f_risky = sp.Rational(1, 2) * (sp.ln(z + 2) + sp.ln(z - 1))

    # Algebraic reduction: ln(z) = ½(ln(z+2)+ln(z-1))
    #   => z² = (z+2)(z-1)  => z² - (z+2)(z-1) = 0  => z - 2 = 0
    poly_eq = z**2 - (z + 2) * (z - 1)   # = z² - z² - z + 2 = -z + 2
    solutions = sp.solve(poly_eq, z)
    # Keep real positive solutions > 1
    crossing_solutions = [s for s in solutions if s.is_real and s > 1]
    crossing_z = crossing_solutions[0] if crossing_solutions else sp.Integer(2)

    # Verify symbolically that the polynomial reduction is consistent with logs
    # (ln(2) == ½(ln(4) + ln(1)) = ½ ln 4 = ln 2  ✓)
    assert sp.simplify(f_safe.subs(z, crossing_z) - f_risky.subs(z, crossing_z)) == 0, \
        "Crossing verification failed at z=2"

    crossing_value = f_safe.subs(z, crossing_z)

    # Derivatives at crossing
    df_safe = sp.diff(f_safe, z)
    df_risky = sp.diff(f_risky, z)

    left_deriv = sp.simplify(df_safe.subs(z, crossing_z))
    right_deriv = sp.simplify(df_risky.subs(z, crossing_z))

    return {
        "crossing_z": crossing_z,
        "crossing_value": crossing_value,
        "crossing_value_float": float(crossing_value),
        "left_derivative": left_deriv,
        "left_derivative_float": float(left_deriv),
        "right_derivative": right_deriv,
        "right_derivative_float": float(right_deriv),
        "derivative_jumps_up": bool(right_deriv > left_deriv),
        "non_concave": bool(right_deriv > left_deriv),
    }


def finite_difference_second_derivative(
    z_grid: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Compute finite-difference second differences as a shape diagnostic.

    A concave function has non-positive second differences everywhere.
    A positive second difference at any point signals non-concavity.

    Parameters
    ----------
    z_grid : z values (uniformly spaced)
    values : function values at z_grid

    Returns
    -------
    ndarray : second differences (length len(z_grid)-2)
    """
    return np.diff(values, n=2)


# ---------------------------------------------------------------------------
# Main Example 2.9 computation
# ---------------------------------------------------------------------------

@dataclass
class Example29Result:
    """Container for Example 2.9 outputs."""
    parameters: Dict
    z_grid: np.ndarray
    branch_safe_values: np.ndarray
    branch_risky_values: np.ndarray
    indirect_utility_values: np.ndarray
    symbolic_analysis: Dict
    second_differences: np.ndarray
    max_second_difference: float
    non_concave_flag: bool


def run_example_29(
    z_start: float = 1.01,
    z_end: float = 6.0,
    n_points: int = 500,
) -> Example29Result:
    """Reproduce Example 2.9 exactly.

    Parameters
    ----------
    z_start  : start of z-grid (must be > 1)
    z_end    : end of z-grid
    n_points : number of grid points

    Returns
    -------
    Example29Result
    """
    assert z_start > 1.0, "Domain requires z > 1"

    params = {
        "model": "one-step frictionless",
        "feasible_set": [0, 1],
        "utility": "ln(x)",
        "R_plus": 2,
        "R_minus": -1,
        "prob_plus": 0.5,
        "prob_minus": 0.5,
        "domain": "z > 1",
        "z_start": z_start,
        "z_end": z_end,
        "n_points": n_points,
    }

    z_grid = np.linspace(z_start, z_end, n_points)

    bs = branch_safe(z_grid)
    br = branch_risky(z_grid)
    iu = indirect_utility(z_grid)

    sym = solve_crossing_symbolic()

    second_diffs = finite_difference_second_derivative(z_grid, iu)
    max_sd = float(np.max(second_diffs))
    non_concave_flag = max_sd > 0 or sym["non_concave"]

    return Example29Result(
        parameters=params,
        z_grid=z_grid,
        branch_safe_values=bs,
        branch_risky_values=br,
        indirect_utility_values=iu,
        symbolic_analysis=sym,
        second_differences=second_diffs,
        max_second_difference=max_sd,
        non_concave_flag=non_concave_flag,
    )


def print_report(result: Example29Result) -> None:
    """Print human-readable report of Example 2.9 results."""
    print("=" * 70)
    print("EXAMPLE 2.9 — Non-concavity of Indirect Utility")
    print("=" * 70)
    print("\nParameter Manifest:")
    for k, v in result.parameters.items():
        print(f"  {k} = {v}")

    sym = result.symbolic_analysis
    print("\nSymbolic Analysis:")
    print(f"  Crossing z*  = {sym['crossing_z']}  (expected 2)")
    print(f"  ū(z*) = ln(z*) = {sym['crossing_value']}  ≈ {sym['crossing_value_float']:.6f}")
    print(f"  Left  derivative at z*: d/dz ln(z)|_{{z=2}} = {sym['left_derivative']}  "
          f"≈ {sym['left_derivative_float']:.6f}  (expected 0.5)")
    print(f"  Right derivative at z*: d/dz ½[ln(z+2)+ln(z-1)]|_{{z=2}} = {sym['right_derivative']}  "
          f"≈ {sym['right_derivative_float']:.6f}  (expected 0.625)")
    print(f"  Derivative jumps upward: {sym['derivative_jumps_up']}")

    print(f"\nNumerical Shape Diagnostic:")
    print(f"  Max second-difference of ū: {result.max_second_difference:.6e}")
    print(f"  Positive second-diff detected: {result.max_second_difference > 0}")

    print(f"\nConclusion: ū is non-concave = {result.non_concave_flag}")
    if result.non_concave_flag:
        print("  Even though ln is concave, the indirect utility ū(z) displays")
        print("  an upward derivative jump at the branch-switching point z=2,")
        print("  proving non-concavity induced by the non-convex feasible set Φ={0,1}.")
    print("=" * 70)
