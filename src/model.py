"""
Core transient price impact model mechanics.

Implements the discrete-time transient price impact model following the paper's
notation exactly. Key quantities:

  - Resilience accumulation factors  ρ_{j,t} = exp(-∑_{i=j}^{t-1} r_i)
  - Half-spread recursion            ζ_{t+1} = e^{-r_t} ζ_t + |H_{t+1}| / δ_{t+1}
  - Cash recursion                   ξ_{t+1} - ξ_t = -P_{t+1} H_{t+1} - ζ_{t+1} |H_{t+1}|
  - Explicit terminal cash           ξ_T = -∑_t P_t H_t - ∑_t |H_t|(ρ_{0,t} ζ_0 + ∑_j ρ_{j,t} |H_j| / δ_j)
  - Innovation function              κ_t(h) = -P_t h_t - |h_t|(ρ_{0,t} ζ_0 + ∑_j ρ_{j,t} |h_j| / δ_j)
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Resilience accumulation factors
# ---------------------------------------------------------------------------

def resilience_factor(r: Sequence[float], j: int, t: int) -> float:
    """Compute ρ_{j,t} = exp(-∑_{i=j}^{t-1} r_i).

    Parameters
    ----------
    r : resilience rates r_0, r_1, ..., r_{T-1} (non-negative)
    j : start index (inclusive)
    t : end index (exclusive, i.e. sum from j to t-1)

    Returns
    -------
    float : ρ_{j,t}

    Notes
    -----
    ρ_{j,j} = 1  (empty sum)
    """
    if j >= t:
        return 1.0
    return math.exp(-sum(r[i] for i in range(j, t)))


def build_rho_table(r: Sequence[float], T: int) -> np.ndarray:
    """Build full table of resilience factors ρ_{j,t} for 0 <= j <= t <= T.

    Parameters
    ----------
    r : resilience rates (length T, r[t] applies from t to t+1)
    T : horizon

    Returns
    -------
    rho : (T+1, T+1) array where rho[j, t] = ρ_{j,t}
    """
    rho = np.ones((T + 1, T + 1))
    for j in range(T + 1):
        for t in range(j + 1, T + 1):
            rho[j, t] = resilience_factor(r, j, t)
    return rho


# ---------------------------------------------------------------------------
# Recursive spread and cash dynamics
# ---------------------------------------------------------------------------

def spread_recursion(
    zeta_t: float, r_t: float, H_next: float, delta_next: float
) -> float:
    """Compute half-spread at t+1 given half-spread at t and trade increment.

    ζ_{t+1} = e^{-r_t} ζ_t + |H_{t+1}| / δ_{t+1}

    Parameters
    ----------
    zeta_t    : half-spread at time t
    r_t       : resilience rate from t to t+1
    H_next    : trade increment H_{t+1}
    delta_next: market depth δ_{t+1}

    Returns
    -------
    float : ζ_{t+1}
    """
    return math.exp(-r_t) * zeta_t + abs(H_next) / delta_next


def cash_recursion(
    xi_t: float, P_next: float, H_next: float, zeta_next: float
) -> float:
    """Compute cash at t+1 given cash at t.

    ξ_{t+1} = ξ_t - P_{t+1} H_{t+1} - ζ_{t+1} |H_{t+1}|

    Parameters
    ----------
    xi_t     : cash position at time t
    P_next   : price at t+1
    H_next   : trade increment H_{t+1}
    zeta_next: half-spread at t+1 (post-trade)

    Returns
    -------
    float : ξ_{t+1}
    """
    return xi_t - P_next * H_next - zeta_next * abs(H_next)


def simulate_recursive(
    H: Sequence[float],
    P: Sequence[float],
    r: Sequence[float],
    delta: Sequence[float],
    xi_0: float = 0.0,
    zeta_0: float = 0.0,
) -> Tuple[List[float], List[float]]:
    """Simulate spread and cash recursively for a given strategy.

    Parameters
    ----------
    H     : trade increments H_1, ..., H_T (length T); 1-indexed
    P     : prices P_1, ..., P_T  (length T); 1-indexed
    r     : resilience rates r_0, ..., r_{T-1} (length T)
    delta : market depths δ_1, ..., δ_T (length T); 1-indexed
    xi_0  : initial cash
    zeta_0: initial half-spread

    Returns
    -------
    (zeta_path, xi_path) : both of length T+1, index 0 is initial value
    """
    T = len(H)
    assert len(P) == T and len(r) >= T and len(delta) == T

    zeta_path = [zeta_0]
    xi_path = [xi_0]

    zeta_t = zeta_0
    xi_t = xi_0

    for t in range(T):  # t = 0, ..., T-1 corresponds to step t+1
        zeta_next = spread_recursion(zeta_t, r[t], H[t], delta[t])
        xi_next = cash_recursion(xi_t, P[t], H[t], zeta_next)
        zeta_path.append(zeta_next)
        xi_path.append(xi_next)
        zeta_t = zeta_next
        xi_t = xi_next

    return zeta_path, xi_path


# ---------------------------------------------------------------------------
# Explicit terminal cash formula
# ---------------------------------------------------------------------------

def terminal_cash_explicit(
    H: Sequence[float],
    P: Sequence[float],
    r: Sequence[float],
    delta: Sequence[float],
    zeta_0: float = 0.0,
    xi_0: float = 0.0,
) -> float:
    """Compute terminal cash via the explicit closed-form formula.

    ξ_T = xi_0 - ∑_{t=1}^T P_t H_t
               - ∑_{t=1}^T |H_t|(ρ_{0,t} ζ_0 + ∑_{j=1}^t ρ_{j,t} |H_j| / δ_j)

    Parameters
    ----------
    H     : trade increments H_1, ..., H_T (length T, 1-indexed internally)
    P     : prices P_1, ..., P_T  (length T)
    r     : resilience rates r_0, ..., r_{T-1} (length T; r[i] is r_i)
    delta : market depths δ_1, ..., δ_T (length T)
    zeta_0: initial half-spread
    xi_0  : initial cash

    Returns
    -------
    float : terminal cash ξ_T
    """
    T = len(H)
    assert len(P) == T and len(delta) == T and len(r) >= T

    rho = build_rho_table(r, T)

    price_term = sum(P[t] * H[t] for t in range(T))

    impact_term = 0.0
    for t in range(T):  # t is 0-indexed, corresponds to paper's t+1
        # ρ_{0, t+1} ζ_0
        impact = rho[0, t + 1] * zeta_0
        # ∑_{j=1}^{t+1} ρ_{j, t+1} |H_j| / δ_j
        for j in range(t + 1):  # j 0-indexed, corresponds to paper's j+1
            impact += rho[j + 1, t + 1] * abs(H[j]) / delta[j]
        impact_term += abs(H[t]) * impact

    return xi_0 - price_term - impact_term


# ---------------------------------------------------------------------------
# Innovation function κ_t
# ---------------------------------------------------------------------------

def innovation(
    t: int,
    h_t: float,
    H_prev: Sequence[float],
    P_t: float,
    r: Sequence[float],
    delta: Sequence[float],
    zeta_0: float = 0.0,
) -> float:
    """Compute the incremental cash contribution κ_t of a single trade.

    κ_t = -P_t h_t - |h_t|(ρ_{0,t} ζ_0 + ∑_{j=1}^t ρ_{j,t} |h_j| / δ_j)

    Parameters
    ----------
    t       : time step (1-indexed, 1..T)
    h_t     : trade increment at t
    H_prev  : previous trade increments H_1, ..., H_{t-1} (length t-1)
    P_t     : price at t
    r       : resilience rates (0-indexed; r[i] = r_i)
    delta   : market depths (0-indexed; delta[i] = δ_{i+1})
    zeta_0  : initial half-spread

    Returns
    -------
    float : κ_t
    """
    rho = build_rho_table(r, t)

    impact = rho[0, t] * zeta_0
    for j in range(t - 1):  # j 0-indexed corresponds to j+1 in paper
        impact += rho[j + 1, t] * abs(H_prev[j]) / delta[j]
    # Add current trade (j = t in 0-indexed, corresponds to t+1 in 1-indexed but same t)
    impact += rho[t, t] * abs(h_t) / delta[t - 1]  # ρ_{t,t}=1, delta[t-1]=δ_t

    return -P_t * h_t - abs(h_t) * impact


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def utility_piecewise_logistic(
    x: float | np.ndarray, threshold: float = 5.0
) -> float | np.ndarray:
    """Piecewise utility from Example 2.10.

    u(x) = x                                         for x <= 0
    u(x) = [σ(x-5) - σ(-5)] / [1 - σ(-5)]          for x > 0

    where σ(x) = 1 / (1 + exp(-x)).

    Parameters
    ----------
    x         : wealth value(s)
    threshold : logistic shift parameter (default 5)

    Returns
    -------
    float or ndarray : utility value(s)
    """
    from scipy.special import expit  # stable sigmoid  # Using scipy.special.expit — Context7 confirmed

    sigma_neg5 = expit(-threshold)
    normalizer = 1.0 - sigma_neg5

    scalar = np.isscalar(x)
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    result = np.where(
        x_arr <= 0,
        x_arr,
        (expit(x_arr - threshold) - sigma_neg5) / normalizer,
    )
    return float(result[0]) if scalar else result


def utility_logistic_bounded(
    x: float | np.ndarray, a: float = 1.0, b: float = 0.0
) -> float | np.ndarray:
    """Bounded-above concave logistic utility: u(x) = σ(ax + b).

    Satisfies: non-decreasing, bounded above by 1, continuous.

    Parameters
    ----------
    x : wealth value(s)
    a : slope (positive)
    b : intercept

    Returns
    -------
    float or ndarray : utility value(s)
    """
    from scipy.special import expit  # Using scipy.special.expit — Context7 confirmed

    return expit(a * x + b)


def utility_linear_bounded(
    x: float | np.ndarray, cap: float = 1.0
) -> float | np.ndarray:
    """Bounded piecewise-linear utility: u(x) = min(x, cap) for x >= 0, else x.

    Parameters
    ----------
    x   : wealth value(s)
    cap : upper bound for x > 0

    Returns
    -------
    float or ndarray : utility value(s)
    """
    scalar = np.isscalar(x)
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    result = np.where(x_arr > 0, np.minimum(x_arr, cap), x_arr)
    return float(result[0]) if scalar else result


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def verify_liquidation(H: Sequence[float], tol: float = 1e-12) -> bool:
    """Check that the strategy is liquidating: ∑ H_t = 0."""
    return abs(sum(H)) < tol


def verify_recursive_vs_explicit(
    H: Sequence[float],
    P: Sequence[float],
    r: Sequence[float],
    delta: Sequence[float],
    xi_0: float = 0.0,
    zeta_0: float = 0.0,
    tol: float = 1e-12,
) -> Tuple[float, float, bool]:
    """Compare recursive terminal cash with explicit formula.

    Returns
    -------
    (xi_recursive, xi_explicit, matches)
    """
    _, xi_path = simulate_recursive(H, P, r, delta, xi_0, zeta_0)
    xi_recursive = xi_path[-1]
    xi_explicit = terminal_cash_explicit(H, P, r, delta, zeta_0, xi_0)
    matches = abs(xi_recursive - xi_explicit) < tol
    return xi_recursive, xi_explicit, matches
