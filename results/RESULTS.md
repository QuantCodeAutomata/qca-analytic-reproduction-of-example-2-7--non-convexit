# Experiment Results

Discrete-time transient price impact model — analytic and dynamic-programming reproductions of Examples 2.7, 2.9, and 2.10 from the paper, plus a non-monotone liquidity stress-test suite.

All results were produced by running `exp/run_exp{1..4}.py` and verified by `python -m pytest tests/ -v` (**134/134 tests passed**).

---

## EXP 1 — Example 2.7: Non-convexity of the Attainable Set

**Objective**: Show that the superhedgeable payoff set 𝒮 can fail convexity when the monotonicity condition on liquidity is dropped.

### Parameters

| Symbol | Value |
|--------|-------|
| T | 3 |
| rₜ | 0 (all t) |
| δ₁, δ₂, δ₃ | 1, 10, 10 |
| ζ₀, P₀, B, z | 0 |
| ρⱼ,ₜ | 1 (all j,t — because rₜ=0) |

### Witness Strategies

| Strategy | H₁ | H₂ | H₃ | Liquidates? |
|----------|-----|-----|-----|-------------|
| H | 1.0 | 1.0 | -2.0 | ✓ |
| G | 1.5 | 0.0 | -1.5 | ✓ |
| H̄ = (H+G)/2 | 1.25 | 0.5 | -1.75 | ✓ |

### Quadratic Form  Q(h₁, h₂) = (21/10)h₁² + (23/10)h₁h₂ + (3/10)h₂²

| Point | Numeric | Exact (sympy) | Expected |
|-------|---------|---------------|----------|
| Q(H) = Q(1, 1) | 4.7 | 47/10 | **4.7** |
| Q(G) = Q(1.5, 0) | 4.725 | 189/40 | **4.725** |
| Q(H̄) = Q(1.25, 0.5) | 4.79375 | 767/160 | **4.79375** |
| ½(Q(H)+Q(G)) | 4.7125 | — | **4.7125** |
| Contradiction gap = avg − Q(H̄) | **−0.08125** | −13/160 | **−0.08125** |

### Conclusion

The midpoint strategy H̄ carries a strictly higher impact cost than the average of H and G (gap = −0.08125 < 0).  
By Gaussian argument, the midpoint payoff is therefore not attainable in the superhedging sense.  
**𝒮 is not convex** — fully consistent with the paper's claim.

Recursive cash formula verified to equal explicit terminal cash formula for all three strategies (all prices zero, symbolic check).

**Artifacts**: `results/exp1_quadratic_form.png`, `results/exp1_results.json`

---

## EXP 2 — Example 2.9: Non-concavity of Indirect Utility

**Objective**: Show that a non-convex feasible strategy set Φ = {0, 1} induces non-concavity of the indirect utility ū even though u(x) = ln x is concave.

### Setup

| Symbol | Value |
|--------|-------|
| Utility u | ln(x) |
| Feasible Φ | {0, 1} |
| Return R | +2 (prob 1/2), −1 (prob 1/2) |
| Domain | z > 1 |
| Evaluation grid | [1.01, 6], 500 points |

### Closed-form Branches

| Branch | Formula |
|--------|---------|
| Safe (φ=0) | ln(z) |
| Risky (φ=1) | ½[ln(z+2) + ln(z−1)], valid for z > 1 |
| ū(z) | max{safe, risky} |

### Symbolic Results

| Quantity | Exact | Expected |
|----------|-------|----------|
| Crossing z* | **2** | 2 |
| ū(z*) | log(2) ≈ 0.6931 | ln 2 |
| Left derivative at z=2 | **1/2** | 1/2 |
| Right derivative at z=2 | **5/8** | 5/8 |
| Derivative jump upward | **True** | True |
| ū non-concave flag | **True** | True |

### Conclusion

The derivative of ū jumps upward from 1/2 to 5/8 at the branch-switching point z = 2.  
An upward derivative jump is incompatible with concavity.  
**ū is not concave**, confirming the paper's example that non-convex feasible sets destroy concavity of the indirect utility.

**Artifacts**: `results/exp2_indirect_utility.png`, `results/exp2_results.json`

---

## EXP 3 — Example 2.10: Finite-State DP under Transient Price Impact

**Objective**: Exact finite-state backward dynamic-programming replication; verify internal consistency and qualitative non-concavity of G̃₀(z).

### Parameters

| Symbol | Value |
|--------|-------|
| T | 4 |
| rₜ | 0.01 (all t) |
| δ₁,δ₂,δ₃,δ₄ | 10, 10, 15, 5 |
| Price shocks ε | +1 (p=0.77) / −1 (p=0.23) |
| Scenario paths | 2⁴ = 16 |
| Action grid H₁,H₂,H₃ | 17 equidistant pts on [−4, 4] |
| H₄ | −H₁ − H₂ − H₃ (liquidation) |
| Utility u(x) | x for x≤0; [σ(x−5)−σ(−5)] / [1−σ(−5)] for x>0 |

### Consistency Checks

| Check | Result |
|-------|--------|
| Forward policy value = DP value | **True** |
| DP value = exhaustive brute-force | **True** |
| Recursive cash = explicit terminal cash (all paths) | **True** |

### Shape of G̃₀(z)

| z | G̃₀(z) | No-trade u(z) | Optimal policy (h₁,h₂,h₃) |
|---|--------|--------------|---------------------------|
| −5.00 | −3.335 | −5.000 | (2.0, 0.0, −0.5) |
| −0.71 | −0.314 | −0.714 | (0.5, 0.0, 0.0) |
| 0.51 | 0.004 | 0.004 | (0.0, 0.0, 0.0) |
| 2.76 | 0.395 | 0.090 | (1.5, 0.0, 0.0) |
| 5.00 | 0.718 | 0.497 | (1.5, 0.0, −0.5) |

Max second-difference of G̃₀: **6.81 × 10⁻²**  
Non-concavity detected: **True**

### Grid-Refinement Sensitivity

A grid refinement study across multiple action-grid sizes confirms that the non-concavity finding is **stable** across grid resolutions.

### Conclusion

The backward DP is fully internally consistent (forward-policy match, brute-force match, accounting check).  
G̃₀(z) displays qualitative non-concavity consistent with the paper's qualitative claim.

**Artifacts**: `results/exp3_dp_value_function.png`, `results/exp3_grid_refinement.png`, `results/exp3_results.json`

---

## EXP 4 — Stress Tests: Non-Monotone Liquidity Environments

**Objective**: Verify that backward DP remains internally consistent and produces feasible liquidating strategies across diverse synthetic instances, even when the ρ₀,ₜ²δₜ monotonicity condition is violated.

### Test Families

| Family | Description | Instances |
|--------|-------------|-----------|
| A | Non-monotone liquidity, various depth profiles | 5 |
| B | Monotone-liquidity control cases | 5 |
| C | Horizon sensitivity T ∈ {2, 3, 4, 5} | 4 |
| D | Utility-shape sensitivity (3 utility families) | 3 |

**Total instances**: 17

### Aggregate Results

| Metric | Value |
|--------|-------|
| DP = brute-force | **17/17** |
| Forward policy = DP | **17/17** |
| Recursive = explicit cash | **17/17** |
| Overall consistency rate | **100%** |
| Non-monotone consistency | **12/12 (100%)** |

### Per-Instance Summary

| ID | Family | T | Mono | DP=BF | Fwd=DP | Acct | Max 2nd-diff | Non-Concave |
|----|--------|---|------|-------|--------|------|-------------|-------------|
| A1 | A | 3 | No | ✓ | ✓ | ✓ | 4.40e-03 | Yes |
| A2 | A | 3 | No | ✓ | ✓ | ✓ | 4.40e-03 | Yes |
| A3 | A | 4 | No | ✓ | ✓ | ✓ | 1.73e-01 | Yes |
| A4 | A | 3 | No | ✓ | ✓ | ✓ | 3.32e-02 | Yes |
| A5 | A | 2 | No | ✓ | ✓ | ✓ | 4.40e-03 | Yes |
| B1 | B | 3 | Yes | ✓ | ✓ | ✓ | 4.40e-03 | Yes |
| B2 | B | 3 | Yes | ✓ | ✓ | ✓ | 3.52e-03 | Yes |
| B3 | B | 4 | Yes | ✓ | ✓ | ✓ | 5.93e-03 | Yes |
| B4 | B | 3 | Yes | ✓ | ✓ | ✓ | 5.55e-16 | No |
| B5 | B | 2 | Yes | ✓ | ✓ | ✓ | 4.40e-03 | Yes |
| C_T2 | C | 2 | No | ✓ | ✓ | ✓ | 4.40e-03 | Yes |
| C_T3 | C | 3 | No | ✓ | ✓ | ✓ | 6.66e-03 | Yes |
| C_T4 | C | 4 | No | ✓ | ✓ | ✓ | 6.90e-03 | Yes |
| C_T5 | C | 5 | No | ✓ | ✓ | ✓ | 9.24e-03 | Yes |
| D_logistic_bounded | D | 3 | No | ✓ | ✓ | ✓ | 4.40e-03 | Yes |
| D_linear_bounded | D | 3 | No | ✓ | ✓ | ✓ | 5.55e-16 | No |
| D_piecewise_logistic | D | 3 | No | ✓ | ✓ | ✓ | 2.67e-02 | Yes |

### Conclusion

Backward DP produces feasible optimizers and remains internally consistent in **all** non-monotone liquidity environments tested.  
This confirms the paper's numerical claim that optimizer existence does not require the monotonicity condition on ρ₀,ₜ²δₜ.  
Shape diagnostics show that non-concavity of continuation values is common but not universal; its presence depends on utility shape and liquidity profile.

**Artifacts**: `results/exp4_family_results.png`, `results/exp4_consistency.png`, `results/exp4_results.json`

---

## Test Suite

```
pytest tests/ -v
134 passed, 5 warnings in 1.56s
```

All warnings are benign:
- One `RuntimeWarning: divide by zero` in `branch_risky(z=1)` — expected, guarded by the `z > 1` domain test.
- Four `PytestRemovedIn10Warning` for class-scoped instance-method fixtures — style deprecation only, no logic impact.
