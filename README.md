# Analytic Reproduction of Transient Price Impact Examples

Analytic and finite-state computational reproductions of four examples from the
paper on transient price impact, non-convexity of the attainable payoff set, and
non-concavity of indirect utility.

---

## Repository Structure

```
.
├── config/              Static JSON parameter manifests for all four experiments
├── data/                No market data (all experiments use synthetic/analytic inputs)
├── exp/                 Experiment runner scripts
│   ├── run_exp1.py      Example 2.7 — non-convexity of attainable set
│   ├── run_exp2.py      Example 2.9 — non-concavity of indirect utility
│   ├── run_exp3.py      Example 2.10 — finite-state DP replication
│   ├── run_exp4.py      Stress-test suite (non-monotone liquidity)
│   └── run_all_experiments.py   Master runner (all four experiments)
├── src/
│   ├── model.py         Core transient price impact mechanics
│   ├── example_27.py    Example 2.7 implementation + quadratic form
│   ├── example_29.py    Example 2.9 implementation + symbolic analysis
│   ├── example_210.py   Example 2.10 backward DP implementation
│   └── stress_tests.py  Exp 4 stress-test suite (four families A–D)
├── tests/
│   ├── test_exp1.py     pytest tests for Experiment 1
│   ├── test_exp2.py     pytest tests for Experiment 2
│   ├── test_exp3.py     pytest tests for Experiment 3
│   └── test_exp4.py     pytest tests for Experiment 4
└── results/             Output metrics, plots, and RESULTS.md
```

---

## Experiments

### Experiment 1 — Example 2.7: Non-convexity of Attainable Set

**Claim:** Without the monotonicity condition on ρ²δ, the superhedging set S can
fail convexity.

**Method:** Symbolic/numeric witness construction with T=3, r_t=0, δ=(1,10,10).
Evaluates the quadratic form Q(h_1,h_2) = (21/10)h_1² + (23/10)h_1h_2 + (3/10)h_2²
at the two witness strategies H=(1,1,−2), G=(1.5,0,−1.5) and their midpoint H̄.

**Expected:** Q(H)=4.7, Q(G)=4.725, Q(H̄)=4.79375, gap=−0.08125 < 0.

### Experiment 2 — Example 2.9: Non-concavity of Indirect Utility

**Claim:** A non-convex feasible strategy set can make the indirect utility non-concave.

**Method:** One-step frictionless model with Φ={0,1}, u(x)=ln(x), R∈{2,−1}.
Analytic crossing at z=2; derivative jump from 1/2 to 5/8 proves non-concavity.

### Experiment 3 — Example 2.10: Finite-State Dynamic Programming

**Claim:** Backward recursion concavifies continuation values but G̃_0 can remain
non-concave.

**Method:** T=4 binary scenario tree, r_t=0.01, δ=(10,10,15,5), P(up)=0.77.
Backward DP with exhaustive action grid; validated via forward policy evaluation
and pathwise accounting.

### Experiment 4 — Stress Tests: Non-Monotone Liquidity

**Claim:** Optimizer existence and DP consistency hold even in non-monotone
liquidity environments.

**Method:** Four families (A=non-monotone, B=monotone control, C=horizon
sensitivity, D=utility-shape sensitivity). Three-way cross-validation: DP == 
forward policy == brute-force exhaustive search.

---

## Usage

### Run all experiments

```bash
python exp/run_all_experiments.py
```

### Run individual experiments

```bash
python exp/run_exp1.py
python exp/run_exp2.py
python exp/run_exp3.py
python exp/run_exp4.py
```

### Run tests

```bash
pytest tests/ -v
```

---

## Dependencies

- Python ≥ 3.10
- numpy
- sympy
- scipy
- matplotlib

No market data API is required. All experiments use synthetic/analytic inputs.

---

## Results

See `results/RESULTS.md` for a full summary of all experiment findings, including
exact numeric outcomes, validation status, and plots.
