# Uncertain Volatility Model (UVM)

A Python implementation of the **Black–Scholes–Barenblatt (BSB) uncertain-volatility model** for pricing European-style derivatives under bounded, potentially time- and state-dependent volatility.

The project uses a finite-difference discretization and **fully implicit time stepping with policy iteration** to compute upper and lower no-arbitrage valuation bounds. It includes vanilla options, cash-or-nothing digital options, and continuously monitored knock-out barriers.

## Model

Under the risk-neutral dynamics

\[
\frac{dS_t}{S_t}=(r-q)\,dt+\sigma_t\,dW_t,
\qquad \sigma_t\in[\sigma_{\min},\sigma_{\max}],
\]

where \(r\) is the continuously compounded risk-free rate and \(q\) is the continuous dividend yield, the upper and lower value functions satisfy the BSB equations. In time to maturity \(\tau=T-t\),

\[
\frac{\partial V}{\partial\tau}
=\underset{\sigma\in[\sigma_{\min},\sigma_{\max}]}{\operatorname{ext}}
\left\{\frac12\sigma^2S^2\frac{\partial^2V}{\partial S^2}
+(r-q)S\frac{\partial V}{\partial S}-rV\right\},
\]

with **ext = max** for the upper valuation bound and **ext = min** for the lower bound.

The optimizing volatility is determined by the local curvature (gamma) of the value function:

| Gamma | Upper bound | Lower bound |
|---|---|---|
| \(V_{SS}\geq0\) | \(\sigma_{\max}\) | \(\sigma_{\min}\) |
| \(V_{SS}<0\) | \(\sigma_{\min}\) | \(\sigma_{\max}\) |

Consequently, a volatility bound need not equal the Black–Scholes price evaluated at one endpoint volatility: for contracts whose gamma changes sign, the selected volatility can vary across the computational domain and over time.

When \(\sigma_{\min}=\sigma_{\max}\), the equation reduces to the standard Black–Scholes PDE.

## Numerical method

- **Space:** three-point finite differences for first and second derivatives, including support for nonuniform spot grids.
- **Time:** backward (fully implicit) Euler stepping in time to maturity.
- **Nonlinearity:** policy iteration updates the volatility selection from the current estimate of gamma and solves the resulting tridiagonal linear system.
- **Linear algebra:** SciPy's banded tridiagonal solver.
- **Boundaries:** natural or prescribed Dirichlet conditions, depending on the contract and domain.

For smooth solutions on a suitable grid, the underlying discretization has first-order temporal and second-order spatial accuracy. Payoff nonsmoothness, boundaries, and volatility switching can affect observed convergence rates.

## Supported contracts

| Contract | Description |
|---|---|
| European vanilla call / put | Standard terminal payoff |
| Cash-or-nothing digital call / put | Binary payoff, with optional linear smoothing near the strike |
| Up-and-out call / put | Knocked out if the underlying touches the upper barrier |
| Down-and-out call / put | Knocked out if the underlying touches the lower barrier |

Barrier contracts are **continuously monitored and zero-rebate**. The knock-out barrier is represented by an absorbing boundary, \(V(B,\tau)=0\), at the edge of the computational grid. Knock-in contracts are not currently implemented.

A smoothed digital is a **different payoff** from an exact digital. The smoothing parameter controls the width of a linear transition region and can improve finite-difference behavior near a discontinuity.

## Repository layout

```text
uncertain-volatility-model/
├── uvm/
│   ├── __init__.py
│   ├── operators.py      # Finite-difference operators and gamma
│   ├── payoffs.py        # Terminal payoff functions
│   ├── solver.py         # Implicit BSB solver and policy iteration
│   └── pricing.py        # Contract-level pricing interfaces
├── tests/                # Numerical and analytical validation
├── README.md
├── requirements.txt
└── .gitignore
```

## Requirements

- Python 3.10+ (recommended)
- NumPy
- SciPy

Install the runtime dependencies from the project root:

```bash
python -m pip install numpy scipy
```

If a `requirements.txt` file is included, you can instead run:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Run the following from the repository root (or with the repository root on your Python import path):

```python
import numpy as np
from uvm.pricing import price_vanilla, price_digital, price_barrier

# European call: upper and lower uncertain-volatility bounds
spot_grid = np.linspace(0.0, 2.5, 1001)

common = dict(
    spot_grid=spot_grid,
    strike=1.0,
    maturity=1.0,
    n_steps=1000,
    r=0.035,
    q=0.005,
    sigma_min=0.10,
    sigma_max=0.20,
    option="call",
    spot=1.0,
)

upper = price_vanilla(**common, bound="upper")
lower = price_vanilla(**common, bound="lower")
print(f"Vanilla call bounds: [{lower:.8f}, {upper:.8f}]")

# Smoothed cash-or-nothing digital call
# smoothing=0.005 gives a linear transition on [K(1-0.01), K].
digital = price_digital(
    **common,
    bound="upper",
    cash=1.0,
    smoothing=0.005,
)
print(f"Upper digital call price: {digital:.8f}")

# Up-and-out call: the upper endpoint must coincide with the barrier.
barrier = 1.4
barrier_grid = np.linspace(0.0, barrier, 1001)
barrier_price = price_barrier(
    spot_grid=barrier_grid,
    strike=1.0,
    barrier=barrier,
    maturity=1.0,
    n_steps=1000,
    r=0.035,
    q=0.005,
    sigma_min=0.10,
    sigma_max=0.20,
    option="call",
    direction="up-and-out",
    bound="upper",
    spot=1.0,
)
print(f"Upper up-and-out call price: {barrier_price:.8f}")
```

**Note:** The barrier example uses the original `direction="up-and-out"` API. If the pricing interface is later refactored to separate `direction="up"` and `knock="out"`, update this example accordingly.

## Numerical validation

The constant-volatility limit (\(\sigma_{\min}=\sigma_{\max}\)) provides analytical benchmarks:

1. **Vanilla calls and puts:** compare against Black–Scholes prices.
2. **Digital calls and puts:** compare exact digitals against Black–Scholes cash-or-nothing formulas. For linearly smoothed digitals, compare against normalized vanilla option spreads using the same smoothing interval.
3. **Knock-out barriers:** compare against continuously monitored Black–Scholes barrier formulas, with consistent monitoring, rebate, dividend, and boundary conventions.
4. **Grid convergence:** vary time and space resolutions independently and verify the observed convergence rates.

### Up-and-out call: sample convergence study

For one constant-volatility barrier benchmark, the analytical value was **0.006828439494024313**. The following numerical values were obtained using fully implicit Euler:

| Time steps | 200 spatial steps | 400 spatial steps |
|---:|---:|---:|
| 200 | 0.006866068503089297 | 0.006868466147657601 |
| 400 | 0.006845640461574124 | 0.006848032708713526 |

Applying first-order Richardson extrapolation in time,

\[
R_M=2V_{400,M}-V_{200,M},
\]

followed by second-order extrapolation in space,

\[
V_{\mathrm{extrap}}=\frac{4R_{400}-R_{200}}{3},
\]

gives approximately **0.00682839488634**, an absolute difference of about **4.46 × 10⁻⁸** from the analytical benchmark.

This is a validation experiment, **not** an automatic guarantee of extrapolation accuracy for every payoff or volatility regime. In particular, extrapolating solutions of nonlinear BSB equations requires checking that the assumed convergence behavior holds.

## Limitations and potential extensions

- The current barrier implementation covers **knock-out** contracts only; knock-in valuation under uncertain volatility requires additional treatment of the activation state.
- Digital payoff smoothing changes the instrument being priced; the smoothing width should be disclosed in comparisons.
- A centered finite-difference discretization on strongly nonuniform grids may require additional monotonicity checks or an alternative drift discretization.
- Richardson extrapolation is demonstrated as a validation technique, rather than integrated into the core solver.
- Numerical agreement in the constant-volatility limit does not, by itself, establish correctness of nonlinear volatility switching; additional upper/lower-bound and policy-iteration tests are appropriate.

## Disclaimer

This repository is intended for quantitative research, numerical experimentation, and education. It is not investment advice or a production-certified valuation system.
