# Uncertain Volatility Model

A Python implementation of the **Black–Scholes–Barenblatt (BSB) uncertain-volatility model** for pricing European-style derivatives under bounded volatility uncertainty.

Rather than assuming a single volatility parameter, the model considers all admissible volatility processes within a prescribed interval and computes the upper and lower no-arbitrage valuation bounds.

The framework implements a nonlinear finite-difference PDE solver with implicit time stepping and policy iteration, supporting vanilla options, cash-or-nothing digital options, and continuously monitored knock-out barrier options.

## 1. Mathematical Framework

Following the uncertain-volatility framework of Avellaneda, Levy, and Parás (1995), the underlying asset evolves according to

```math
dS_t=(r-q)S_t\,dt+\sigma_t S_t\,dW_t,
\qquad
\sigma_t\in[\sigma_{\min},\sigma_{\max}],
```

where \(r\) is the continuously compounded risk-free rate, \(q\) is the continuous dividend yield, and volatility is allowed to vary within fixed bounds.

The upper and lower option values are obtained by maximizing or minimizing the discounted expected payoff over admissible volatility processes.

Writing \(\tau=T-t\) for time to maturity, the corresponding Black–Scholes–Barenblatt PDE is

```math
\frac{\partial V}{\partial \tau}
=
\underset{\sigma\in[\sigma_{\min},\sigma_{\max}]}{\operatorname{ext}}
\left[
\frac{1}{2}\sigma^2S^2
\frac{\partial^2 V}{\partial S^2}
+(r-q)S\frac{\partial V}{\partial S}
-rV
\right],
```

where \(\operatorname{ext}\) denotes a supremum for the upper valuation bound and an infimum for the lower valuation bound.

### Gamma-Dependent Volatility Selection

The optimal local volatility is determined by the sign of option gamma.

For the **upper valuation bound**,

```math
\sigma^*_{\mathrm{upper}}=
\begin{cases}
\sigma_{\max}, & V_{SS}\geq 0,\\
\sigma_{\min}, & V_{SS}<0.
\end{cases}
```

For the **lower valuation bound**,

```math
\sigma^*_{\mathrm{lower}}=
\begin{cases}
\sigma_{\min}, & V_{SS}\geq 0,\\
\sigma_{\max}, & V_{SS}<0.
\end{cases}
```

When the volatility bounds coincide, the nonlinear PDE reduces to the standard Black–Scholes PDE.

## 2. Numerical Method

The BSB PDE is solved using a fully implicit finite-difference scheme on a potentially nonuniform asset-price grid.

### Spatial Discretization

First- and second-order spatial derivatives are approximated using three-point finite-difference stencils adapted to nonuniform grid spacing.

The resulting spatial operator has a tridiagonal structure.

### Implicit Time Stepping

With \(\Delta\tau=T/N\), the fully implicit discretization takes the form

```math
\left(I-\Delta\tau L_{\sigma^*}\right)
V^{n+1}=V^n,
```

where \(L_{\sigma^*}\) is the discrete Black–Scholes operator evaluated under the selected volatility policy.

The implicit scheme provides first-order temporal accuracy, while the interior spatial discretization is second-order accurate for sufficiently smooth solutions.

### Policy Iteration

Because the volatility policy depends on the unknown solution gamma, each time step requires solving a nonlinear system.

Policy iteration is used to resolve this dependence:

1. Initialize the volatility policy.
2. Construct the tridiagonal operator under the current policy.
3. Solve the implicit linear system.
4. Recompute discrete gamma and update the volatility policy.
5. Repeat until the policy and solution converge.

Tridiagonal systems are solved using SciPy's banded linear solver.

The framework supports Dirichlet and natural boundary conditions, including time-dependent Dirichlet values.

## 3. Supported Derivatives

### Vanilla European Options

European calls and puts with terminal payoffs

```math
C(S,T)=(S-K)^+,
\qquad
P(S,T)=(K-S)^+.
```

For these convex payoffs, the upper and lower uncertain-volatility valuations recover the Black–Scholes prices at \(\sigma_{\max}\) and \(\sigma_{\min}\), respectively.

### Cash-or-Nothing Digital Options

European digital calls and puts with terminal payoffs

```math
D_{\mathrm{call}}(S,T)
=R\,\mathbf{1}_{\{S\geq K\}},
\qquad
D_{\mathrm{put}}(S,T)
=R\,\mathbf{1}_{\{S<K\}}.
```

Both discontinuous digital payoffs and optionally smoothed payoffs are supported.

Smoothing replaces the discontinuity near the strike with a linear transition, corresponding under constant volatility to a normalized European option spread.

Unlike vanilla options, digital option gamma changes sign across the pricing domain, making the volatility-switching mechanism particularly relevant.

### Continuously Monitored Knock-Out Barrier Options

The framework supports four European knock-out structures:

- Up-and-out calls
- Up-and-out puts
- Down-and-out calls
- Down-and-out puts

Barrier monitoring is continuous, with the knock-out level imposed as an absorbing boundary in the PDE.

For a zero-rebate knock-out,

```math
V(B,\tau)=0.
```

An optional cash rebate paid immediately upon barrier breach is supported through

```math
V(B,\tau)=R.
```

The corresponding terminal payoff is the vanilla payoff on the surviving domain, with the rebate imposed at the barrier.

The PDE formulation naturally incorporates barrier monitoring without explicit path simulation.

## 4. Numerical Validation

The implementation is validated using automated tests covering:

- Analytical Black–Scholes recovery for European calls and puts.
- Analytical digital pricing and normalized call/put spread relationships.
- Digital payoff complementarity.
- Analytical knock-out barrier pricing under constant volatility, including nonzero rebates.
- Absorbing and rebate boundary conditions.
- Upper and lower uncertain-volatility valuation ordering.
- Barrier monotonicity and numerical grid refinement.

The current automated validation suite contains **63 passing tests**.

## References

1. **Avellaneda, M., Levy, A., and Parás, A. (1995).** *Pricing and Hedging Derivative Securities in Markets with Uncertain Volatilities.* Applied Mathematical Finance, 2(2), 73–88.

2. **Reiner, E., and Rubinstein, M. (1991).** *Breaking Down the Barriers.* Risk, 4(8), 28–35.
