import numpy as np
from scipy.linalg import solve_banded

from .operators import pde_matrix, gamma


def solve_uvm(
    spot_grid,
    payoff,
    maturity,
    n_steps,
    r,
    q,
    sigma_min,
    sigma_max,
    bound="upper",
    lower_bc="natural",
    upper_bc="natural",
    lower_value=None,
    upper_value=None,
    tol=1e-10,
    max_iter=100
):
    """
    Solve the uncertain-volatility pricing PDE using fully implicit Euler
    time stepping and nonlinear policy iteration.

    Parameters
    ----------
    spot_grid : array_like
        Strictly increasing stock-price grid.
    payoff : array_like
        Terminal payoff evaluated on the stock-price grid.
    maturity : float
        Time to maturity.
    n_steps : int
        Number of time steps.
    r, q : float
        Risk-free rate and dividend yield.
    sigma_min, sigma_max : float
        Lower and upper volatility bounds.
    bound : {'upper', 'lower'}
        Select upper or lower uncertain-volatility valuation.
    lower_bc, upper_bc : {'natural', 'dirichlet'}
        Spatial boundary conditions.
    lower_value, upper_value : callable or float, optional
        Dirichlet boundary values. Callables take time-to-maturity tau.
    tol : float
        Policy-iteration convergence tolerance.
    max_iter : int
        Maximum policy iterations per time step.

    Returns
    -------
    numpy.ndarray
        Option values at all stock-price grid points.
    """

    spot_grid = np.asarray(spot_grid, dtype=float)
    p = np.asarray(payoff, dtype=float).copy()

    if len(spot_grid) != len(p):
        raise ValueError("spot_grid and payoff must have the same length.")

    if bound not in ("upper", "lower"):
        raise ValueError("bound must be 'upper' or 'lower'.")

    if not 0 <= sigma_min <= sigma_max:
        raise ValueError("Invalid volatility bounds.")

    if maturity < 0 or n_steps < 1:
        raise ValueError("Invalid maturity or number of time steps.")

    if lower_bc == "dirichlet" and lower_value is None:
        raise ValueError("lower_value is required for Dirichlet boundary.")

    if upper_bc == "dirichlet" and upper_value is None:
        raise ValueError("upper_value is required for Dirichlet boundary.")

    if maturity == 0:
        return p

    dt = maturity / n_steps

    # Construct implicit Euler matrices: E = I - dt*L
    _, Y_min = pde_matrix(spot_grid, dt, r, q, sigma_min, lower_bc, upper_bc)
    _, Y_max = pde_matrix(spot_grid, dt, r, q, sigma_max, lower_bc, upper_bc)

    E_min = 2.0 * Y_min
    E_max = 2.0 * Y_max

    E_min[1, :] -= 1.0
    E_max[1, :] -= 1.0

    E = np.empty_like(E_min)

    def boundary_value(value, tau):
        return value(tau) if callable(value) else value

    for step in range(n_steps):

        tau = (step + 1) * dt

        rhs = p.copy()

        # Dirichlet conditions at the new time level
        if lower_bc == "dirichlet":
            rhs[0] = boundary_value(lower_value, tau)

        if upper_bc == "dirichlet":
            rhs[-1] = boundary_value(upper_value, tau)

        p_new = p.copy()

        for iteration in range(max_iter):

            # Determine volatility policy from current gamma
            use_max = gamma(spot_grid, p_new) >= 0.0

            if bound == "lower":
                use_max = ~use_max

            # Assemble matrix under current volatility policy
            E[:] = E_min

            E[1, 1:-1] = np.where(
                use_max, E_max[1, 1:-1], E_min[1, 1:-1]
            )

            E[0, 2:] = np.where(
                use_max, E_max[0, 2:], E_min[0, 2:]
            )

            E[2, :-2] = np.where(
                use_max, E_max[2, :-2], E_min[2, :-2]
            )

            candidate = solve_banded(
                (1, 1), E, rhs, check_finite=False
            )

            error = np.max(np.abs(candidate - p_new))
            p_new = candidate

            if error < tol:
                break

        else:
            raise RuntimeError(
                f"Policy iteration failed at time step {step + 1}."
            )

        p = p_new

    return p