import numpy as np

from .payoffs import (
    call_payoff,
    put_payoff,
    digital_call_payoff,
    digital_put_payoff,
    smoothed_digital_call_payoff,
    smoothed_digital_put_payoff,
)
from .solver import solve_uvm


def _validate_option(option):
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'.")


def _validate_grid(spot_grid):
    spot_grid = np.asarray(spot_grid, dtype=float)

    if spot_grid.ndim != 1 or len(spot_grid) < 3:
        raise ValueError("spot_grid must be a one-dimensional grid with at least 3 nodes.")

    if np.any(np.diff(spot_grid) <= 0):
        raise ValueError("spot_grid must be strictly increasing.")

    if spot_grid[0] < 0:
        raise ValueError("Stock spot_grid cannot be negative.")

    return spot_grid


def _payoff(spot_grid, strike, option):
    return (
        call_payoff(spot_grid, strike)
        if option == "call"
        else put_payoff(spot_grid, strike)
    )


def _vanilla_boundaries(spot_grid, strike, r, q, option):
    """
    Standard asymptotic vanilla boundary conditions.
    Lower boundary formulas are exact when S_min = 0.
    Upper boundary formulas are large-S approximations.
    """
    if option == "call":
        lower = lambda tau: 0.0
        upper = lambda tau: (
            spot_grid[-1] * np.exp(-q * tau)
            - strike * np.exp(-r * tau)
        )
    else:
        lower = lambda tau: strike * np.exp(-r * tau)
        upper = lambda tau: 0.0

    return lower, upper


def price_vanilla(
    spot_grid,
    strike,
    maturity,
    n_steps,
    r,
    q,
    sigma_min,
    sigma_max,
    option="call",
    bound="upper",
    spot=None
):
    """Price a European vanilla option under uncertain volatility."""

    _validate_option(option)
    spot_grid = _validate_grid(spot_grid)

    payoff = _payoff(spot_grid, strike, option)

    lower_value, upper_value = _vanilla_boundaries(
        spot_grid, strike, r, q, option
    )

    # At a positive artificial lower boundary, use the PDE
    # rather than imposing the S=0 asymptotic value.
    lower_bc = "dirichlet" if spot_grid[0] == 0.0 else "natural"

    values = solve_uvm(
        spot_grid=spot_grid,
        payoff=payoff,
        maturity=maturity,
        n_steps=n_steps,
        r=r,
        q=q,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        bound=bound,
        lower_bc=lower_bc,
        upper_bc="dirichlet",
        lower_value=lower_value if lower_bc == "dirichlet" else None,
        upper_value=upper_value,
    )
    
    if spot is None:
        return values
    
    if not spot_grid[0] <= spot <= spot_grid[-1]:
        raise ValueError("spot must lie within spot_grid.")
    
    return np.interp(spot, spot_grid, values)



def price_digital(
    spot_grid,
    strike,
    maturity,
    n_steps,
    r,
    q,
    sigma_min,
    sigma_max,
    option="call",
    bound="upper",
    cash=1.0,
    smoothing=None,
    spot=None
):
    """
    Price a cash-or-nothing digital option under uncertain volatility.

    smoothing : float or None
        If provided, use the linearly smoothed payoff with parameter g.
    """

    _validate_option(option)
    spot_grid = _validate_grid(spot_grid)

    if smoothing is None:
        payoff = (
            digital_call_payoff(spot_grid, strike, cash)
            if option == "call"
            else digital_put_payoff(spot_grid, strike, cash)
        )
    else:
        payoff = (
            smoothed_digital_call_payoff(spot_grid, strike, smoothing, cash)
            if option == "call"
            else smoothed_digital_put_payoff(spot_grid, strike, smoothing, cash)
        )

    if option == "call":
        lower_value = 0.0
        upper_value = lambda tau: cash * np.exp(-r * tau)
    else:
        lower_value = lambda tau: cash * np.exp(-r * tau)
        upper_value = 0.0

    lower_bc = "dirichlet" if spot_grid[0] == 0.0 else "natural"

    values = solve_uvm(
        spot_grid=spot_grid,
        payoff=payoff,
        maturity=maturity,
        n_steps=n_steps,
        r=r,
        q=q,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        bound=bound,
        lower_bc=lower_bc,
        upper_bc="dirichlet",
        lower_value=lower_value if lower_bc == "dirichlet" else None,
        upper_value=upper_value,
    )
    
    if spot is None:
        return values
    
    if not spot_grid[0] <= spot <= spot_grid[-1]:
        raise ValueError("spot must lie within spot_grid.")
    
    return np.interp(spot, spot_grid, values)


def price_barrier(
    spot_grid,
    strike,
    barrier,
    maturity,
    n_steps,
    r,
    q,
    sigma_min,
    sigma_max,
    option="call",
    direction="up-and-out",
    bound="upper",
    spot=None,
    rebate=0.0,
):
    """
    Price a continuously monitored knock-out option with an
    immediate-payment rebate upon hitting the barrier.

    The stock grid must include the barrier as its relevant endpoint:
        up-and-out:   spot_grid[-1] == barrier
        down-and-out: spot_grid[0]  == barrier

    rebate:
        Cash amount paid immediately when the barrier is hit.
        Default is zero.

    Returns the full value grid if spot is None;
    otherwise returns the interpolated price at spot.
    """

    _validate_option(option)
    spot_grid = _validate_grid(spot_grid)

    if direction not in ("up-and-out", "down-and-out"):
        raise ValueError("Unsupported barrier direction.")

    if rebate < 0:
        raise ValueError("rebate must be nonnegative.")

    if direction == "up-and-out":
        if not np.isclose(spot_grid[-1], barrier):
            raise ValueError(
                "For up-and-out options, the grid must end at the barrier."
            )

        lower_bc = "dirichlet" if spot_grid[0] == 0.0 else "natural"
        upper_bc = "dirichlet"

        if lower_bc == "dirichlet":
            lower_value = (
                0.0 if option == "call"
                else lambda tau: strike * np.exp(-r * tau)
            )
        else:
            lower_value = None

        # Immediate rebate at the upper barrier.
        upper_value = rebate

    else:
        if not np.isclose(spot_grid[0], barrier):
            raise ValueError(
                "For down-and-out options, the grid must start at the barrier."
            )

        lower_bc = "dirichlet"
        upper_bc = "dirichlet"

        # Immediate rebate at the lower barrier.
        lower_value = rebate

        upper_value = (
            lambda tau: (
                spot_grid[-1] * np.exp(-q * tau)
                - strike * np.exp(-r * tau)
            )
            if option == "call"
            else 0.0
        )

    payoff = _payoff(spot_grid, strike, option)

    # Apply the rebate at the knock-out boundary at maturity.
    if direction == "up-and-out":
        payoff[-1] = rebate
    else:
        payoff[0] = rebate

    values = solve_uvm(
        spot_grid=spot_grid,
        payoff=payoff,
        maturity=maturity,
        n_steps=n_steps,
        r=r,
        q=q,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        bound=bound,
        lower_bc=lower_bc,
        upper_bc=upper_bc,
        lower_value=lower_value,
        upper_value=upper_value,
    )

    if spot is None:
        return values

    if not spot_grid[0] <= spot <= spot_grid[-1]:
        raise ValueError("spot must lie within spot_grid.")

    return np.interp(spot, spot_grid, values)
