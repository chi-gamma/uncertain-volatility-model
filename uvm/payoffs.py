
import numpy as np


def call_payoff(spot_grid, strike):
    """Compute the terminal payoff of a European call."""
    return np.maximum(spot_grid - strike, 0.0)


def put_payoff(spot_grid, strike):
    """Compute the terminal payoff of a European put."""
    return np.maximum(strike - spot_grid, 0.0)


def digital_call_payoff(spot_grid, strike, cash=1.0):
    """Compute the terminal payoff of a cash-or-nothing digital call."""

    return cash * np.where(spot_grid >= strike, 1.0, 0.0)


def digital_put_payoff(spot_grid, strike, cash=1.0):
    """Compute the terminal payoff of a cash-or-nothing digital put."""
    return cash * np.where(spot_grid < strike, 1.0, 0.0)


def smoothed_digital_call_payoff(spot_grid, strike, g, cash=1.0):
    """Compute a linearly smoothed cash-or-nothing digital call payoff."""
    if not 0 < g < 0.5:
        raise ValueError("g must be between 0 and 0.5.")

    lower = strike * (1 - 2 * g)

    return cash * np.clip((spot_grid - lower) / (strike - lower), 0.0, 1.0)


def smoothed_digital_put_payoff(spot_grid, strike, g, cash=1.0):
    """Compute a linearly smoothed cash-or-nothing digital put payoff."""
    if not 0 < g < 0.5:
        raise ValueError("g must be between 0 and 0.5.")

    lower = strike * (1 - 2 * g)

    return cash * np.clip((strike - spot_grid) / (strike - lower), 0.0, 1.0)