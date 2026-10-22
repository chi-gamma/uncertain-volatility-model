
import numpy as np
import pytest
from scipy.stats import norm

from uvm.pricing import price_digital


# ------------------------------------------------------------
# Test parameters
# ------------------------------------------------------------

SPOT = 1.0
STRIKE = 1.1
MATURITY = 1.0
R = 0.035
Q = 0.005

SIGMA = 0.115
SIGMA_MIN = 0.115
SIGMA_MAX = 0.25

CASH = 1.0
SMOOTHING = 0.005

N_SPOT_STEPS = 1000
N_TIME_STEPS = 1000

SPOT_GRID = np.linspace(0.5, 2.5, N_SPOT_STEPS + 1)

TOL_SMOOTH = 1e-4
TOL_EXACT = 5e-3


# ------------------------------------------------------------
# Analytical Black-Scholes formulas
# ------------------------------------------------------------

def black_scholes(spot, strike, maturity, r, q, sigma, option):
    d1 = (
        np.log(spot / strike)
        + (r - q + 0.5 * sigma**2) * maturity
    ) / (sigma * np.sqrt(maturity))

    d2 = d1 - sigma * np.sqrt(maturity)

    if option == "call":
        return (
            spot * np.exp(-q * maturity) * norm.cdf(d1)
            - strike * np.exp(-r * maturity) * norm.cdf(d2)
        )

    if option == "put":
        return (
            strike * np.exp(-r * maturity) * norm.cdf(-d2)
            - spot * np.exp(-q * maturity) * norm.cdf(-d1)
        )

    raise ValueError("option must be 'call' or 'put'.")


def analytical_digital(option, sigma):
    d2 = (
        np.log(SPOT / STRIKE)
        + (R - Q - 0.5 * sigma**2) * MATURITY
    ) / (sigma * np.sqrt(MATURITY))

    probability = norm.cdf(d2) if option == "call" else norm.cdf(-d2)

    return CASH * np.exp(-R * MATURITY) * probability


def analytical_smoothed_digital(option, sigma):
    k1 = STRIKE * (1 - 2 * SMOOTHING)
    k2 = STRIKE

    if option == "call":
        spread = (
            black_scholes(SPOT, k1, MATURITY, R, Q, sigma, "call")
            - black_scholes(SPOT, k2, MATURITY, R, Q, sigma, "call")
        )
    else:
        spread = (
            black_scholes(SPOT, k2, MATURITY, R, Q, sigma, "put")
            - black_scholes(SPOT, k1, MATURITY, R, Q, sigma, "put")
        )

    return CASH * spread / (k2 - k1)


# ------------------------------------------------------------
# PDE pricing helper
# ------------------------------------------------------------

def pde_price(
    option,
    bound="upper",
    sigma_min=SIGMA,
    sigma_max=SIGMA,
    smoothing=None,
):
    return price_digital(
        spot_grid=SPOT_GRID,
        strike=STRIKE,
        maturity=MATURITY,
        n_steps=N_TIME_STEPS,
        r=R,
        q=Q,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        option=option,
        bound=bound,
        cash=CASH,
        smoothing=smoothing,
        spot=SPOT,
    )


# ------------------------------------------------------------
# 1. Exact digital: PDE vs analytical Black-Scholes
# ------------------------------------------------------------

@pytest.mark.parametrize("option", ["call", "put"])
def test_exact_digital(option):
    numerical = pde_price(option, smoothing=None)
    analytical = analytical_digital(option, SIGMA)

    assert abs(numerical - analytical) < TOL_EXACT


# ------------------------------------------------------------
# 2. Smoothed digital: PDE vs analytical option spread
# ------------------------------------------------------------

@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("bound", ["upper", "lower"])
def test_smoothed_digital(option, bound):
    numerical = pde_price(
        option,
        bound=bound,
        smoothing=SMOOTHING,
    )

    analytical = analytical_smoothed_digital(option, SIGMA)

    assert abs(numerical - analytical) < TOL_SMOOTH


# ------------------------------------------------------------
# 3. Call-put complementarity (constant volatility)
# ------------------------------------------------------------

def test_digital_complementarity():
    call = pde_price("call", smoothing=SMOOTHING)
    put = pde_price("put", smoothing=SMOOTHING)

    expected = CASH * np.exp(-R * MATURITY)

    assert abs(call + put - expected) < TOL_SMOOTH


# ------------------------------------------------------------
# 4. Upper and lower UVM bounds
# ------------------------------------------------------------

@pytest.mark.parametrize("option", ["call", "put"])
def test_uvm_bounds(option):
    lower = pde_price(
        option,
        bound="lower",
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        smoothing=SMOOTHING,
    )

    upper = pde_price(
        option,
        bound="upper",
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        smoothing=SMOOTHING,
    )

    assert lower <= upper + 1e-10


# ------------------------------------------------------------
# 5. Complementarity across opposite UVM bounds
# ------------------------------------------------------------

def test_uvm_complementarity():
    call_upper = pde_price(
        "call", "upper", SIGMA_MIN, SIGMA_MAX, SMOOTHING
    )

    put_lower = pde_price(
        "put", "lower", SIGMA_MIN, SIGMA_MAX, SMOOTHING
    )

    expected = CASH * np.exp(-R * MATURITY)

    assert abs(call_upper + put_lower - expected) < TOL_SMOOTH
