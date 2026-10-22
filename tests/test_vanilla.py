
import numpy as np
import pytest
from scipy.stats import norm

from uvm.pricing import price_vanilla


# ------------------------------------------------------------
# Test parameters
# ------------------------------------------------------------

SPOT = 1.0
STRIKE = 1.0
MATURITY = 1.0
R = 0.035
Q = 0.005

SIGMA_MIN = 0.115
SIGMA_MAX = 0.25

N_SPOT_STEPS = 500
N_TIME_STEPS = 500

SPOT_GRID = np.linspace(0.0, 3.0, N_SPOT_STEPS + 1)

TOL = 1e-4


# ------------------------------------------------------------
# Analytical Black-Scholes
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


# ------------------------------------------------------------
# PDE pricing helper
# ------------------------------------------------------------

def pde_price(option, bound, sigma_min, sigma_max):
    return price_vanilla(
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
        spot=SPOT,
    )


# ------------------------------------------------------------
# 1. Constant volatility: PDE vs Black-Scholes
# ------------------------------------------------------------

@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("bound", ["upper", "lower"])
def test_constant_volatility(option, bound):
    sigma = 0.115

    numerical = pde_price(option, bound, sigma, sigma)
    analytical = black_scholes(
        SPOT, STRIKE, MATURITY, R, Q, sigma, option
    )

    assert abs(numerical - analytical) < TOL


# ------------------------------------------------------------
# 2. Upper UVM bound: Black-Scholes at maximum volatility
# ------------------------------------------------------------

@pytest.mark.parametrize("option", ["call", "put"])
def test_upper_bound(option):
    numerical = pde_price(
        option, "upper", SIGMA_MIN, SIGMA_MAX
    )

    analytical = black_scholes(
        SPOT, STRIKE, MATURITY, R, Q, SIGMA_MAX, option
    )

    assert abs(numerical - analytical) < TOL


# ------------------------------------------------------------
# 3. Lower UVM bound: Black-Scholes at minimum volatility
# ------------------------------------------------------------

@pytest.mark.parametrize("option", ["call", "put"])
def test_lower_bound(option):
    numerical = pde_price(
        option, "lower", SIGMA_MIN, SIGMA_MAX
    )

    analytical = black_scholes(
        SPOT, STRIKE, MATURITY, R, Q, SIGMA_MIN, option
    )

    assert abs(numerical - analytical) < TOL


# ------------------------------------------------------------
# 4. Put-call parity under constant volatility
# ------------------------------------------------------------

def test_put_call_parity():
    sigma = 0.115

    call = pde_price("call", "upper", sigma, sigma)
    put = pde_price("put", "upper", sigma, sigma)

    parity = (
        SPOT * np.exp(-Q * MATURITY)
        - STRIKE * np.exp(-R * MATURITY)
    )

    assert abs((call - put) - parity) < TOL
