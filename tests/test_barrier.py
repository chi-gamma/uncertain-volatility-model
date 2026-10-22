
import numpy as np
import pytest

from uvm.pricing import price_barrier
from tests.analytical_barriers import (
    up_out_call_barrier,
    up_out_put_barrier,
    down_out_call_barrier,
    down_out_put_barrier,
)


# ------------------------------------------------------------
# Model parameters
# ------------------------------------------------------------

S0 = 1.0
K = 1.0
T = 1.0

R = 0.035
Q = 0.005

SIGMA = 0.115
SIGMA_MIN = 0.115
SIGMA_MAX = 0.25

N_SPACE = 500
N_TIME = 500

PRICE_TOL = 5e-5
BOUND_TOL = 1e-9


# ------------------------------------------------------------
# Contracts
# ------------------------------------------------------------

CONTRACTS = [
    ("call", "up-and-out", 1.25),
    ("put", "up-and-out", 1.25),
    ("call", "down-and-out", 0.75),
    ("put", "down-and-out", 0.75),
]

ANALYTICAL_FUNCTIONS = {
    ("call", "up-and-out"): up_out_call_barrier,
    ("put", "up-and-out"): up_out_put_barrier,
    ("call", "down-and-out"): down_out_call_barrier,
    ("put", "down-and-out"): down_out_put_barrier,
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def make_grid(direction, barrier, n_space=N_SPACE):
    if direction == "up-and-out":
        return np.linspace(0.0, barrier, n_space + 1)

    if direction == "down-and-out":
        return np.linspace(barrier, 3.0, n_space + 1)

    raise ValueError("Unsupported barrier direction.")


def analytical_price(option, direction, barrier, rebate=0.0):
    pricing_function = ANALYTICAL_FUNCTIONS[(option, direction)]

    return pricing_function(
        S0=S0,
        K=K,
        T=T,
        r=R,
        q=Q,
        sigma=SIGMA,
        B=barrier,
        R=rebate,
    )


def numerical_price(
    option,
    direction,
    barrier,
    rebate=0.0,
    bound="upper",
    sigma_min=SIGMA,
    sigma_max=SIGMA,
    n_space=N_SPACE,
    n_time=N_TIME,
    spot=S0,
):
    return price_barrier(
        spot_grid=make_grid(direction, barrier, n_space),
        strike=K,
        barrier=barrier,
        maturity=T,
        n_steps=n_time,
        r=R,
        q=Q,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        option=option,
        direction=direction,
        bound=bound,
        spot=spot,
        rebate=rebate,
    )


# ------------------------------------------------------------
# 1. Analytical Black-Scholes benchmarks
# ------------------------------------------------------------

@pytest.mark.parametrize("option,direction,barrier", CONTRACTS)
@pytest.mark.parametrize("rebate", [0.0, 0.01])
def test_analytical_barrier(option, direction, barrier, rebate):
    analytical = analytical_price(
        option, direction, barrier, rebate
    )

    numerical = numerical_price(
        option, direction, barrier,
        rebate=rebate,
        n_space=1000,
        n_time=1000,
    )

    error = abs(numerical - analytical)

    assert error < PRICE_TOL, (
        f"Analytical: {analytical:.12f}, "
        f"Numerical: {numerical:.12f}, "
        f"Error: {error:.3e}"
    )


# ------------------------------------------------------------
# 2. Boundary value equals rebate
# ------------------------------------------------------------

@pytest.mark.parametrize("option,direction,barrier", CONTRACTS)
@pytest.mark.parametrize("rebate", [0.0, 0.01])
def test_barrier_boundary(option, direction, barrier, rebate):
    values = numerical_price(
        option, direction, barrier,
        rebate=rebate,
        spot=None,
    )

    index = -1 if direction == "up-and-out" else 0

    assert abs(values[index] - rebate) < BOUND_TOL


# ------------------------------------------------------------
# 3. Constant-volatility upper/lower consistency
# ------------------------------------------------------------

@pytest.mark.parametrize("option,direction,barrier", CONTRACTS)
def test_constant_volatility(option, direction, barrier):
    upper = numerical_price(
        option, direction, barrier,
        rebate=0.01,
        bound="upper",
    )

    lower = numerical_price(
        option, direction, barrier,
        rebate=0.01,
        bound="lower",
    )

    assert abs(upper - lower) < BOUND_TOL


# ------------------------------------------------------------
# 4. Uncertain-volatility upper/lower ordering
# ------------------------------------------------------------

@pytest.mark.parametrize("option,direction,barrier", CONTRACTS)
@pytest.mark.parametrize("rebate", [0.0, 0.01])
def test_uvm_bounds(option, direction, barrier, rebate):
    lower = numerical_price(
        option, direction, barrier,
        rebate=rebate,
        bound="lower",
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
    )

    upper = numerical_price(
        option, direction, barrier,
        rebate=rebate,
        bound="upper",
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
    )

    assert lower <= upper + BOUND_TOL


# ------------------------------------------------------------
# 5. Barrier monotonicity for zero-rebate contracts
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "option,direction,near_barrier,far_barrier",
    [
        ("call", "up-and-out", 1.25, 1.40),
        ("put", "up-and-out", 1.25, 1.40),
        ("call", "down-and-out", 0.75, 0.60),
        ("put", "down-and-out", 0.75, 0.60),
    ],
)
def test_barrier_monotonicity(
    option, direction, near_barrier, far_barrier
):
    near_price = numerical_price(
        option, direction, near_barrier,
        rebate=0.0,
    )

    far_price = numerical_price(
        option, direction, far_barrier,
        rebate=0.0,
    )

    MONOTONICITY_TOL = 5e-6

    assert near_price <= far_price + MONOTONICITY_TOL, (
        f"Near barrier price: {near_price:.12f}, "
        f"Far barrier price: {far_price:.12f}, "
        f"Difference: {near_price - far_price:.3e}"
    )


# ------------------------------------------------------------
# 6. Increasing rebate increases option value
# ------------------------------------------------------------

@pytest.mark.parametrize("option,direction,barrier", CONTRACTS)
def test_rebate_monotonicity(option, direction, barrier):
    without_rebate = numerical_price(
        option, direction, barrier,
        rebate=0.0,
    )

    with_rebate = numerical_price(
        option, direction, barrier,
        rebate=0.01,
    )

    assert with_rebate >= without_rebate - BOUND_TOL


# ------------------------------------------------------------
# 7. Grid refinement and analytical accuracy
# ------------------------------------------------------------

@pytest.mark.parametrize("option,direction,barrier", CONTRACTS)
@pytest.mark.parametrize("rebate", [0.0, 0.01])
def test_grid_convergence(option, direction, barrier, rebate):
    analytical = analytical_price(
        option, direction, barrier, rebate
    )

    coarse = numerical_price(
        option, direction, barrier,
        rebate=rebate,
        n_space=200,
        n_time=200,
    )

    fine = numerical_price(
        option, direction, barrier,
        rebate=rebate,
        n_space=800,
        n_time=800,
    )

    coarse_error = abs(coarse - analytical)
    fine_error = abs(fine - analytical)

    assert fine_error < PRICE_TOL

    assert fine_error <= max(
        2.0 * coarse_error,
        PRICE_TOL,
    )
