
"""
Analytical Black-Scholes prices for continuously monitored
European knock-out barrier options.

Supported contracts:
    - Up-and-out call
    - Up-and-out put
    - Down-and-out call
    - Down-and-out put

R is the cash rebate paid immediately upon barrier breach.
R=0 corresponds to a zero-rebate knock-out option.
"""

import numpy as np
from scipy.stats import norm


def _barrier_price(S0, K, T, r, q, sigma, B, R, phi, eta):
    """
    Reiner-Rubinstein-style analytical barrier pricing formula.

    Parameters
    ----------
    S0 : float
        Initial underlying price.
    K : float
        Strike.
    T : float
        Time to maturity.
    r : float
        Continuously compounded risk-free rate.
    q : float
        Continuous dividend yield.
    sigma : float
        Constant volatility.
    B : float
        Knock-out barrier.
    R : float
        Cash rebate paid immediately upon knock-out.
    phi : int
        +1 for calls, -1 for puts.
    eta : int
        +1 for down barriers, -1 for up barriers.
    """

    if S0 <= 0 or K <= 0 or B <= 0:
        raise ValueError("S0, K, and B must be positive.")

    if T <= 0:
        raise ValueError("T must be positive.")

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    if R < 0:
        raise ValueError("R must be nonnegative.")

    if phi not in (-1, 1) or eta not in (-1, 1):
        raise ValueError("phi and eta must be +1 or -1.")

    # Already knocked out at valuation.
    if eta == -1 and S0 >= B:
        return float(R)

    if eta == 1 and S0 <= B:
        return float(R)

    vol_t = sigma * np.sqrt(T)

    h = (r - q + 0.5 * sigma**2) / sigma**2
    mu = r - q - 0.5 * sigma**2

    a = mu / sigma**2
    b = np.sqrt(mu**2 + 2.0 * r * sigma**2) / sigma**2

    ratio = B / S0

    discount_r = np.exp(-r * T)
    discount_q = np.exp(-q * T)

    x = np.log(S0 / K) / vol_t + h * vol_t
    x1 = np.log(S0 / B) / vol_t + h * vol_t

    y = np.log(B**2 / (S0 * K)) / vol_t + h * vol_t
    y1 = np.log(B / S0) / vol_t + h * vol_t

    z = np.log(B / S0) / vol_t + b * vol_t

    # --------------------------------------------------------
    # Standard Black-Scholes components
    # --------------------------------------------------------

    def vanilla_component(d):
        return phi * (
            S0 * discount_q * norm.cdf(phi * d)
            - K * discount_r * norm.cdf(phi * (d - vol_t))
        )

    def reflected_component(d):
        return phi * (
            S0 * discount_q * ratio**(2.0 * h)
            * norm.cdf(eta * d)
            - K * discount_r * ratio**(2.0 * h - 2.0)
            * norm.cdf(eta * (d - vol_t))
        )

    v1 = vanilla_component(x)
    v2 = vanilla_component(x1)

    v3 = reflected_component(y)
    v4 = reflected_component(y1)

    # --------------------------------------------------------
    # Immediate-payment rebate component
    # --------------------------------------------------------

    v6 = R * (
        ratio**(a + b) * norm.cdf(eta * z)
        + ratio**(a - b)
        * norm.cdf(eta * (z - 2.0 * b * vol_t))
    )

    # --------------------------------------------------------
    # Select the appropriate barrier payoff formula
    # --------------------------------------------------------

    if eta == -1:
        # Up-and-out

        if phi == 1:
            # Up-and-out call
            if K >= B:
                value = v6
            else:
                value = v1 - v2 + v3 - v4 + v6

        else:
            # Up-and-out put
            if K > B:
                value = v2 - v4 + v6
            else:
                value = v1 - v3 + v6

    else:
        # Down-and-out

        if phi == 1:
            # Down-and-out call
            if K > B:
                value = v1 - v3 + v6
            else:
                value = v2 - v4 + v6

        else:
            # Down-and-out put
            if K > B:
                value = v1 - v2 + v3 - v4 + v6
            else:
                value = v6

    return float(value)


# ------------------------------------------------------------
# Public analytical pricing functions
# ------------------------------------------------------------

def up_out_call_barrier(S0, K, T, r, q, sigma, B, R=0.0):
    """Analytical up-and-out European call."""
    return _barrier_price(
        S0, K, T, r, q, sigma, B, R,
        phi=1, eta=-1,
    )


def up_out_put_barrier(S0, K, T, r, q, sigma, B, R=0.0):
    """Analytical up-and-out European put."""
    return _barrier_price(
        S0, K, T, r, q, sigma, B, R,
        phi=-1, eta=-1,
    )


def down_out_call_barrier(S0, K, T, r, q, sigma, B, R=0.0):
    """Analytical down-and-out European call."""
    return _barrier_price(
        S0, K, T, r, q, sigma, B, R,
        phi=1, eta=1,
    )


def down_out_put_barrier(S0, K, T, r, q, sigma, B, R=0.0):
    """Analytical down-and-out European put."""
    return _barrier_price(
        S0, K, T, r, q, sigma, B, R,
        phi=-1, eta=1,
    )
