import numpy as np
from scipy.special import ndtr


def pde_matrix(
    prices, 
    d, 
    r, 
    q, 
    v,
    lower_bc="natural",
    upper_bc="natural"
):
    
    """
    Construct finite-difference matrices on a potentially nonuniform stock-price grid.
    
    
    Parameters
    ----------
    - prices : Array of strictly increasing stock prices
    - d : Time step size
    - r : Risk-free rate
    - q : Dividend yield
    - v : Volatility
    - lower_bc : lower boundary condition ('natural' or 'dirichlet')
    - upper_bc : upper boundary condition ('natural' or 'dirichlet')

    Returns
    - x and y matrices with boundary and interior coefficients set

    """
    
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    ds = np.diff(prices)
    
    #============== Y matrix ===============#
    X = np.zeros((n, 3))
    
    # Lower boundary
    if lower_bc == "natural":
        z = prices[0] / ds[0]

        X[0, 1] = 1 + 0.5 * d * (z * (q - r) - r)
        X[0, 2] = 0.5 * d * z * (r - q)

    elif lower_bc == "dirichlet":
        X[0, 1] = 1.0

    else:
        raise ValueError("Unsupported lower boundary condition")

    # Interior
    S = prices[1:-1]

    h_minus = ds[:-1]
    h_plus = ds[1:]

    # first-derivative coefficients
    D1_lower = -h_plus / (h_minus * (h_minus + h_plus))
    D1_main = (h_plus - h_minus) / (h_minus * h_plus)
    D1_upper = h_minus / (h_plus * (h_minus + h_plus))

    # second-derivative coefficients
    D2_lower = 2.0 / (h_minus * (h_minus + h_plus))
    D2_main = -2.0 / (h_minus * h_plus)
    D2_upper = 2.0 / (h_plus * (h_minus + h_plus))

    # PDE coefficients
    diffusion = 0.5 * v**2 * S**2
    drift = (r - q) * S

    # Spatial operator L
    L_lower = diffusion * D2_lower + drift * D1_lower
    L_main = diffusion * D2_main + drift * D1_main - r
    L_upper = diffusion * D2_upper + drift * D1_upper

    # X = I + 0.5*d*L
    X[1:-1, 0] = 0.5 * d * L_lower
    X[1:-1, 1] = 1.0 + 0.5 * d * L_main
    X[1:-1, 2] = 0.5 * d * L_upper

    # Upper boundary

    if upper_bc == "natural":
        z = prices[-1] / ds[-1]

        X[-1, 0] = 0.5 * d * z * (q - r)
        X[-1, 1] = 1 + 0.5 * d * (z * (r - q) - r)

    elif upper_bc == "dirichlet":
        X[-1, 1] = 1.0

    else:
        raise ValueError("Unsupported upper boundary condition")

    #============== Y matrix ===============#

    Y = np.zeros((3, n))

    Y[0, 1:] = -X[:-1, 2]
    Y[1, :] = 2.0 - X[:, 1]
    Y[2, :-1] = -X[1:, 0]

    return X, Y
    

def apply_operator(X, p):
    """Apply a row-aligned tridiagonal matrix X to the vector p."""
    
    result = X[:, 1] * p
    result[1:] += X[1:, 0] * p[:-1]
    result[:-1] += X[:-1, 2] * p[1:]
    
    return result


def gamma(prices, values):
    """Compute option gamma at interior nodes on a nonuniform grid."""

    ds = np.diff(prices)

    h_minus = ds[:-1]
    h_plus = ds[1:]

    return 2.0 / (h_minus + h_plus) * (
        (values[2:] - values[1:-1]) / h_plus
        - (values[1:-1] - values[:-2]) / h_minus
    )


def black_76(S, K, t, r, q, sigma, phi):
    F = S * np.exp((r - q) * t)
    df = np.exp(-r * t)

    d1 = (np.log(F / K) + (0.5 * sigma**2 * t)) / (sigma * np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))
    
    return df * phi * (F * ndtr(phi * d1) - K * ndtr(phi * d2))



def digital(S, K, t, r, q, sigma, phi):
    F = S * np.exp((r - q) * t)
    df = np.exp(-r * t)
    d2 = (np.log(F / K) - (0.5 * sigma**2 * t)) / (sigma * np.sqrt(t))
    
    return df * ndtr(phi * d2)







