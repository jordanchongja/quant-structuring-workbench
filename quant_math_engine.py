import numpy as np
import math
import warnings
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import brentq
from scipy.stats import norm

# ==============================================================================
# 1. BLACK-SCHOLES & IMPLIED VOLATILITY
# ==============================================================================
def bs_call_price(sigma, S, K, T, r, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(target_price, S, K, T, r, q):
    intrinsic = max(S * np.exp(-q*T) - K * np.exp(-r*T), 0)
    if target_price <= intrinsic:
        return np.nan 

    def objective(sigma):
        return bs_call_price(sigma, S, K, T, r, q) - target_price

    try:
        return brentq(objective, 1e-4, 5.0) 
    except ValueError:
        return np.nan

# ==============================================================================
# 2. HESTON MODEL (STOCHASTIC VOLATILITY)
# ==============================================================================
def heston_characteristic_function(u, S0, K, T, r, q, v0, kappa, theta, xi, rho):
    alpha = -u**2 / 2 - 1j * u / 2
    beta = kappa - rho * xi * 1j * u
    gamma = xi**2 / 2
    
    d = np.sqrt(beta**2 - 4 * alpha * gamma)
    r_plus = (beta + d) / (xi**2)
    r_minus = (beta - d) / (xi**2)
    g = r_minus / r_plus
    
    C = kappa * (r_minus * T - (2 / (xi**2)) * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D = r_minus * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    
    return np.exp(C * theta + D * v0 + 1j * u * np.log(S0 * np.exp((r - q) * T)))

def heston_call_price(S0, K, T, r, q, v0, kappa, theta, xi, rho):
    def integrand1(u):
        cf = heston_characteristic_function(u - 1j, S0, K, T, r, q, v0, kappa, theta, xi, rho)
        return (np.exp(-1j * u * np.log(K)) * cf / (1j * u * S0 * np.exp((r-q)*T))).real

    def integrand2(u):
        cf = heston_characteristic_function(u, S0, K, T, r, q, v0, kappa, theta, xi, rho)
        return (np.exp(-1j * u * np.log(K)) * cf / (1j * u)).real

    limit_max = 2000 # Gibbs phenomenon fix
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        P1_int = quad(integrand1, 0, limit_max, epsabs=1e-4, epsrel=1e-4, limit=200)[0]
        P2_int = quad(integrand2, 0, limit_max, epsabs=1e-4, epsrel=1e-4, limit=200)[0]
        
    P1 = 0.5 + (1 / np.pi) * P1_int
    P2 = 0.5 + (1 / np.pi) * P2_int
    
    return max(0.0, S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2)

# ==============================================================================
# 3. MERTON MODEL (JUMP DIFFUSION)
# ==============================================================================
def merton_jump_call(S, K, T, r, q, sigma, lam, mu_j, delta, max_jumps=40):
    k = np.exp(mu_j + 0.5 * delta**2) - 1
    lambda_prime = lam * (1 + k)
    price = 0.0
    
    for n in range(max_jumps):
        poisson_weight = np.exp(-lambda_prime * T) * ((lambda_prime * T)**n) / math.factorial(n)
        sigma_n = np.sqrt(sigma**2 + (n * delta**2) / T)
        r_n = r - lam * k + (n * np.log(1 + k)) / T
        
        price += poisson_weight * bs_call_price(sigma_n, S, K, T, r_n, q)
        
    return price

# ==============================================================================
# 4. BATES MODEL (STOCHASTIC VOLATILITY + JUMPS)
# ==============================================================================
def bates_characteristic_function(u, S0, K, T, r, q, v0, kappa, theta, xi, rho, lam, mu_j, delta):
    # Heston Component
    alpha = -u**2 / 2 - 1j * u / 2
    beta = kappa - rho * xi * 1j * u
    gamma = xi**2 / 2
    
    d = np.sqrt(beta**2 - 4 * alpha * gamma)
    r_plus = (beta + d) / (xi**2)
    r_minus = (beta - d) / (xi**2)
    g = r_minus / r_plus
    
    C = kappa * (r_minus * T - (2 / (xi**2)) * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D = r_minus * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    heston_cf = np.exp(C * theta + D * v0 + 1j * u * np.log(S0 * np.exp((r - q) * T)))
    
    # Merton Component
    k = np.exp(mu_j + 0.5 * delta**2) - 1 
    jump_term = np.exp(mu_j * 1j * u - 0.5 * delta**2 * u**2) - 1
    merton_cf = np.exp(lam * T * (jump_term - 1j * u * k))
    
    return heston_cf * merton_cf

def bates_call_price(S0, K, T, r, q, v0, kappa, theta, xi, rho, lam, mu_j, delta):
    def integrand1(u):
        cf = bates_characteristic_function(u - 1j, S0, K, T, r, q, v0, kappa, theta, xi, rho, lam, mu_j, delta)
        return (np.exp(-1j * u * np.log(K)) * cf / (1j * u * S0 * np.exp((r-q)*T))).real

    def integrand2(u):
        cf = bates_characteristic_function(u, S0, K, T, r, q, v0, kappa, theta, xi, rho, lam, mu_j, delta)
        return (np.exp(-1j * u * np.log(K)) * cf / (1j * u)).real

    limit_max = 2000 
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        P1_int = quad(integrand1, 0, limit_max, epsabs=1e-4, epsrel=1e-4, limit=200)[0]
        P2_int = quad(integrand2, 0, limit_max, epsabs=1e-4, epsrel=1e-4, limit=200)[0]
        
    P1 = 0.5 + (1 / np.pi) * P1_int
    P2 = 0.5 + (1 / np.pi) * P2_int
    
    return max(0.0, S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2)