"""Statistical helpers for STAAR computations."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import special, stats

# Try to import numba for JIT acceleration
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


def _normalize_weights(weights: np.ndarray, expected_size: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size != expected_size:
        raise ValueError("The length of weights should be the same as that of the p-values!")
    if np.any(np.isnan(weights)):
        raise ValueError("Cannot have NAs in the weights!")
    if np.any(weights < 0):
        raise ValueError("All the weights must be non-negative!")

    weight_sum = float(np.sum(weights))
    if (not np.isfinite(weight_sum)) or weight_sum <= 0.0:
        raise ValueError("Sum of weights must be positive!")
    return weights / weight_sum


def cct_pval(pvals: np.ndarray, weights: np.ndarray) -> float:
    """Cauchy combination test with weights (matches CCT_pval.cpp)."""
    pvals = np.asarray(pvals, dtype=float).reshape(-1)
    if pvals.size == 0:
        raise ValueError("At least one p-value is required.")
    if np.any(np.isnan(pvals)):
        raise ValueError("Cannot have NAs in the p-values!")
    if np.any(pvals < 0) or np.any(pvals > 1):
        raise ValueError("All p-values must be between 0 and 1!")

    weights = _normalize_weights(weights, expected_size=pvals.size)
    valid = pvals < 1.0
    if not np.any(valid):
        return 1.0
    if not np.all(valid):
        pvals = pvals[valid]
        weights = weights[valid]
        weights = weights / np.sum(weights)

    cct_stat = 0.0
    is_small = pvals < 1e-16
    if np.any(is_small):
        cct_stat += np.sum(weights[is_small] / pvals[is_small] / np.pi)
    if np.any(~is_small):
        cct_stat += np.sum(weights[~is_small] * np.tan((0.5 - pvals[~is_small]) * np.pi))

    if cct_stat > 1e15:
        return (1.0 / cct_stat) / math.pi
    return float(stats.cauchy.sf(cct_stat, loc=0.0, scale=1.0))


def cct(pvals: Iterable[float], weights: Iterable[float] | None = None) -> float:
    """Cauchy combination test (matches CCT.R)."""
    pvals = np.asarray(list(pvals), dtype=float).reshape(-1)
    if pvals.size == 0:
        raise ValueError("At least one p-value is required.")

    if np.any(np.isnan(pvals)):
        raise ValueError("Cannot have NAs in the p-values!")
    if np.any(pvals < 0) or np.any(pvals > 1):
        raise ValueError("All p-values must be between 0 and 1!")

    is_zero = np.any(pvals == 0)
    is_one = np.any(pvals == 1)
    if is_zero and is_one:
        raise ValueError("Cannot have both 0 and 1 p-values!")
    if is_zero:
        return 0.0
    if is_one:
        return 1.0

    if weights is None:
        weights = np.full_like(pvals, 1.0 / pvals.size)
    else:
        weights = _normalize_weights(list(weights), expected_size=pvals.size)

    is_small = pvals < 1e-16
    if not np.any(is_small):
        cct_stat = np.sum(weights * np.tan((0.5 - pvals) * math.pi))
    else:
        cct_stat = np.sum(weights[is_small] / pvals[is_small] / math.pi)
        cct_stat += np.sum(weights[~is_small] * np.tan((0.5 - pvals[~is_small]) * math.pi))

    if cct_stat > 1e15:
        return (1.0 / cct_stat) / math.pi
    return float(stats.cauchy.sf(cct_stat, loc=0.0, scale=1.0))


# =============================================================================
# Numba-accelerated saddle point approximation functions
# =============================================================================

@njit(cache=True, fastmath=True)
def _K_numba(x: float, eigenvals: np.ndarray) -> float:
    """K(x) - Cumulant generating function for eigenvalues."""
    n = len(eigenvals)
    result = 0.0
    for i in range(n):
        result += math.log(1.0 - 2.0 * eigenvals[i] * x)
    return -0.5 * result


@njit(cache=True, fastmath=True)
def _K1_numba(x: float, eigenvals: np.ndarray, q: float) -> float:
    """K'(x) - First derivative."""
    n = len(eigenvals)
    result = 0.0
    for i in range(n):
        result += eigenvals[i] / (1.0 - 2.0 * eigenvals[i] * x)
    return result - q


@njit(cache=True, fastmath=True)
def _K2_numba(x: float, eigenvals: np.ndarray) -> float:
    """K''(x) - Second derivative."""
    n = len(eigenvals)
    result = 0.0
    for i in range(n):
        denom = 1.0 - 2.0 * eigenvals[i] * x
        result += (eigenvals[i] ** 2) / (denom ** 2)
    return 2.0 * result


@njit(cache=True)
def _bisection_numba(eigenvals: np.ndarray, q: float, xmin: float, xmax: float) -> float:
    """Bisection method for finding xhat."""
    xupper = xmax
    xlower = xmin
    x0 = 0.0
    while abs(xupper - xlower) > 1e-8:
        x0 = (xupper + xlower) / 2.0
        k1 = _K1_numba(x0, eigenvals, q)
        if k1 == 0.0:
            break
        if k1 > 0.0:
            xupper = x0
        else:
            xlower = x0
    return x0


@njit(cache=True)
def _saddle_core_numba(q: float, eigenvals: np.ndarray) -> float:
    """Core saddle point computation (without ndtr)."""
    lambdamax = eigenvals[0]
    for i in range(1, len(eigenvals)):
        if eigenvals[i] > lambdamax:
            lambdamax = eigenvals[i]

    q_scaled = q / lambdamax
    eigenvals_scaled = eigenvals / lambdamax

    eigensum = 0.0
    for i in range(len(eigenvals_scaled)):
        eigensum += eigenvals_scaled[i]

    if q_scaled > eigensum:
        xmin = -0.01
    else:
        xmin = -len(eigenvals_scaled) / (2.0 * q_scaled)

    xmax = 0.5 * 0.99999  # 1/(2*lambdamax) with lambdamax=1

    xhat = _bisection_numba(eigenvals_scaled, q_scaled, xmin, xmax)

    k_val = _K_numba(xhat, eigenvals_scaled)
    w_sq = 2.0 * (xhat * q_scaled - k_val)
    if w_sq < 0.0:
        w_sq = 0.0
    w = math.sqrt(w_sq)
    if xhat < 0.0:
        w = -w

    k2_val = _K2_numba(xhat, eigenvals_scaled)
    v = xhat * math.sqrt(k2_val)

    if abs(xhat) < 1e-4:
        return 2.0  # Degenerate case

    z = w + math.log(v / w) / w

    # Return z for ndtr(-z) computation in Python
    return z


def _K(x: float, eigenvals: np.ndarray) -> float:
    return float(-0.5 * np.sum(np.log(1.0 - 2.0 * eigenvals * x)))


def _K1(x: float, eigenvals: np.ndarray, q: float) -> float:
    return float(np.sum(eigenvals / (1.0 - 2.0 * eigenvals * x)) - q)


def _K2(x: float, eigenvals: np.ndarray) -> float:
    return float(2.0 * np.sum((eigenvals**2) / (1.0 - 2.0 * eigenvals * x) ** 2))


def _bisection(eigenvals: np.ndarray, q: float, xmin: float, xmax: float) -> float:
    xupper = xmax
    xlower = xmin
    x0 = 0.0
    while abs(xupper - xlower) > 1e-8:
        x0 = (xupper + xlower) / 2.0
        k1 = _K1(x0, eigenvals, q)
        if k1 == 0:
            break
        if k1 > 0:
            xupper = x0
        else:
            xlower = x0
    return x0


def saddle(q: float, eigenvals: np.ndarray) -> float:
    """Saddle point approximation for p-value calculation."""
    eigenvals = np.asarray(eigenvals, dtype=float)

    if NUMBA_AVAILABLE:
        # Use Numba-accelerated core computation
        eigenvals_c = np.ascontiguousarray(eigenvals)
        z = _saddle_core_numba(q, eigenvals_c)
        if z == 2.0:  # Degenerate case marker
            return 2.0
        return float(special.ndtr(-z))

    # Pure Python fallback
    lambdamax = float(np.max(eigenvals))
    q = q / lambdamax
    eigenvals = eigenvals / lambdamax
    lambdamax = 1.0

    if q > np.sum(eigenvals):
        xmin = -0.01
    else:
        xmin = -len(eigenvals) / (2.0 * q)

    xmax = 1.0 / (2.0 * lambdamax) * 0.99999

    xhat = _bisection(eigenvals, q, xmin, xmax)
    w = math.sqrt(2.0 * (xhat * q - _K(xhat, eigenvals)))
    if xhat < 0:
        w = -w
    v = xhat * math.sqrt(_K2(xhat, eigenvals))

    if abs(xhat) < 1e-4:
        return 2.0

    z = w + math.log(v / w) / w
    return float(special.ndtr(-z))
