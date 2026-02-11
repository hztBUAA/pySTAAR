"""Numba-accelerated SPA (Saddlepoint Approximation) functions.

This module provides JIT-compiled versions of the K-function family and
Newton-Raphson solver used in Binary SPA calculations. When numba is available,
these functions offer 2-5x speedup over the pure NumPy implementations.

Usage:
    from pystaar._spa_numba import (
        k_binary_spa_numba,
        k1_binary_spa_numba,
        k2_binary_spa_numba,
        nr_binary_spa_numba,
        NUMBA_AVAILABLE,
    )
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# Try to import numba, fall back to no-op decorators if unavailable
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Provide no-op decorator fallbacks
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    def prange(*args, **kwargs):
        return range(*args)


# =============================================================================
# K-function family (Cumulant Generating Function and derivatives)
# =============================================================================

@njit(cache=True, fastmath=True)
def _clip_scalar(x: float, lo: float, hi: float) -> float:
    """Clip a scalar value to [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@njit(cache=True, fastmath=True)
def k_binary_spa_numba(x: float, muhat: np.ndarray, g: np.ndarray) -> float:
    """K(x) - Cumulant generating function.

    Computes: sum(-x * muhat * g + log1p(-muhat + muhat * exp(clip(x * g))))
    """
    n = len(muhat)
    result = 0.0
    for i in range(n):
        xg = _clip_scalar(x * g[i], -700.0, 700.0)
        exp_xg = math.exp(xg)
        term = -x * muhat[i] * g[i] + math.log1p(-muhat[i] + muhat[i] * exp_xg)
        result += term
    return result


@njit(cache=True, fastmath=True)
def k_binary_spa_alt_numba(x: float, muhat: np.ndarray, g: np.ndarray) -> float:
    """K(x) - Alternative implementation for numerical stability."""
    n = len(muhat)
    result = 0.0
    for i in range(n):
        xg = _clip_scalar(-x * g[i], -700.0, 700.0)
        exp_xg = math.exp(xg)
        term = math.log((1.0 - muhat[i]) * exp_xg + muhat[i])
        result += term
    return result


@njit(cache=True, fastmath=True)
def k1_binary_spa_numba(x: float, muhat: np.ndarray, g: np.ndarray, q: float) -> float:
    """K'(x) - First derivative of cumulant generating function.

    Computes: sum(-muhat * g + muhat * g / (muhat + (1 - muhat) * exp(-x * g))) - q
    """
    n = len(muhat)
    result = 0.0
    for i in range(n):
        xg = _clip_scalar(-x * g[i], -700.0, 700.0)
        exp_xg = math.exp(xg)
        denom = muhat[i] + (1.0 - muhat[i]) * exp_xg
        term = muhat[i] * g[i] / denom - muhat[i] * g[i]
        result += term
    return result - q


@njit(cache=True, fastmath=True)
def k1_binary_spa_alt_numba(x: float, muhat: np.ndarray, g: np.ndarray, q: float) -> float:
    """K'(x) - Alternative implementation for numerical stability."""
    n = len(muhat)
    result = 0.0
    for i in range(n):
        xg = _clip_scalar(x * g[i], -700.0, 700.0)
        exp_xg = math.exp(xg)
        denom = muhat[i] * exp_xg + (1.0 - muhat[i])
        term = muhat[i] * g[i] * exp_xg / denom - muhat[i] * g[i]
        result += term
    return result - q


@njit(cache=True, fastmath=True)
def k2_binary_spa_numba(x: float, muhat: np.ndarray, g: np.ndarray) -> float:
    """K''(x) - Second derivative of cumulant generating function.

    Computes: sum(muhat * (1 - muhat) * g^2 * exp(-x*g) / (muhat + (1-muhat)*exp(-x*g))^2)
    """
    n = len(muhat)
    result = 0.0
    for i in range(n):
        xg = _clip_scalar(-x * g[i], -700.0, 700.0)
        exp_xg = math.exp(xg)
        g_sq = g[i] * g[i]
        num = muhat[i] * (1.0 - muhat[i]) * g_sq * exp_xg
        denom = muhat[i] + (1.0 - muhat[i]) * exp_xg
        denom_sq = denom * denom
        result += num / denom_sq
    return result


@njit(cache=True, fastmath=True)
def k2_binary_spa_alt_numba(x: float, muhat: np.ndarray, g: np.ndarray) -> float:
    """K''(x) - Alternative implementation for numerical stability."""
    n = len(muhat)
    result = 0.0
    for i in range(n):
        xg = _clip_scalar(x * g[i], -700.0, 700.0)
        exp_xg = math.exp(xg)
        g_sq = g[i] * g[i]
        num = muhat[i] * (1.0 - muhat[i]) * g_sq
        denom = muhat[i] * exp_xg + (1.0 - muhat[i])
        denom_sq = denom * denom
        result += num / denom_sq
    return result


# =============================================================================
# Newton-Raphson solver
# =============================================================================

@njit(cache=True)
def _is_bad_number_numba(x: float) -> bool:
    """Check if a number is NaN or Inf."""
    return not math.isfinite(x)


@njit(cache=True)
def nr_binary_spa_numba(
    muhat: np.ndarray,
    g: np.ndarray,
    q: float,
    init: float,
    tol: float,
    max_iter: int,
) -> float:
    """Newton-Raphson solver for finding xhat in SPA.

    This is the main performance hotspot - called for every score test.
    """
    xi = init
    xi_update = init

    # Initial step
    k1 = k1_binary_spa_numba(xi, muhat, g, q)
    if abs(k1) > tol:
        k2 = k2_binary_spa_numba(xi, muhat, g)
        if k2 != 0.0:
            xi_update = xi - k1 / k2

    no_iter = 0
    while (
        math.isfinite(xi_update)
        and abs(xi_update - xi) > tol
        and abs(k1_binary_spa_numba(xi_update, muhat, g, q)) > tol
        and no_iter < max_iter
    ):
        no_iter += 1
        xi = xi_update

        # Compute numerator (first derivative)
        numerator = k1_binary_spa_numba(xi, muhat, g, q)
        if _is_bad_number_numba(numerator):
            numerator = k1_binary_spa_alt_numba(xi, muhat, g, q)

        # Compute denominator (second derivative)
        denominator = k2_binary_spa_numba(xi, muhat, g)
        if _is_bad_number_numba(denominator):
            denominator = k2_binary_spa_alt_numba(xi, muhat, g)

        if denominator != 0.0:
            xi_update = xi - numerator / denominator

    if _is_bad_number_numba(xi_update):
        xi_update = xi

    # Handle max iterations case
    if no_iter == max_iter:
        xhat = xi_update
        k_val = k_binary_spa_numba(xhat, muhat, g)
        w_sq = max(0.0, 2.0 * (xhat * q - k_val))
        w = math.sqrt(w_sq)
        if xhat < 0.0:
            w = -w

        k2_val = k2_binary_spa_numba(xhat, muhat, g)
        if _is_bad_number_numba(k2_val):
            k2_val = k2_binary_spa_alt_numba(xhat, muhat, g)
        ki = xhat * math.sqrt(max(0.0, k2_val))

        if w != 0.0:
            ratio = ki / w
            if ratio > 0.0:
                z = w + math.log(ratio) / w
                if abs(z) < 38.0:
                    xi_update = 0.0

    return xi_update


# =============================================================================
# Bisection solver (fallback)
# =============================================================================

@njit(cache=True)
def _have_same_sign_numba(a: float, b: float) -> bool:
    """Check if two numbers have the same sign."""
    return (a >= 0.0) == (b >= 0.0)


@njit(cache=True)
def bisection_binary_spa_numba(
    muhat: np.ndarray,
    g: np.ndarray,
    q: float,
    xmin: float,
    xmax: float,
    tol: float,
) -> float:
    """Bisection method for finding xhat when Newton-Raphson fails."""
    a = xmin
    b = xmax

    k1a = k1_binary_spa_numba(a, muhat, g, q)
    if _is_bad_number_numba(k1a):
        k1a = k1_binary_spa_alt_numba(a, muhat, g, q)

    k1b = k1_binary_spa_numba(b, muhat, g, q)
    if _is_bad_number_numba(k1b):
        k1b = k1_binary_spa_alt_numba(b, muhat, g, q)

    while abs(b - a) > tol:
        mid = 0.5 * (a + b)
        k1mid = k1_binary_spa_numba(mid, muhat, g, q)
        if _is_bad_number_numba(k1mid):
            k1mid = k1_binary_spa_alt_numba(mid, muhat, g, q)

        if abs(k1mid) < tol:
            return mid

        if _have_same_sign_numba(k1a, k1mid):
            a = mid
            k1a = k1mid
        else:
            b = mid
            k1b = k1mid

    return 0.5 * (a + b)


@njit(cache=True)
def golden_section_search_sign_change_numba(
    a: float,
    b: float,
    muhat: np.ndarray,
    g: np.ndarray,
    q: float,
    tol: float,
    max_iter: int,
) -> Tuple[float, float]:
    """Golden section search for sign change interval."""
    x0 = a
    x1 = b
    iter_count = 0
    phi = (1.0 + math.sqrt(5.0)) / 2.0

    k1x0 = k1_binary_spa_numba(x0, muhat, g, q)
    if _is_bad_number_numba(k1x0):
        k1x0 = k1_binary_spa_alt_numba(x0, muhat, g, q)

    k1x1 = k1_binary_spa_numba(x1, muhat, g, q)
    if _is_bad_number_numba(k1x1):
        k1x1 = k1_binary_spa_alt_numba(x1, muhat, g, q)

    while iter_count < max_iter and abs(x1 - x0) > tol and _have_same_sign_numba(k1x0, k1x1):
        x1 = b - (b - a) / phi
        x0 = a + (b - a) / phi

        k1x0 = k1_binary_spa_numba(x0, muhat, g, q)
        if _is_bad_number_numba(k1x0):
            k1x0 = k1_binary_spa_alt_numba(x0, muhat, g, q)

        k1x1 = k1_binary_spa_numba(x1, muhat, g, q)
        if _is_bad_number_numba(k1x1):
            k1x1 = k1_binary_spa_alt_numba(x1, muhat, g, q)

        if k1x1 < k1x0:
            b = x0
        else:
            a = x1

        iter_count += 1

    return x0, x1


# =============================================================================
# High-level SPA functions
# =============================================================================

@njit(cache=True)
def saddle_binary_spa_numba(
    q: float,
    muhat: np.ndarray,
    g: np.ndarray,
    tol: float,
    max_iter: int,
    lower: bool,
) -> float:
    """Saddlepoint approximation for binary SPA p-value calculation."""
    xhat = nr_binary_spa_numba(muhat, g, q, 0.0, tol, max_iter)

    k_val = k_binary_spa_numba(xhat, muhat, g)
    w_sq = max(0.0, 2.0 * (xhat * q - k_val))
    w = math.sqrt(w_sq)
    if xhat < 0.0:
        w = -w

    k2_val = k2_binary_spa_numba(xhat, muhat, g)
    if _is_bad_number_numba(k2_val):
        k2_val = k2_binary_spa_alt_numba(xhat, muhat, g)

    ki = xhat * math.sqrt(max(0.0, k2_val))

    if w == 0.0:
        return 0.5  # No evidence either way

    ratio = ki / w
    if ratio <= 0.0:
        return 0.5

    z = w + math.log(ratio) / w

    # Use normal CDF approximation
    # For Numba compatibility, use error function
    if lower:
        # P(Z < z) = 0.5 * (1 + erf(z / sqrt(2)))
        pval = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    else:
        # P(Z > z) = 0.5 * (1 - erf(z / sqrt(2)))
        pval = 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))

    return pval


@njit(cache=True)
def burden_spa_two_sided_pvalue_numba(
    score: float,
    muhat: np.ndarray,
    g_col: np.ndarray,
    tol: float,
    max_iter: int,
    xmin: float,
    xmax: float,
) -> float:
    """Compute two-sided p-value using SPA for burden test."""
    if abs(score) < 1e-15:
        return 1.0

    # Upper tail
    p_upper = saddle_binary_spa_numba(score, muhat, g_col, tol, max_iter, False)

    # Lower tail
    p_lower = saddle_binary_spa_numba(-score, muhat, g_col, tol, max_iter, True)

    pval = p_upper + p_lower

    # Clamp to valid probability range
    if pval < 0.0:
        pval = 0.0
    if pval > 1.0:
        pval = 1.0

    return pval


# =============================================================================
# Batch processing for burden SPA (reduces Python/Numba boundary overhead)
# =============================================================================

@njit(cache=True, parallel=True)
def staartest_burden_binary_spa_batch_numba(
    scores: np.ndarray,
    muhat: np.ndarray,
    G_cumu: np.ndarray,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """Batch compute burden SPA p-values using parallel processing.

    This function processes all weight columns in parallel, reducing
    the Python/Numba boundary crossing overhead from O(wn) to O(1).

    Args:
        scores: Array of burden scores (length wn)
        muhat: Predicted probabilities (length n)
        G_cumu: Cumulative genotype matrix (n x wn)
        tol: Convergence tolerance
        max_iter: Maximum iterations for Newton-Raphson

    Returns:
        Array of p-values (length wn)
    """
    wn = len(scores)
    res = np.ones(wn, dtype=np.float64)

    for i in prange(wn):
        score = scores[i]
        g_col = G_cumu[:, i].copy()  # Ensure contiguous

        if abs(score) < 1e-15:
            res[i] = 1.0
            continue

        q_abs = abs(score)

        # Upper tail: P(X > |score|)
        p_upper = _saddle_pvalue_numba(q_abs, muhat, g_col, tol, max_iter, False)

        # Lower tail: P(X < -|score|)
        p_lower = _saddle_pvalue_numba(-q_abs, muhat, g_col, tol, max_iter, True)

        pval = p_upper + p_lower
        if pval < 0.0:
            pval = 0.0
        if pval > 1.0:
            pval = 1.0

        res[i] = pval

    return res


@njit(cache=True)
def _saddle_pvalue_numba(
    q: float,
    muhat: np.ndarray,
    g: np.ndarray,
    tol: float,
    max_iter: int,
    lower: bool,
) -> float:
    """Internal saddlepoint p-value calculation with fallback handling."""
    xhat = nr_binary_spa_numba(muhat, g, q, 0.0, tol, max_iter)

    k_val = k_binary_spa_numba(xhat, muhat, g)
    if _is_bad_number_numba(k_val):
        k_val = k_binary_spa_alt_numba(xhat, muhat, g)

    w_sq = max(0.0, 2.0 * (xhat * q - k_val))
    w = math.sqrt(w_sq)
    if xhat < 0.0:
        w = -w

    k2_val = k2_binary_spa_numba(xhat, muhat, g)
    if _is_bad_number_numba(k2_val):
        k2_val = k2_binary_spa_alt_numba(xhat, muhat, g)

    ki = xhat * math.sqrt(max(0.0, k2_val))

    # Check for degenerate cases - return 1.0 to trigger bisection fallback
    if abs(xhat) < 1e-4 or w == 0.0:
        return 1.0

    ratio = ki / w
    if ratio <= 0.0 or not math.isfinite(ratio):
        return 1.0

    z = w + math.log(ratio) / w

    # Normal CDF using error function
    if lower:
        pval = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    else:
        pval = 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))

    return pval


# =============================================================================
# Warmup function to trigger JIT compilation
# =============================================================================

def warmup_spa_jit():
    """Pre-compile all JIT functions with representative data.

    Call this at module load time to avoid JIT compilation overhead
    during actual computation.
    """
    if not NUMBA_AVAILABLE:
        return

    # Create small representative arrays
    n = 100
    muhat = np.random.uniform(0.1, 0.9, n)
    g = np.random.randn(n)
    q = 0.5

    # Trigger compilation of all functions
    _ = k_binary_spa_numba(0.1, muhat, g)
    _ = k_binary_spa_alt_numba(0.1, muhat, g)
    _ = k1_binary_spa_numba(0.1, muhat, g, q)
    _ = k1_binary_spa_alt_numba(0.1, muhat, g, q)
    _ = k2_binary_spa_numba(0.1, muhat, g)
    _ = k2_binary_spa_alt_numba(0.1, muhat, g)
    _ = nr_binary_spa_numba(muhat, g, q, 0.0, 1e-6, 100)
    _ = bisection_binary_spa_numba(muhat, g, q, -1.0, 1.0, 1e-6)
    _ = golden_section_search_sign_change_numba(-1.0, 1.0, muhat, g, q, 1e-6, 100)
    _ = saddle_binary_spa_numba(q, muhat, g, 1e-6, 100, True)
    _ = burden_spa_two_sided_pvalue_numba(q, muhat, g, 1e-6, 100, -1.0, 1.0)

    # Warmup batch processing function
    wn = 5
    scores = np.random.randn(wn)
    G_cumu = np.random.randn(n, wn)
    _ = staartest_burden_binary_spa_batch_numba(scores, muhat, G_cumu, 1e-6, 100)
