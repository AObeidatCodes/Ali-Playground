"""Spectrum processing: put spectra on a common grid, correct the baseline,
and build the derivative form that the library search compares.

The derivative pipeline copies Open Specy's defaults (process_spec):
interpolate to a 5 cm-1 grid, Savitzky-Golay first derivative (window 11,
polynomial 3), absolute value, then min-max normalisation. A derivative
removes sloped and curved baselines without having to model them.
"""
import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

CO2_REGION = (2280.0, 2400.0)


def to_grid(wavenumber, intensity, grid):
    """Linear interpolation onto `grid`; NaN where the spectrum has no data."""
    out = np.interp(grid, wavenumber, intensity)
    out[(grid < wavenumber.min()) | (grid > wavenumber.max())] = np.nan
    return out


def minmax(y):
    lo, hi = np.nanmin(y), np.nanmax(y)
    return (y - lo) / (hi - lo) if hi > lo else y * 0.0


def abs_derivative(y, window=11, polyorder=3):
    """|first derivative| of the finite run of y, min-max normalised."""
    out = np.full_like(y, np.nan, dtype=float)
    valid = np.isfinite(y)
    if valid.sum() <= window:
        return out
    idx = np.flatnonzero(valid)
    run = slice(idx[0], idx[-1] + 1)
    out[run] = np.abs(savgol_filter(y[run], window, polyorder, deriv=1))
    return minmax(out)


def baseline_als(y, lam=1e5, p=0.01, n_iter=10):
    """Asymmetric least squares baseline (Eilers & Boelens, 2005).

    Points above the current baseline get weight p and points below get
    1 - p, so the fit hugs the underside of the spectrum and ignores the bands.
    """
    n = len(y)
    d = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(n, n - 2))
    penalty = lam * d.dot(d.transpose())
    w = np.ones(n)
    for _ in range(n_iter):
        weights = sparse.spdiags(w, 0, n, n)
        z = spsolve((weights + penalty).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def corrected_absorbance(wavenumber, absorbance, lam=1e5, p=0.01):
    """Baseline-corrected absorbance scaled to 0-1, for reading bands by eye."""
    corrected = absorbance - baseline_als(absorbance, lam=lam, p=p)
    return minmax(corrected)


def co2_mask(grid, region=CO2_REGION):
    return (grid >= region[0]) & (grid <= region[1])
