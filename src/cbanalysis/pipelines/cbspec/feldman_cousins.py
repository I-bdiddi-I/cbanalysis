"""
Feldman-Cousins confidence intervals using the NIST FCpy package.

This module provides a thin wrapper around FCpy API:

    FC.FC_poisson(n0, b, t, conf=0.95, useCorrection=False, ...)

For cbspec pipeline:
    - Each energy bin is treated as a Poisson process
    - Background is assumed to be zero (b = 0)
    - The exposure scaling is handled elsewhere, so we set t = 1
    - Confidence intervals are computed directly on counts
"""

import numpy as np
from FCpy import FC # FCpy package form GitHub

def feldman_cousins_interval(n_obs, cl=0.68, use_correction=False):
    """
    Compute Feldman-Cousins confidence intervals for a single observed count.

    :param n_obs: int
                  Observed number of events in an energy bin
    :param cl: float, optional
               Confidence level (default 0.68)
    :param use_correction: bool, optional
                           Whether to apply the Roe & Woodroofe correction
    :return mu_low: float
                    Lower confidence interval on the true mean (in counts)
    :return mu_high: float
                     Upper confidence interval on the true mean (in counts)
    """
    ci = FC.FC_poisson(
        n0=int(n_obs),
        b=0.,
        t=1.,
        conf=float(cl),
        useCorrection=bool(use_correction),
    )
    return float(ci[0]), float(ci[1])

def feldman_cousins_vector(counts, cl=0.68, use_correction=False):
    """
    Vectorized Feldman-Cousins confidence intervals for an array of observed counts.

    :param counts: array-like
                   Observed counts per energy bin
    :param cl: float, optional
               Confidence level (default 0.68)
    :param use_correction: bool, optional
                           Whether to apply the Roe & Woodroofe correction
    :return mu_low: np.ndarray
                    Array of lower FC limit on counts
    :return mu_high: np.ndarray
                     Array of upper FC limit on counts
    """
    counts = np.asarray(counts, dtype=int)

    mu_low = np.zeros_like(counts, dtype=float)
    mu_high = np.zeros_like(counts, dtype=float)

    for i, n in enumerate(counts):
        low, high = feldman_cousins_interval(
            n_obs=n,
            cl=cl,
            use_correction=use_correction,
        )
        mu_low[i] = low
        mu_high[i] = high

    return mu_low, mu_high