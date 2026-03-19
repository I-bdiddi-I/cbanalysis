"""
Compute the differential flux J(E) in log10(E/eV) bins.

This module implements the standard Telescope Array / Auger convention:

    J(E) ≈ N_data(E) / ( Exposure(E) × Δlog10(E) )

where:
    - N_data(E)     = reconstructed data counts in the energy bin E
    - Exposure(E)   = AΩ(E) × T  [m² sr s]
    - Δlog10(E)     = bin width in log10(E/eV)
    - J(E)          = differential flux [m⁻² sr⁻¹ s⁻¹ eV⁻¹]
                      up to the binning convention

The Feldman-Cousins confidence intervals are computed in a separate module
(feldman_cousins.py) and applied after this step.
"""


import numpy as np


def compute_flux(n_events, exposure, delta_ev):
    """
    Compute the differential flux J(E) in each log10(E/eV) bin.

    :param n_events: array-like
                     Data counts per bin (after quality cuts and filtering)
    :param exposure: array-like
                     Exposure per bin [m² sr s], computed as:
                        Exposure(E) = AΩ(E) × run_time_s
    :param delta_ev: array-like
                     Width of each energy bin in eV
    :return flux: np.ndarray
                  Differential flux J(E) in each bin
                  Units:
                    J(E)≈ [counts] / ([m² sr s] × Δlog10(E))
                    which corresponds to:
                    [m⁻² sr⁻¹ s⁻¹ eV⁻¹] up to the log-binning convention.

    Notes:
        - Division-by-zero is safely handled using np.where
        - Bins with zero exposure or zero width return flux = 0
    """
    n_events = np.asarray(n_events, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    delta_ev = np.asarray(delta_ev, dtype=float)

    with np.errstate(divide='ignore', invalid='ignore'):
        flux = np.where(
            (exposure > 0) & (delta_ev > 0),
            n_events / (exposure * delta_ev),
            0.,
        )
    return flux

