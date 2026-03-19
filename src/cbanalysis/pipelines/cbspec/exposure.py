"""
Compute aperture AΩ(E) and exposure λ(E) for each log10(E/eV) bin.

This module implements the standard Telescope Array / Auger convention:

    AΩ(E) = (N_MC_reco(E) / N_MC_raw(E)) × (A_gen × Ω_gen)

    λ(E) = AΩ(E) × T

where:
    - N_MC_reco(E)  = number of reconstructed MC events in bin E
    - N_MC_raw(E)   = number of thrown MC events in bin E
    - A_gen         = generated area (m²)
    - Ω_gen         = generated solid angle (sr)
    - T             = experimental live time (s)

The acceptance fraction (N_MC_reco(E) / N_MC_raw(E)) is the key physics quantity that
encodes reconstruction efficiency and quality-cut survival probability.
"""


import numpy as np


# Aperture
def compute_aperture(mc_counts, mc_raw_counts, generated_area_m2, generated_solid_angle_sr):
    """
    Compute the detector aperture like in Dmitri Ivanov's thesis.

    :param mc_counts: array-like
                      Reconstructed MC counts per bin (after quality cuts)
    :param mc_raw_counts: array-like
                          Thrown MC counts per bin
    :param generated_area_m2: float
                              A_gen -- MC generated area in m²
    :param generated_solid_angle_sr: float
                                     Ω_gen -- MC generated solid angle in sr
    :return aperture: np.ndarray
                      Aperture per bin [m² sr]

    Notes:
        - The acceptance fraction is:
            frac(E) = N_MC_reco(E) / N_MC_raw(E)
        - Bins with N_MC_raw = 0 or 1 are masked upstream in binning+filter_bins,
          but we still guard against division-by-zero here
        - Units:
            AΩ(E) = [dimensionless] × [m²] × [sr]
                  = [m² sr]
    """
    mc_counts = np.asarray(mc_counts, dtype=float)
    mc_raw_counts = np.asarray(mc_raw_counts, dtype=float)

    # Initialize acceptance fraction
    frac = np.zeros_like(mc_counts, dtype=float)

    # Safe division: only compute where N_MC_raw(E) > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mc_raw_counts > 0
        frac[mask] = mc_counts[mask] / mc_raw_counts[mask]

    # Compute aperture
    aperture = frac * generated_area_m2 * generated_solid_angle_sr
    return aperture


# Exposure
def compute_exposure(aperture, time_s):
    """
    Compute the exposure like in Dmitri Ivanov's thesis.

    :param aperture: array-like
                     Aperture per bin [m² sr]
    :param time_s: float
                   Experimental live time in seconds
    :return exposure: np.ndarray
                      Exposure per bin [m² sr s]

    Notes:
        - Exposure is effective collecting power of the detector
        - Units:
            λ(E) = [m² sr] × [s]
                 = [m² sr s]
    """
    aperture = np.asarray(aperture, dtype=float)
    return aperture * time_s