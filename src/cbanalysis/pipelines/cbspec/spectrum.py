"""
Convert differential flux J(E) into energy-scaled spectrum:

    S(E) = E³ J(E)

This is standard representation used by Telescope Array, Auger, and HiRes
because multiplying by E³ visually flattens the spectrum and highlights
features such as the ankle and GZK suppression.

All computations are performed in log10(E/eV) space, constants with the rest
of the pipeline.

add: fitting to HiRes spectrum
"""

import numpy as np

def flux_to_spectrum(centers_ev, flux, flux_lower, flux_upper):
    """
    Convert flux J(E) into spectrum S(E) = E³ J(E).

    :param centers_ev: array-like
                       Energy bin centers in eV
    :param flux: array-like
                 Differential flux J(E) in each bin
    :param flux_lower: array-like
                     Lower Feldman_Cousins bound on J(E)
    :param flux_upper: array-like
                      Upper Feldman-Cousins bound on J(E)
    :return spectrum: np.ndarray
                      Energy-scaled spectrum S(E) = E³ J(E)
    :return spectrum_lower: np.ndarray
                          Lower Feldman_Cousins bound in spectrum space
    :return spectrum_upper: np.ndarray
                           Upper Feldman_Cousins bound in spectrum space

    Notes:
        - Units:
            J(E)    ~ m⁻² sr⁻¹ s⁻¹ eV⁻¹
            E³J(E)  ~ eV² m⁻² sr⁻¹ s⁻¹

    These match the conventions used in TA and HiRes publications.
    """
    # Compute E³
    e3 = centers_ev**3

    # Apply scaling
    spectrum = e3 * flux
    spectrum_lower = e3 * flux_lower
    spectrum_upper = e3 * flux_upper

    return spectrum, spectrum_lower, spectrum_upper