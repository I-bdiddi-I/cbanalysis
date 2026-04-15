"""
Efficiency computation utilities for the cbefficiency pipeline.

This module contains only the physics logic:
    - computing efficiency per energy bin
    - fitting a smooth efficiency curve
    - detecting where the efficiency curve levels off (plateau energy)
"""

import numpy as np
from scipy.optimize import curve_fit


# Basic efficiency computation + binomial error
def compute_efficiency_with_error(full_counts, geom_counts):
    """
    Compute efficiency and binomial error per energy bin.

    Efficiency is defined as:
        eff = N_fullcuts / N_geomcuts

    Binomial error is defined as:
        sigma = sqrt(eff * (1 - eff) / N_geomcuts)

    :param full_counts: array-like
                        Count per bin of MC thrown energies that passed full reconstruction + full quality cuts
    :param geom_counts: array-like
                        Count per bin of MC thrown energies that passed full reconstruction + geometry cuts

    :return eff: np.ndarray
                 Efficiency per energy bin.
    :return bi_err: np.ndarray
                    Binomial error per energy bin.
    """
    full_counts = np.asarray(full_counts, dtype=float)
    geom_counts = np.asarray(geom_counts, dtype=float)

    eff = np.zeros_like(full_counts, dtype=float)
    bi_err = np.zeros_like(full_counts, dtype=float)

    for i in range(len(full_counts)):
        n = full_counts[i]
        k = geom_counts[i]

        if k <= 0:
            eff[i] = 0.0
            bi_err[i] = 0.0
            continue

        e = n / k
        err = np.sqrt(e * (1 - e) / k)

        eff[i] = e
        bi_err[i] = err

    return eff, bi_err


# Logistic model + fitting
def _logistic(E, k, E0):
    """
    Simple logistic model for efficiency:

        f(E) = 1 / (1 + exp(-k * (E - E0)))

    :param E: array-like
              log10(E/eV) bin centers
    :param k: float
              Steepness parameter
    :param E0: float
               Midpoint (energy where f = 0.5)

    :return f: np.ndarray
               Modeled efficiency values
    """
    E = np.asarray(E, dtype=float)
    return 1.0 / (1.0 + np.exp(-k * (E - E0)))


def fit_efficiency_curve(centers, eff, bi_err):
    """
    Fit a logistic curve to the efficiency data using binomial errors as weights.

    :param centers: array-like
                    log10(E/eV) bin centers
    :param eff: array-like
                Efficiency per energy bin
    :param bi_err: array-like
                   Binomial error per energy bin

    :return params: dict
                    {"k": steepness, "E0": midpoint}
    :return model_eff: np.ndarray
                       Logistic model evaluated at centers
    """
    centers = np.asarray(centers, dtype=float)
    eff = np.asarray(eff, dtype=float)
    bi_err = np.asarray(bi_err, dtype=float)

    # Avoid zero weights
    bi_err = np.where(bi_err > 0, bi_err, 1.0)

    # Initial guesses
    try:
        idx_mid = np.argmin(np.abs(eff - 0.5))
        E0_guess = centers[idx_mid]
    except Exception:
        E0_guess = np.median(centers)

    k_guess = 5.0

    try:
        popt, _ = curve_fit(
            _logistic,
            centers,
            eff,
            sigma=bi_err,
            absolute_sigma=True,
            p0=[k_guess, E0_guess],
            bounds=([0.01, centers.min()], [100.0, centers.max()]),
            maxfev=10000,
        )
        k_fit, E0_fit = popt
    except Exception:
        k_fit, E0_fit = k_guess, E0_guess

    model_eff = _logistic(centers, k_fit, E0_fit)

    return {"k": k_fit, "E0": E0_fit}, model_eff


# Derivatives of logistic model
def logistic_derivative(E, k, E0):
    """
    First derivative of logistic efficiency curve.

    Math:
    f(E) = 1 / (1 + exp(-k * (E - E0))) = (1 + exp(-k * (E - E0)))^(-1)

    => f'(E) = df/dE = (k * exp(-k * (E - E0))) / (1 + exp(-k * (E - E0)))^(2)
                     = k * f(E) * (1 - f(E))
    """
    f = _logistic(E, k, E0)
    return k * f * (1.0 - f)

def logistic_second_derivative(E, k, E0):
    """
    Second derivative of logistic efficiency curve.

    Math:
    f(E) = 1 / (1 + exp(-k * (E - E0))) = (1 + exp(-k * (E - E0))) ^ (-1)

    = > f''(E) = d/dE(df/dE)
               = d/dE(k * f(E) * (1 - f(E)))
               = k * [df/dE * (1 - f(E)) + f * d/dE(1 - f(E))]
               = k * [k * f(E) * (1 - f(E)) * (1 - f(E)) - f(E) * (k * f(E) * (1 - f(E))]
               = k^2 * f(E) * (1 - f(E)) * [(1 - f(E)) - f(E)]
               = k^2 * f(E) * (1 - f(E)) * (1 - 2 * f(E))
    """
    f = _logistic(E, k, E0)
    return (k ** 2) * f * (1.0 - f) * (1.0 - 2.0 * f)


# Plateau detection methods
def plateau_by_derivative_threshold(centers, params, threshold=0.01):
    """
    Method A: plateau where |f'(E)| < threshold.

    :param centers: array-like
                    log10(E/eV) bin centers
    :param params: dict
                   Logistic parameters {"k": steepness, "E0": midpoint}
    :param threshold: float
                      Absolute derivative threshold

    :return E_plateau: float or None
                       Energy where derivative first falls below threshold
                       None if no such point is found
    """
    centers = np.asarray(centers, dtype=float)
    k, E0 = params["k"], params["E0"]
    deriv = np.abs(logistic_derivative(centers, k, E0))
    mask = deriv < threshold
    return centers[mask][0] if np.any(mask) else None


def plateau_by_curvature_threshold(centers, params, threshold=0.001):
    """
    Method B: plateau where |f''(E)| < threshold.

    :param centers: array-like
                    log10(E/eV) bin centers
    :param params: dict
                   Logistic parameters {"k": steepness, "E0": midpoint}
    :param threshold: float
                      Absolute curvature threshold

    :return E_plateau: float or None
                       Energy where curvature first falls below threshold
                       None if no such point is found
    """
    centers = np.asarray(centers, dtype=float)
    k, E0 = params["k"], params["E0"]
    curv = np.abs(logistic_second_derivative(centers, k, E0))
    mask = curv < threshold
    return centers[mask][0] if np.any(mask) else None


def plateau_by_fraction_of_max_derivative(centers, params, fraction=0.05):
    """
    Method C: plateau where |f'(E)| < fraction * max|f'(E)|.

    :param centers: array-like
                    log10(E/eV) bin centers
    :param params: dict
                   Logistic parameters {"k": steepness, "E0": midpoint}
    :param fraction: float
                     Fraction of maximum derivative (e.g., 0.05 = 5%)

    :return E_plateau: float or None
                       Energy where derivative first falls below fraction * max derivative
                       None if no such point is found
    """
    centers = np.asarray(centers, dtype=float)
    k, E0 = params["k"], params["E0"]
    deriv = np.abs(logistic_derivative(centers, k, E0))
    max_deriv = np.max(deriv)
    if max_deriv <= 0:
        return None
    threshold = fraction * max_deriv
    mask = deriv < threshold
    return centers[mask][0] if np.any(mask) else None


# High-level wrapper
def analyze_efficiency_curve(center, eff):
    """
    High-level analysis of efficiency curve.

    Steps:
        1. Fit logistic model
        2. Compute model efficiency
        3. Compute plateau energies using:
            - Methode A: derivative threshold
            - Methode B: curvature threshold
            - Methode C: fraction of maximum derivative

    :param center: array-like
                   log10(E/eV) bin centers
    :param eff: array-like
                Efficiency per bin

    :return result: dict
                    {
                        "params": {"k": steepness, "E0": midpoint},
                        "model_eff": np.ndarray,
                        "plateau_A": float or None,
                        "plateau_B": float or None,
                        "plateau_C": float or None,
                    }
    """
    params, model_eff = fit_efficiency_curve(center, eff)

    E_A = plateau_by_derivative_threshold(center, params, threshold=0.01)
    E_B = plateau_by_curvature_threshold(center, params, threshold=0.01)
    E_C = plateau_by_fraction_of_max_derivative(center, params, fraction=0.05)

    return {
        "params": params,
        "model_eff": model_eff,
        "plateau_A": E_A,
        "plateau_B": E_B,
        "plateau_C": E_C,
    }
