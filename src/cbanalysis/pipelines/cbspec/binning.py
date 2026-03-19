"""
Energy binning utilities in log10(E/eV) space.

This module is intentionally simple: it defines the bin edges, centers, and
widths used throughout the physics pipeline, and provies histogram helpers
for MC reconstruction, MC thrown, and data reconstructed energies.

All downstream physics (aperture, exposure, flux, spectrum) depends on these
bins being consistent and reproducible.
"""

import numpy as np


def make_energy_bins(en_range):
    """
    Constructs bin edges, centers, and widths from log10(E/eV) edges.
    :param en_range: array-like
                     Bin edges in log10(E/eV)
                     These come directly from YAML
    :return edges: np.ndarray
                   Bin edges in log10(E/eV)
    :return centers: np.ndarray
                     Bin centers in log10(E/eV)
    :return widths: np.ndarray
                    Bin widths in log10(E/eV)
    """
    edges = np.asarray(en_range, dtype=float)
    centers = 0.5 * (edges[1:] + edges[:-1])
    widths = edges[1:] - edges[:-1]
    return edges, centers, widths


def histogram_events(log_energy, edges):
    """
    Histogram events into energy bins.
    :param log_energy: array-like
                       log10(E/eV) values
    :param edges: array-like
                  Bin edges in log10(E/eV)
    :return counts: np.ndarray
                    Counts per energy bin
    """
    counts, _ = np.histogram(log_energy, bins=edges)
    return counts


def histgram_data_per_bin(mc_log_energy, dt_log_energy, mc_thrown_log_energy, edges):
    """
    Histogram MC reconstructed, data reconstructed, and MC thrown energies into energy bins.
    :param mc_log_energy: array-like
                          Reconstructed MC log10(E/eV)
    :param dt_log_energy: array-like
                          Reconstructed data log10(E/eV)
    :param mc_thrown_log_energy: array-like
                              Thrown MC log10(E/eV)
    :param edges: array-like
                  Bin edges in log10(E/eV)
    :return mc_counts: np.ndarray
                       Reconstructed MC counts per energy bin
    :return dt_counts: np.ndarray
                       Reconstructed data counts per energy bin
    :return mc_thrown_counts: np.ndarray
                           Thrown MC counts per energy bin
    """
    mc_counts = histogram_events(mc_log_energy, edges)
    dt_counts = histogram_events(dt_log_energy, edges)
    mc_thrown_counts = histogram_events(mc_thrown_log_energy, edges)
    return mc_counts, dt_counts, mc_thrown_counts

def filter_bins(mc_counts, dt_counts, mc_raw_counts, centers):
    """
    Apply physics-motivated bin filters:
        1. energy_mask:
            Only keep bins with log10(E_thrown/eV) > 18.5
            Below this, CBSD reconstruction is not reliable

        2. raw_mask:
            Only keep bins with N_MC_thrown > 1
            This ensures that aperture denominator is non-zero and statistically
            meaningful

    These filters ensure that downstream modules (exposure, flux, spectrum,
    plotting) only see physically valid bins
    :param mc_counts: np.ndarray
                      Reconstructed MC counts per bin
    :param dt_counts: np.ndarray
                      Data counts per bin
    :param mc_raw_counts: np.ndarray
                          Thrown MC counts per bin
    :param centers: np.ndarray
                    log10(E/eV) bin centers
    :return mask: np.ndarray (bool)
                  Combined mask applied to all arrays
    :return mc_counts_f: np.ndarray
                         Filtered reconstructed MC counts per energy bin
    :return dt_counts_f: np.ndarray
                         Filtered reconstructed data counts per energy bin
    :return mc_raw_counts_f: np.ndarray
                             Filtered thrown MC counts per energy bin
    :return centers_f: np.ndarray
                       Filtered log10(E/eV) bin centers
    """
    centers = np.asarray(centers, dtype=float)
    mc_counts = np.asarray(mc_counts, dtype=float)
    dt_counts = np.asarray(dt_counts, dtype=float)
    mc_raw_counts = np.asarray(mc_raw_counts, dtype=float)

    # 1. Energy mask
    energy_mask = centers > 18.5

    # 2. Raw mask
    raw_mask = mc_raw_counts > 1

    # Combined mask
    mask = energy_mask & raw_mask

    return (
        mask,
        mc_counts[mask],
        dt_counts[mask],
        mc_raw_counts[mask],
        centers[mask],
    )


def energy_conv(centers, widths):
    energies_ev = 10. ** np.asarray(centers, dtype=float)

    delta_energies_ev = energies_ev * (10 ** (widths / 2) - 10 ** (-widths / 2))

    return energies_ev, delta_energies_ev