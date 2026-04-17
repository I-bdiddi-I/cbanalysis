"""
Publication-quality plotting utilities for the cbanalysis project.

This module supports multiple pipelines:
    - cbspec (aperture, exposure, flux, spectrum, histograms)
    - cbefficiency (efficiency curves, logistic fits, plateau markers)

All plots are saved to BOTH:
    - global_output_dir/plots/
    - run_output_dir/plots/

Filenames are automatically array-tagged (and optionally period-tagged):
    - TASD_flux.png
    - CBSD_spectrum.png
    - TASD_efficiency_080514_160504.png     (period-tagged)
"""


import os
import matplotlib.pyplot as plt
import numpy as np

from .output_utils import ensure_dir, save_plot, period_suffix_and_title
from .constants import m2_to_km2, s_to_yr
from .logging_utils import RunLogger


# Basic plot structures
def plot_scatter_log_energy(centers, y_comp):
    """
    Create a basic scatter plots vs. log10(E/eV).

    :param centers: array-like
                    log10(E/eV) bin centers
    :param y_comp: array-like
                   Quantity to plot on the y-axis

    Notes:
        - This function does NOT save the plot
        - It is intended to be followed by axis labels, titles, and save_plot()
    """
    plt.figure(figsize=[8, 6])
    plt.scatter(centers, y_comp)
    plt.yscale("log")
    plt.xlim(17.8, 20.5)
    plt.xlabel(r"$\log_{10}(E/eV)$")


def plot_error_bars_log_energy(centers, y_comp, lower, upper):
    """
    Create an error-bar plot vs. log10(E/eV).

    :param centers: array-like
                    log10(E/eV) bin centers
    :param y_comp: array-like
                   Central values
    :param lower: array-like
                  Lower error bounds
    :param upper: array-like
                  Upper error bounds

    Notes:
        - This function does NOT save the plot
        - It is intended to be followed by axis labels, titles, and save_plot()
    """
    yerr = [y_comp - np.asarray(lower), np.asarray(upper) - y_comp]

    plt.figure(figsize=[8, 6])
    plt.errorbar(
        centers,
        y_comp,
        yerr=yerr,
        fmt="o",
        markersize=4,
        capsize=3,
        color="k",
        ecolor="k",
        linewidth=1,
    )
    plt.yscale("log")
    plt.xlim(17.8, 20.5)
    plt.xlabel(r"$\log_{10}(E/eV)$")


def plot_histogram(data):
    """
    Plot a histogram of log10(E/eV).

    :param data: array-like
                 log10(E/eV) values

    Notes:
        - This function does NOT save the plot
        - It is intended to be followed by axis labels, titles, and save_plot()
    """
    plt.figure(figsize=[8, 6])
    plt.hist(data, range=(18, 21), bins=30)
    plt.yscale("log")
    plt.xlabel(r"$\log_{10}(E/eV)$")


# cbspec plots
def plot_aperture(
        centers,
        aperture,
        array_type,
        global_output_dir,
        run_output_dir,
        logger: RunLogger,
        period_range=None,
):
    """
    Plot aperture vs. log10(E/eV).

    :param centers: array-like
                    log10(E/eV) bin centers
    :param aperture: array-like
                     Aperture values in [m^2 sr]
    :param array_type: str
                       "TASD" or "CBSD"
    :param global_output_dir: str or Path
                              Base directory for global outputs
    :param run_output_dir: str or Path
                           Base directory for run-specific outputs
    :param logger: RunLogger
                   Logger instance
    :param period_range: tuple or None
                         ("yymmdd_start", "yymmdd_end") or None

    Notes:
        - If period_range is provided, filenames and titles include date tags
    """
    suffix, title = period_suffix_and_title(array_type, "Aperture", period_range)
    filename = f"{array_type}_aperture{suffix}.png"

    # Convert aperture from [m^2 sr] to [km^2 sr]
    aperture = np.asarray(aperture, dtype=float) * m2_to_km2

    # Basic structure
    plot_scatter_log_energy(centers, aperture)

    # Aperture specific parameters
    plt.ylim(5, 5 * 10**3)
    plt.title(title)
    plt.ylabel(r"Aperture [km$^{2}$ sr]")

    save_plot(global_output_dir, run_output_dir, filename, logger)

    plt.close()


def plot_exposure(
        centers,
        exposure,
        array_type,
        global_output_dir,
        run_output_dir,
        logger: RunLogger,
        period_range=None,
):
    """
    Plot exposure vs. log10(E/eV).

    :param centers: array-like
                    log10(E/eV) bin centers
    :param exposure: array-like
                     Exposure values in [m^2 sr s]
    :param array_type: str
                       "TASD" or "CBSD"
    :param global_output_dir: str or Path
                              Base directory for global outputs
    :param run_output_dir: str or Path
                           Base directory for run-specific outputs
    :param logger: RunLogger
                   Logger instance
    :param period_range: tuple or None
                         ("yymmdd_start", "yymmdd_end") or None
    """
    suffix, title = period_suffix_and_title(array_type, "Exposure", period_range)
    filename = f"{array_type}_exposure{suffix}.png"

    # Convert exposure from [m^2 sr s] to [km^2 sr yr]
    exposure = np.asarray(exposure, dtype=float) * m2_to_km2 * s_to_yr

    plot_scatter_log_energy(centers, exposure)

    # Exposure specific parameters
    plt.ylim(1, 2 * 10**4)
    plt.title(title)
    plt.ylabel(r"Exposure [km$^{2}$ sr yr]")

    save_plot(global_output_dir, run_output_dir, filename, logger)

    plt.close()


def plot_flux(
        centers,
        flux,
        flux_lower,
        flux_upper,
        array_type,
        global_output_dir,
        run_output_dir,
        logger: RunLogger,
        period_range=None,
):
    """
    Plot flux vs. log10(E/eV).

    :param centers: array-like
                    log10(E/eV) bin centers
    :param flux: array-like
                     Flux values
    :param flux_lower: array-like
                       Lower error bounds
    :param flux_upper: array-like
                       Upper error bounds
    :param array_type: str
                       "TASD" or "CBSD"
    :param global_output_dir: str or Path
                              Base directory for global outputs
    :param run_output_dir: str or Path
                           Base directory for run-specific outputs
    :param logger: RunLogger
                   Logger instance
    :param period_range: tuple or None
                         ("yymmdd_start", "yymmdd_end") or None
    """
    suffix, title = period_suffix_and_title(array_type, "Flux", period_range)
    filename = f"{array_type}_flux{suffix}.png"

    # Convert flux and error bars to 10**30 scale
    flux = np.asarray(flux, dtype=float) * 10**30
    flux_lower = np.asarray(flux_lower, dtype=float) * 10**30
    flux_upper = np.asarray(flux_upper, dtype=float) * 10**30

    plot_error_bars_log_energy(centers, flux, flux_lower, flux_upper)

    # Flux specific parameters
    plt.ylim(10 ** (-7), 2)
    plt.title(title)
    plt.ylabel(r"J × 10$^{30}$ [eV$^{-1}$ m$^{-2}$ sr$^{-1}$ s$^{-1}$]")

    save_plot(global_output_dir, run_output_dir, filename, logger)

    plt.close()


def plot_spectrum(
        centers,
        spectrum,
        spectrum_lower,
        spectrum_upper,
        array_type,
        global_output_dir,
        run_output_dir,
        logger: RunLogger,
        period_range=None,
):
    """
    Plot spectrum vs. log10(E/eV).

    :param centers: array-like
                    log10(E/eV) bin centers
    :param spectrum: array-like
                     Spectrum values
    :param spectrum_lower: array-like
                           Lower error bounds
    :param spectrum_upper: array-like
                           Upper error bounds
    :param array_type: str
                       "TASD" or "CBSD"
    :param global_output_dir: str or Path
                              Base directory for global outputs
    :param run_output_dir: str or Path
                           Base directory for run-specific outputs
    :param logger: RunLogger
                   Logger instance
    :param period_range: tuple or None
                         ("yymmdd_start", "yymmdd_end") or None
    """
    suffix, title = period_suffix_and_title(array_type, "Spectrum", period_range)
    filename = f"{array_type}_spectrum{suffix}.png"

    # Convert spectrum and error bars to 10**-24 scale
    spectrum = np.asarray(spectrum, dtype=float) / 10**24
    spectrum_lower = np.asarray(spectrum_lower, dtype=float) / 10**24
    spectrum_upper = np.asarray(spectrum_upper, dtype=float) / 10**24

    plot_error_bars_log_energy(centers, spectrum, spectrum_lower, spectrum_upper)

    # Spectrum specific parameters
    plt.ylim(4 * 10 ** (-1), 4)
    plt.title(title)
    plt.ylabel(r"E$^{3}$ J / 10$^{24}$ [eV$^{2}$ m$^{-2}$ sr$^{-1}$ s$^{-1}$]")

    save_plot(global_output_dir, run_output_dir, filename, logger)

    plt.close()


def mc_recon_hist(
        mc_array,
        array_type,
        global_output_dir,
        run_output_dir,
        logger: RunLogger,
        period_range=None,
):
    """
    Plot histogram of MC reconstructed energies.

    :param mc_array: array-like
                     log10(E/eV) values
    :param array_type: str
                       "TASD" or "CBSD"
    :param global_output_dir: str or Path
                              Base directory for global outputs
    :param run_output_dir: str or Path
                           Base directory for run-specific outputs
    :param logger: RunLogger
                   Logger instance
    :param period_range: tuple or None
                         ("yymmdd_start", "yymmdd_end") or None
    """
    suffix, title = period_suffix_and_title(array_type, "MC Reconstructed Energies Histogram", period_range)
    filename = f"{array_type}_MC_recon_hist{suffix}.png"

    plot_histogram(mc_array)

    plt.title(title)
    plt.ylabel("N$^{MC}_{REC}$")

    save_plot(global_output_dir, run_output_dir, filename, logger)

    plt.close()


def mc_thrown_hist(
        mc_thrown_array,
        array_type,
        global_output_dir,
        run_output_dir,
        logger: RunLogger,
        period_range=None,
):
    """
    Plot histogram of MC thrown energies.

    :param mc_thrown_array: array-like
                            log10(E/eV) values
    :param array_type: str
                       "TASD" or "CBSD"
    :param global_output_dir: str or Path
                              Base directory for global outputs
    :param run_output_dir: str or Path
                           Base directory for run-specific outputs
    :param logger: RunLogger
                   Logger instance
    :param period_range: tuple or None
                         ("yymmdd_start", "yymmdd_end") or None
    """
    suffix, title = period_suffix_and_title(array_type, "MC Thrown Energies Histogram", period_range)
    filename = f"{array_type}_MC_thrown_hist{suffix}.png"

    plot_histogram(mc_thrown_array)

    plt.title(title)
    plt.ylabel("N$^{MC}_{GEN}$")

    save_plot(global_output_dir, run_output_dir, filename, logger)

    plt.close()


def dt_hist(
        dt_array,
        array_type,
        global_output_dir,
        run_output_dir,
        logger: RunLogger,
        period_range=None,
):
    """
    Plot histogram of data reconstructed energies.

    :param dt_array: array-like
                     log10(E/eV) values
    :param array_type: str
                       "TASD" or "CBSD"
    :param global_output_dir: str or Path
                              Base directory for global outputs
    :param run_output_dir: str or Path
                           Base directory for run-specific outputs
    :param logger: RunLogger
                   Logger instance
    :param period_range: tuple or None
                         ("yymmdd_start", "yymmdd_end") or None
    """
    suffix, title = period_suffix_and_title(array_type, "Data Reconstructed Energies Histogram", period_range)
    filename = f"{array_type}_DATA_recon_hist{suffix}.png"

    plot_histogram(dt_array)

    plt.title(title)
    plt.ylabel("N$^{DATA}_{REC}$")

    save_plot(global_output_dir, run_output_dir, filename, logger)

    plt.close()


# cbefficiency plot
def plot_efficiency_curve(
        centers,
        eff,
        bi_err,
        model_eff,
        plateau_A,
        plateau_B,
        plateau_C,
        array_type,
        global_output_dir,
        run_output_dir,
        logger: RunLogger,
        period_range=None,
):
    """
    Plot efficiency curve with:
        - raw efficiency points
        - binomial error bars
        - logistic fit
        - plateau markers (A, B, C)

    :param centers: array-like
                    log10(E/eV) bin centers
    :param eff: array-like
                Raw efficiency values
    :param bi_err: array-like
                   Binomial errors
    :param model_eff: array-like
                      Logistic fit evaluated at centers
    :param plateau_A: float or None
                      Plateau energy from derivative threshold
    :param plateau_B: float or None
                      Plateau energy from curvature threshold
    :param plateau_C: float or None
                      Plateau energy from fraction-of-max-derivative
    :param array_type: str
                       "TASD" or "CBSD"
    :param global_output_dir: str or Path
                              Base directory for global outputs
    :param run_output_dir: str or Path
                           Base directory for run-specific outputs
    :param logger: RunLogger
                   Logger instance
    :param period_range: tuple or None
                         ("yymmdd_start", "yymmdd_end") or None

    Notes:
        - If period_range is provided, filenames and titles include date tags
    """
    suffix, title = period_suffix_and_title(array_type, "Efficiency", period_range)
    filename = f"{array_type}_efficiency_{suffix}.png"

    plt.figure(figsize=[8, 6])

    plt.errorbar(
        centers,
        eff,
        yerr=bi_err,
        fmt="o",
        color="black",
        markersize=5,
        capsize=3,
        label="Raw Efficiency",
    )

    plt.plot(
        centers,
        model_eff,
        color="blue",
        linewidth=2,
        label="Logistic fit",
    )

    if plateau_A is not None:
        plt.axvline(plateau_A, color="red", linestyle="--", linewidth=1.5, label="Derivative threshold")

    if plateau_B is not None:
        plt.axvline(plateau_B, color="green", linestyle="--", linewidth=1.5, label="Curvature threshold")

    if plateau_C is not None:
        plt.axvline(plateau_C, color="purple", linestyle="--", linewidth=1.5, label="Fraction of max derivative")

    plt.xlabel(r"$\log_{10}(E/eV)$")
    plt.ylabel("Efficiency")
    plt.title(title)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_plot(global_output_dir, run_output_dir, filename, logger)

    plt.close()