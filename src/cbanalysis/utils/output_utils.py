"""
Utility functions for saving data products (CSV tables, arrays, plots) for all pipelines in
the cbanalysis project.

This module centralizes:
    - directory creation
    - period-aware filename construction
    - saving CSVs to BOTH global and run-specific directories
    - saving plots to BOTH global and run-specific directories

It ensures:
    - consistent naming conventions across pipelines
    - array-tagged filenames (TASD, CBSD)
    - optional period-tagging (yymmdd_start-yymmdd_end)
    - synchronized global/run outputs
"""


import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

from .logging_utils import RunLogger


# Directory creation helper
def ensure_dir(path: str):
    """
    Create a directory if it does not already exist.

    :param path: str
                 Path to directory to create

    Notes:
        - Used by all pipelines to guarantee directory existence before writing
    """
    os.makedirs(path, exist_ok=True)


# Period-aware filename helper
def period_suffix(array_type: str, base_stem: str, period_range):
    """
    Construct a period-tagged filename for cbprocess-style stems.

    :param array_type: str
                       "TASD" or "CBSD"
    :param base_stem: str
                      e.g., "mc_recon_cut", "mc_thrown_fullcuts"
    :param period_range: tuple or None
                         ("yymmdd_start", "yymmdd_end") or None

    :return filename: str
                      e.g., "TASD_mc_recon_cut_080514_160504.csv"
                      or "TASD_mc_recon_cut.csv" if period_range is None
    Notes:
        - cbprocess uses EXACT filename stems; this function preserves them
        - Only the suffix is added when periods > 1
    """
    if period_range is None:
        return f"{array_type}_{base_stem}.csv"

    start, end = period_range
    return f"{array_type}_{base_stem}_{start}_{end}.csv"


# Period-aware filename + title helper
def period_suffix_and_title(array_type, base_title, period_range):
    """
    Construct period-tagged filename suffix and title additions.

    :param array_type: str
                       "TASD" or "CBSD"
    :param base_title: str
                       e.g., "Aperture", "Exposure", "Efficiency"
    :param period_range: tuple or None
                         ("yymmdd_start", "yymmdd_end") or None

    :return suffix: str
                    e.g., "_080514_160504" or ""
    :return title: str
                   e.g., "080514-160504 TASD Aperture" or "TASD Aperture"

    Notes:
        - Used by plotting utilities (cbspec, cbefficiency)
        - If period_range is None → no tagging
    """
    if period_range is None:
        return "", f"{array_type} {base_title}"

    start, end = period_range
    suffix = f"_{start}_{end}"
    title = f"{start}-{end} {array_type} {base_title}"
    return suffix, title


# Shared save function (global + run)
def save_plot(global_output_dir, run_output_dir, filename, logger: RunLogger):
    """
    Saves plots to both global and run-specific output directories.

    :param global_output_dir: str or Path
                              Base directory for global outputs
    :param run_output_dir: str or Path
                           Base directory for run-specific outputs
    :param filename: str
                     Name of the file to save (e.g., "TASD_flux.png")
    :param logger: RunLogger
                   Logger instance for recording save events.

    Notes:
        - This function does not modify filenames
        - It simply writes the same file to two locations
    """
    global_plot_dir = os.path.join(global_output_dir, "plots")
    run_plot_dir = os.path.join(run_output_dir, "plots")

    ensure_dir(global_plot_dir)
    ensure_dir(run_plot_dir)

    global_path = os.path.join(global_plot_dir, filename)
    run_path = os.path.join(run_plot_dir, filename)

    logger.log_text(f"Saving {filename} to {global_plot_dir}...")
    logger.log_json(event=f"save_{filename}_global")
    plt.savefig(global_path)

    logger.log_text(f"Saving {filename} to {run_plot_dir}...")
    logger.log_json(event=f"save_{filename}_run")
    plt.savefig(run_path)


def save_data_csv(
        global_output_dir: str,
        run_output_dir: str,
        filename: str,
        columns: dict,
        logger: RunLogger,
):
    """
    General-purpose CSV writer for all pipelines

    It writes a CSV with arbitrary columns to BOTH:
        - global_output_dir/data/
        - run_output_dir/data/

    :param global_output_dir: str
                              Path to global output directory
    :param run_output_dir: str
                           Path to run-specific output directory
    :param filename: str
                     Exact filename to write (e.g., "TASD_flux.csv")
                     The caller is responsible for constructing period tags
    :param columns: dict
                    Mapping for column name → array-like values
                    Example:
                        {
                            "Energy": centers,
                            "J": flux,
                            "Lower": flux_lower,
                            "Upper": flux_upper,
                        }
    :param logger: RunLogger
                   Logger instance for recording save events

    Notes:
        - This function is intentionally minimal and pipeline-agnostic
        - All naming conventions are controlled by main.py in each pipeline
        - Period tagging is handled BEFORE calling this function
    """

    # Ensure directories exist
    global_data_dir = os.path.join(global_output_dir, "data")
    run_data_dir = os.path.join(run_output_dir, "data")
    ensure_dir(global_data_dir)
    ensure_dir(run_data_dir)

    # Construct DataFrame
    df = pd.DataFrame(columns)

    # Paths
    global_path = os.path.join(global_data_dir, filename)
    run_path = os.path.join(run_data_dir, filename)

    # Save global
    logger.log_text(f"Saving {filename} to {global_path}...")
    logger.log_json(event=f"save_{filename}_global")
    df.to_csv(global_path, index=False)

    # Save run-specific
    logger.log_text(f"Saving {filename} to {run_path}...")
    logger.log_json(event=f"save_{filename}_run")
    df.to_csv(run_path, index=False)

    return global_path, run_path



# JUNK
def write_large_array_in_chunks(arr, filename, chunk_size=500000):

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(["log10(E/eV)"])

    n = len(arr)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = arr[start:end]

        with open(filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for value in chunk:
                writer.writerow([value])

def save_processed_arrays_csv(
        global_output_dir: str,
        run_output_dir: str,
        array_type: str,
        mc_array,
        dt_array,
        mc_thrown_array,
        logger: RunLogger,
        chunck_size: int = 500000,
):
    """
    Save processed log10(E/eV) arrays (MC recon, data recon, MC thrown) to CSV files.

    Files written to BOTH:
        - global_output_dir/data
        - run_output_dir/data/

    Filenames:
        <array_type>_mc_recon_cut.csv
        <array_type>_data_recon_cut.csv
        <array_type>_mc_thrown_uncut.csv

    Each CSV contains a single column:
        log10(E/eV)
    :param global_output_dir:
    :param run_output_dir:
    :param array_type:
    :param mc_array:
    :param dt_array:
    :param mc_thrown_array:
    :param logger:
    :return:
    """
    # Global directory
    global_data_dir = os.path.join(global_output_dir, "data")
    ensure_dir(global_data_dir)

    # Run-specific directory
    run_data_dir = os.path.join(run_output_dir, "data")
    ensure_dir(run_data_dir)

    # Array-tagged filename
    filenames = {
        f"{array_type}_mc_reco_cut.csv": mc_array,
        f"{array_type}_data_reco_cut.csv": dt_array,
        f"{array_type}_mc_thrown_uncut.csv": mc_thrown_array,
    }

    for filename, arr in filenames.items():
        # Flatten and ensure numeric dtype
        #arr = np.asarray(arr).reshape(-1).astype(float)
        arr = np.asarray(arr, dtype=float).flatten()

        df = pd.DataFrame({"log10(E/eV)": arr})

        global_path = os.path.join(global_data_dir, filename)
        run_path = os.path.join(run_data_dir, filename)

        logger.log_text(f"Saving {filename} to {global_path}...")
        logger.log_json(event=f"save_{filename}_global")
        #write_large_array_in_chunks(arr, global_path)
        df.to_csv(global_path, index=False)

        logger.log_text(f"Saving {filename} to {run_path}...")
        logger.log_json(event=f"save_{filename}_run")
        #write_large_array_in_chunks(arr, run_path)
        df.to_csv(run_path, index=False)

    return True

def save_flux_csv(
        global_output_dir: str,
        run_output_dir: str,
        array_type: str,
        centers,
        widths,
        n_events,
        exposure,
        flux,
        flux_lower,
        flux_upper,
        logger: RunLogger,
):
    """
    Save final flux table to CSV.

    The table is written to BOTH:
        - global_output_dir/data/
        - run_output_dir/data/

    Final CSV columns:
        Energy, Bin_size, N_events, Exposure, J, Lower, Upper

    Here:
        Energy      = log10(E/eV) bin center
        Bin_size    = bin width in log10(E/eV)
        N_events    = data counts per bin
        Exposure    = exposure per bin [m^2 sr s]
        J           = flux J(E) in [m^-2 sr^-1 s^-1 eV^-1] up to a scaling by binning convention
        Lower       = lower FC flux bound
        Upper       = upper FC flux bound

    :param global_output_dir: str
                              Path to global output directory (e.g., "output")
    :param run_output_dir: str
                           Path to run-specific output directory (e.g., "output/runs/<timestamp>")
    :param array_type: str
                       "TASD" or "CBSD". Used to tag filenames
    :param centers: array-like
                    log10(E/eV) bin centers
    :param widths: array-like
                   Bin widths in log10(E/eV)
    :param n_events: array-like
                     Data counts per bin
    :param exposure: array-like
                     Exposure per bin [m² sr s]
    :param flux: array-like
                 Differential flux J(E) [m^-2 sr^-1 s^-1 eV^-1]
    :param flux_lower: array-like
                     Feldman-Cousins lower bounds on J(E)
    :param flux_upper: array-like
                      Feldman-Cousins upper bounds on J(E)
    :param logger: RunLogger
    :return global_path: tuple of str
                         Path to global saved CSV file
    :return run_path: tuple of str
                      Path to run-specific saved CSV file
    """
    # Global directory
    global_data_dir = os.path.join(global_output_dir, "data")
    ensure_dir(global_data_dir)

    # Run-specific directory
    run_data_dir = os.path.join(run_output_dir, "data")
    ensure_dir(run_data_dir)

    # Construct DataFrame
    df = pd.DataFrame({
        "Energy": centers,
        "Bin_size": widths,
        "N_events": n_events,
        "Exposure": exposure,
        "J": flux,
        "Lower": flux_lower,
        "Upper": flux_upper,
    })

    # Array-tagged filename
    filename = f"{array_type}_flux.csv"

    global_path = os.path.join(global_data_dir, filename)
    run_path = os.path.join(run_data_dir, filename)

    # Save to both locations
    logger.log_text(f"Saving {filename} to {global_path}...")
    logger.log_json(event=f"save_{filename}_global")
    df.to_csv(global_path, index=False)

    logger.log_text(f"Saving {filename} to {run_path}...")
    logger.log_json(event=f"save_{filename}_run")
    df.to_csv(run_path, index=False)

    return global_path, run_path


def save_spectrum_csv(
        global_output_dir: str,
        run_output_dir: str,
        array_type: str,
        centers,
        spectrum,
        spectrum_lower,
        spectrum_upper,
        logger: RunLogger,
):
    """
    Save final E³J(E) spectrum table to CSV.

    The table is written to BOTH:
        - global_output_dir/data/
        - run_output_dir/data/

    Final CSV columns:
        Energy, Spectrum, Lower, Upper

    Here:
        Energy      = log10(E/eV) bin center
        Spectrum    = spectrum E^3 J(E) in [eV^2 m^-2 sr^-1 s^-1]
        Lower       = lower FC flux bound
        Upper       = upper FC flux bound

    :param global_output_dir: str
                              Path to global output directory (e.g., "output")
    :param run_output_dir: str
                           Path to run-specific output directory (e.g., "output/runs/<timestamp>")
    :param array_type: str
                       "TASD" or "CBSD". Used to tag filenames
    :param centers: array-like
                    log10(E/eV) bin centers
    :param spectrum: array-like
                     E³J(E) spectrum values
    :param spectrum_lower: array-like
                         Feldman-Cousins lower bounds on J(E) in spectrum space
    :param spectrum_upper: array-like
                          Feldman-Cousins upper bounds on J(E) in spectrum space
    :param logger: RunLogger
    :return global_path: tuple of str
                         Path to global saved CSV file
    :return run_path: tuple of str
                      Path to run-specific saved CSV file
    """
    # Global directory
    global_data_dir = os.path.join(global_output_dir, "data")
    ensure_dir(global_data_dir)

    # Run-specific directory
    run_data_dir = os.path.join(run_output_dir, "data")
    ensure_dir(run_data_dir)

    # Construct DataFrame
    df = pd.DataFrame({
        "Energy": centers,
        "Spectrum": spectrum,
        "Lower": spectrum_lower,
        "Upper": spectrum_upper,
    })

    # Array-tagged filename
    filename = f"{array_type}_spectrum.csv"

    global_path = os.path.join(global_data_dir, filename)
    run_path = os.path.join(run_data_dir, filename)

    # Save to both locations
    logger.log_text(f"Saving {filename} to {global_path}...")
    logger.log_json(event=f"save_{filename}_global")
    df.to_csv(global_path, index=False)

    logger.log_text(f"Saving {filename} to {run_path}...")
    logger.log_json(event=f"save_{filename}_run")
    df.to_csv(run_path, index=False)

    return global_path, run_path