"""
Dataclasses defining the structured configuration objects used throughout
the cbspec pipeline.

These classes provide a clean, typed interface between:
    - YAML configuration
    - parquet ingestion
    - quality cuts
    - physics modules (binning, aperture, exposure, flux, spectrum)
    - output utilities

They intentionally contain *no logic* -- only data. This keeps the pipeline
transparent, modular, and easy to reason out.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ArrayConfig:
    """
    Configuration for detector array type and file paths.

    :param array_type: str
                       Either "TASD" or "CBSD". Determines:
                            - which parquet files to load
                            - zenith-angle correction in process_data.py
                            - output filename prefixes
    :param mc_file: Path
                    MC parquet file path for the chosen array
    :param dt_file: Path
                    Data parquet file path for the chosen array
    """
    array_type: str
    mc_file: Path
    dt_file: Path

@dataclass
class SpectrumConfig:
    """
    Configuration for the physics pipeline: binning, geometry, and run time

    :param en_range: np.ndarray
                     Array of log10(E/EeV) bin edges. These define:
                        - bin centers
                        - bin widths
                        - histogram structure
    :param generated_area_m2: float
                              A_gen - MC generated area in m²
                              From Dmitri Ivanov's thesis: π (25 km)² ≈ 1.96×10⁹ m²
    :param generated_solid_angle_sr: float
                                     Ω_gen - MC generated solid angel in sr
                                     From Dmitri Ivanov's thesis: 3π/4 ≈ 2.356 sr
    :param run_time_s: float
                       Total Telescope Array detector live time in seconds
                       Example: 16 years ≈ 5.049×10⁸ s
    """
    en_range: np.ndarray
    generated_area_m2: float
    generated_solid_angle_sr: float
    run_time_s: float

@dataclass
class QualityCuts:
    """
    Configuration for all TA-style quality cuts.

    These thresholds are applied batch-wise in process_data.py and determine
    which reconstructed events are accepted into the final MC/data histograms.

    Parameters correspond directly to YAML fields.
    """
    number_of_good_sd: int
    theta_deg: float
    boarder_dist_m: float
    geometry_chi2: float
    ldf_chi2: float
    ped_error: float
    frac_s800: float

@dataclass
class OutputConfig:
    """
    Configuration for output directory structure.

    :param base_dir: Path
                     Root output directory (e.g., "output/")
    :param plots_dir: Path
                      Global plots directory (e.g., "output/plots/")
    :param logs_dir: Path
                     Global logs directory (e.g., "output/logs/")
    :param runs_dir: Path
                     Parent directory for run-specific snapshots:
                        output/runs/<timestamp>/
    """
    base_dir: Path
    plots_dir: Path
    logs_dir: Path
    runs_dir: Path