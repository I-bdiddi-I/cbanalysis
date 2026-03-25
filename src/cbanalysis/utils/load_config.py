"""
Unified configuration loader for both cbprocess and cbspec pipelines.

This loader:
    - Reads a YAML configuration file
    - Detects whether it is a cbprocess or cbspec configuration file
    - Contructs the appropriate dataclasses
    - Returns a consistent tuple so the pipelines do not break

cbprocess.yaml must contain:
    array
    data
    processing
    quality_cuts
    output

cbspec.yaml must contain:
    energy
    geometry
    run
    output
"""

from pathlib import Path
import yaml
import numpy as np

from .data_classes import (
    ArrayConfig,
    SpectrumConfig,
    QualityCuts,
    OutputConfig,
)


def load_config(config_path: Path):
    """
    Load a YAML configuration file and return the appropriate dataclasses.

    Returns a 5-tuple for compatiblity with existing pipelines:
        (array_cfg, spectrum_cfg, cuts_cfg, output_cfg, cfg_dict)

    For cbprocess:
        array_cfg       = ArrayConfig
        spectrum_cfg    = None
        cuts_cfg        = QualityCuts
        output_cfg      = OutputConfig

    For cbspec:
        array_cfg       = None
        spectrum_cfg    = SpectrumConfig (energy bins + geometry + run)
        cuts_cfg        = None
        output_cfg      = OutputConfig
    """

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    is_cbprocess = "array" in cfg and "quality_cuts" in cfg
    is_cbspec = "energy" in cfg and "geometry" in cfg

    if not (is_cbprocess or is_cbspec):
        raise ValueError(
           f"Configuration file {config_path} does not match cbprocess or cbspec schema."
        )
    if is_cbprocess:
        array_cfg = ArrayConfig(
            array_type=cfg["array"]["type"],
            mc_file=None,    # filled later in run_cbprocess
            dt_file=None,
        )

        spectrum_cfg = None # cbprocess does not use spectrum config

        qc = cfg["quality_cuts"]
        cuts_cfg = QualityCuts(
            number_of_good_sd=qc["number_of_good_sd"],
            theta_deg=qc["theta_deg"],
            boarder_dist_m=qc["boarder_dist_m"],
            geometry_chi2=qc["geometry_chi2"],
            ldf_chi2=qc["ldf_chi2"],
            ped_error=qc["ped_error"],
            frac_s800=qc["frac_s800"],
        )

        out_cfg = cfg["output"]
        output_cfg = OutputConfig(
            base_dir=Path(out_cfg["base_dir"]),
            plots_dir=Path(out_cfg["plots_dir"]),
            logs_dir=Path(out_cfg["logs_dir"]),
            runs_dir=Path(out_cfg["runs_dir"]),
        )

        return array_cfg, spectrum_cfg, cuts_cfg, output_cfg, cfg

    if is_cbspec:
        array_cfg = None    # cbspec does not use array config

        spectrum_cfg = SpectrumConfig(
            en_range=np.array(cfg["energy"]["bins"], dtype=float),
            generated_area_m2=float(cfg["geometry"]["generated_area_m2"]),
            generated_solid_angle_sr=float(cfg["geometry"]["generated_solid_angle_sr"]),
            run_time_s=float(cfg["run"]["time_s"]),
        )

        cuts_cfg = None     # cbspec does not apply TA quality cuts

        out_cfg = cfg["output"]
        output_cfg = OutputConfig(
            base_dir=Path(out_cfg["base_dir"]),
            plots_dir=Path(out_cfg["plots_dir"]),
            logs_dir=Path(out_cfg["logs_dir"]),
            runs_dir=Path(out_cfg["runs_dir"]),
        )

        return array_cfg, spectrum_cfg, cuts_cfg, output_cfg, cfg