"""
Centralized YAML configuration loader for all cbanalysis pipelines.

This loader:
    - Reads a YAML configuration file
    - Detects which pipeline it belongs to
    - Constructs the appropriate dataclasses
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

cbefficiency.yaml must contain:
    array
    data
    processing
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
    EfficiencyFilesConfig,
    EfficiencyProcessingConfig,
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

    For cbefficiency:
        array_cfg       = ArrayConfig
        eff_proc_cfg    = EfficiencyProcessingConfig
        eff_files_cfg   = EfficiencyFilesConfig
        output_cfg      = OutputConfig
    """

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    is_cbprocess = "array" in cfg and "quality_cuts" in cfg
    is_cbspec = "energy" in cfg and "geometry" in cfg
    is_cbefficiency = (
        "array" in cfg
        and "data" in cfg
        and "processing" in cfg
        and "output" in cfg
        and "quality_cuts" not in cfg   # distinguishes from cbprocess
        and "energy" not in cfg         # distinguishes from cbspec
    )

    if not (is_cbprocess or is_cbspec or is_cbefficiency):
        raise ValueError(
           f"Configuration file {config_path} does not match any cbanalysis pipeline schema."
        )

    # cbprocess branch
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

    # cbspec branch
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

    # cbefficiency branch
    if is_cbefficiency:
        array_cfg = ArrayConfig(
            array_type=cfg["array"]["type"],
            mc_file=None,  # cbefficiency does not use parquet MC/data
            dt_file=None,
        )

        proc = cfg["processing"]
        energy = proc["energy"]
        eff_proc_cfg = EfficiencyProcessingConfig(
            periods=proc["periods"],
            en_min=energy["en_min"],
            en_max=energy["en_max"],
            bin_size=energy["bin_size"],
        )

        arr = cfg["array"]["type"]
        data_selection = cfg["data"]["tasd"] if arr == "TASD" else cfg["data"]["cbsd"]

        eff_files_cfg = EfficiencyFilesConfig(
            mc_thrown_geomcuts=Path(data_selection["mc_thrown_geomcuts"]),
            mc_thrown_fullcuts=Path(data_selection["mc_thrown_fullcuts"]),
        )

        out_cfg = cfg["output"]
        output_cfg = OutputConfig(
            base_dir=Path(out_cfg["base_dir"]),
            plots_dir=Path(out_cfg["plots_dir"]),
            logs_dir=Path(out_cfg["logs_dir"]),
            runs_dir=Path(out_cfg["runs_dir"]),
        )

        return array_cfg, eff_proc_cfg, eff_files_cfg, output_cfg, cfg