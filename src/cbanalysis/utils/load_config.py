"""
Load YAML configuration files and convert them into the structured dataclasses
used throughout the cbanalysis framework.

The YAML file is expected to contain the following top-level keys:

    array:
        array_typ: str
        mc_file: path/to/mc.parquet
        dt_file: path/to/data.parquet

    spectrum:
        en_range: List
        generated_area_m2: float
        generated_solid_angle_sr: float
        run_time_s: float

    quality_cuts:
        number_of_good_sd: int
        theta_deg: float
        boarder_dist_m: float
        geometry_chi2: float
        ldf_chi2: float
        ped_error: float
        frac_s800: float

    output:
        base_dir: str
        plots_dir: str
        log_dir: str
        runs_dir: str

These fields map directly into the dataclasses defined in data_classes.py.
"""

from pathlib import Path
import yaml
import numpy as np
from .data_classes import ArrayConfig, SpectrumConfig, QualityCuts, OutputConfig


def load_config(path: Path):
    """
    Load a YAML configuration file and return the corresponding dataclasses:
        - ArrayConfig
        - SpectrumConfig
        - QualityCuts
        - OutputConfig
        - raw cfg dict
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Array configuration
    array_type = cfg["array"]["type"]

    array_cfg = ArrayConfig(
        array_type=array_type,
        mc_file=None,
        dt_file=None,
    )

    # Spectrum configuration
    spectrum_cfg = SpectrumConfig(
        en_range=np.array(cfg["energy"]["bins"], dtype=float),
        generated_area_m2=float(cfg["geometry"]["generated_area_m2"]),
        generated_solid_angle_sr=float(cfg["geometry"]["generated_solid_angle_sr"]),
        run_time_s=float(cfg["run"]["time_s"]),
    )

    # Quality cuts
    qc = cfg["quality_cuts"]
    quality_cuts = QualityCuts(
        number_of_good_sd=qc["number_of_good_sd"],
        theta_deg=qc["theta_deg"],
        boarder_dist_m=qc["boarder_dist_m"],
        geometry_chi2=qc["geometry_chi2"],
        ldf_chi2=qc["ldf_chi2"],
        ped_error=qc["ped_error"],
        frac_s800=qc["frac_s800"],
    )

    # Output configuration
    out_cfg = cfg["output"]
    #config_root = path.parent.parent
    output_cfg = OutputConfig(
        base_dir=Path(out_cfg["base_dir"]),
        plots_dir=Path(out_cfg["plots_dir"]),
        logs_dir=Path(out_cfg["logs_dir"]),
        runs_dir=Path(out_cfg["runs_dir"]),
    )

    return array_cfg, spectrum_cfg, quality_cuts, output_cfg, cfg