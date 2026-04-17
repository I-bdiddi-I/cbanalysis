"""
Top-level orchestration of the cbprocess pipeline.

This pipeline:
    1. Sets up logging and run directory
    2. Selects MC/data parquet files based on array type
    3. Reads parquet files batch-wise
    4. Applies TA-style quality cuts
    5. Produces:
        - MC reconstructed log10(E/eV)
        - data reconstructed log10(E/eV)
        - MC thrown log10(E/eV) (no cuts)
        - MC thrown log10(E/eV) (full cuts)
        - MC thrown log10(E/eV) (geom cuts)
        - filtered data
    6. Optionally splits all arrays into N time periods
    7. Saves CSVs and Parquet outputs (global + run-specific)
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import awkward as ak

from cbanalysis.utils.data_classes import ArrayConfig, SpectrumConfig, QualityCuts, OutputConfig
from cbanalysis.utils.logging_utils import RunLogger
from cbanalysis.utils.output_utils import save_data_csv, period_suffix
from .process_data import set_up_energy_array

import warnings
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="divide by zero encountered in log10"
)


def _make_run_dir(output_cfg: OutputConfig):
    """
    Create run directory and logs directory

    Notes:
        - Mirrors the structure used by all pipelines
        - Ensures run/data and run/logs exist
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_cfg.runs_dir / timestamp
    logs_dir = run_dir / "logs"

    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, logs_dir

def run_cbprocess(
        array_cfg: ArrayConfig,
        spectrum_cfg: SpectrumConfig,
        cuts_cfg: QualityCuts,
        output_cfg: OutputConfig,
        cfg: dict,
):

    # 1. Run directory + logger
    run_dir, logs_dir = _make_run_dir(output_cfg)
    logger = RunLogger(logs_dir)

    logger.log_text("Starting cbprocess pipeline...")
    logger.log_json(event="start_cbprocess", array=array_cfg.array_type)

    # 2. Select MC/DT files based on array type
    logger.log_text("Determining array type for cbprocess...")
    logger.log_json(event="type_select")
    if array_cfg.array_type == "TASD":
        array_cfg.mc_file = Path(cfg["data"]["tasd"]["mc_file"])
        array_cfg.dt_file = Path(cfg["data"]["tasd"]["dt_file"])
    elif array_cfg.array_type == "CBSD":
        array_cfg.mc_file = Path(cfg["data"]["cbsd"]["mc_file"])
        array_cfg.dt_file = Path(cfg["data"]["cbsd"]["dt_file"])
    else:
        raise TypeError(f"Array type {array_cfg.array_type} is not supported")

    logger.log_text(f"Array type: {array_cfg.array_type}")
    logger.log_json(event=f"{array_cfg.array_type}_array_selected", array=array_cfg.array_type)

    # 3. Parquet ingestion (MC + data)
    logger.log_text("Reading parquet files and applying quality cuts...")
    logger.log_json(event="parquet_ingest")

    periods = cfg["processing"]["periods"]

    results = set_up_energy_array(
        infiles=[array_cfg.mc_file, array_cfg.dt_file],
        array_type=array_cfg.array_type,
        cuts=cuts_cfg,
        periods=periods,
        logger=logger,
    )

    energy = results["energy"]

    logger.log_text(f"Saving results ...")
    logger.log_json(event="save_results")

    # 4. Save period-split CSVs
    logger.log_text("Saving results ...")
    logger.log_json(event="save_results")

    for k in range(periods):
        start, end = results["period_ranges"][k]

        # Convert to universal period format (yymmdd strings)
        period_range = None if periods == 1 else (
            f"{int(start):06d}",
            f"{int(end):06d}",
        )

        logger.log_text(f"Period {k+1}: {int(start):06d} to {int(end):06d}")
        logger.log_json(event=f"period_ranges_{k+1}")

        # MC reconstructed energies
        mc_filename = period_suffix(array_cfg.array_type, "mc_recon_cut", period_range)
        save_data_csv(
            output_cfg.base_dir,
            run_dir,
            mc_filename,
            {"log10(E/eV)": energy["mc_recon"][k]},
            logger,
        )

        # Data reconstructed energies
        dt_filename = period_suffix(array_cfg.array_type, "data_recon_cut", period_range)
        save_data_csv(
            output_cfg.base_dir,
            run_dir,
            dt_filename,
            {"log10(E/eV)": energy["dt_recon"][k]},
            logger,
        )

        # MC thrown energies (no cuts)
        nocuts_filename = period_suffix(array_cfg.array_type, "mc_thrown_nocuts", period_range)
        save_data_csv(
            output_cfg.base_dir,
            run_dir,
            nocuts_filename,
            {"log10(E/eV)": energy["mc_thrown_nocuts"][k]},
            logger,
        )

        # MC thrown energies (full cuts)
        fullcuts_filename = period_suffix(array_cfg.array_type, "mc_thrown_fullcuts", period_range)
        save_data_csv(
            output_cfg.base_dir,
            run_dir,
            fullcuts_filename,
            {"log10(E/eV)": energy["mc_thrown_fullcuts"][k]},
            logger,
        )

        # MC thrown energies (geom cuts)
        geom_filename = period_suffix(array_cfg.array_type, "mc_thrown_geomcuts", period_range)
        save_data_csv(
            output_cfg.base_dir,
            run_dir,
            geom_filename,
            {"log10(E/eV)": energy["mc_thrown_geomcuts"][k]},
            logger,
        )

    # Save pass-cuts data to parquet
    passed_cuts_filename = f"{array_cfg.array_type}_passed_cuts_data.parquet"

    global_parquet_path = output_cfg.base_dir / "data" / passed_cuts_filename
    run_parquet_path = run_dir / "data" / passed_cuts_filename

    # Global save
    logger.log_text(f"Saving {passed_cuts_filename} to {global_parquet_path}...")
    logger.log_json(event=f"save{passed_cuts_filename}_global")
    ak.to_parquet(results["passed_cuts_df"], global_parquet_path)

    # Run-specific save
    logger.log_text(f"Saving {passed_cuts_filename} to {run_parquet_path}...")
    logger.log_json(event=f"save{passed_cuts_filename}_run")
    ak.to_parquet(results["passed_cuts_df"], run_parquet_path)

    logger.log_text("Saved processed arrays.")
    logger.log_json(event="processed_arrays_saved")

    logger.log_text("cbprocess pipeline completed successfully.")
    logger.log_json(event="cbprocess_complete")
    logger.close()

    return results