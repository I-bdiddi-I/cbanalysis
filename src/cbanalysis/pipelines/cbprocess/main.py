"""
Top-level orchestration of the cbprocess pipeline.

This pipeline:
    1. Sets up logging and run directory
    2. Selects MC/data parquet files based on array type
    3. Reads parquet files batch-wise
    4. Applies TA-style quality cuts
    5. Accumulates:
        - MC reconstructed log10(E/eV)
        - data reconstructed log10(E/eV)
        - MC thrown log10(E/eV)
    6. Writes three CSVs (global + run-specific)
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from cbanalysis.utils.data_classes import ArrayConfig, SpectrumConfig, QualityCuts, OutputConfig
from cbanalysis.utils.logging_utils import RunLogger
from cbanalysis.utils.output_utils import save_processed_arrays_csv
from .process_data import set_up_energy_array

def _make_run_dir(output_cfg: OutputConfig):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_cfg.runs_dir / timestamp
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, logs_dir

def run_cbprocess(
        array_cfg: ArrayConfig,
        spectrum_cfg: SpectrumConfig,
        cuts_cfg: QualityCuts,
        output_cfg: OutputConfig,
        cfg: dict,
):

    run_dir, logs_dir = _make_run_dir(output_cfg)
    logger = RunLogger(logs_dir)

    logger.log_text("Starting cbprocess pipeline...")
    logger.log_json(event="start_cbprocess", array=array_cfg.array_type)

    # Select MC/DT files based on array type
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

    # Parquet ingestion (MC + data)
    logger.log_text("Reading parquet files and applying quality cuts...")
    logger.log_json(event="parquet_ingest")

    mc_array, dt_array, mc_thrown_array = set_up_energy_array(
        infiles=[array_cfg.mc_file, array_cfg.dt_file],
        array_type=array_cfg.array_type,
        cuts=cuts_cfg,
        logger=logger,
    )
    logger.log_text(f"MC cut recon events: {len(mc_array)}")
    logger.log_text(f"Data cut recon events: {len(dt_array)}")
    logger.log_text(f"MC uncut thrown events: {len(mc_thrown_array)}")

    save_processed_arrays_csv(
        global_output_dir=str(output_cfg.base_dir),
        run_output_dir=str(run_dir),
        array_type=array_cfg.array_type,
        mc_array=mc_array,
        dt_array=dt_array,
        mc_thrown_array=mc_thrown_array,
        logger=logger,
    )

    logger.log_text("Saved processed arrays.")
    logger.log_json(event="processed_arrays_saved")

    logger.log_text("cbprocess pipeline completed successfully.")
    logger.log_json(event="cbprocess_complete")
    logger.close()

    return {
        "mc_array": mc_array,
        "dt_array": dt_array,
        "mc_thrown_array": mc_thrown_array,
    }