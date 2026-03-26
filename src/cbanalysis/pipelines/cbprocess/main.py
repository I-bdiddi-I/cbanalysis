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
#from cbanalysis.utils.output_utils import save_processed_arrays_csv
from .process_data import set_up_energy_array

def _make_run_dir(output_cfg: OutputConfig):
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

    periods = cfg["processing"]["periods"]

    results = set_up_energy_array(
        infiles=[array_cfg.mc_file, array_cfg.dt_file],
        array_type=array_cfg.array_type,
        cuts=cuts_cfg,
        periods=periods,
        logger=logger,
    )

    logger.log_text(f"Saving results ...")
    logger.log_json(event="save_results")

    # Save period-split CSVs
    for k in range(periods):
        start, end = results["period_ranges"][k]

        logger.log_text(f"Period {k}: {int(start):06d} to {int(end):06d}")
        logger.log_json(event=f"period_ranges_{k}")

        if periods == 1:
            suffix = ""

        else:
            suffix = f"_{int(start):06d}_{int(end):06d}"

        # MC reconstructed energies
        mc_filename = f"{array_cfg.array_type}_mc_recon_cut{suffix}.csv"
        pd.DataFrame({"log10(E/eV)": results["mc_recon_fullcuts"][k]}).to_csv(
            output_cfg.base_dir / "data" / mc_filename, index=False
        )
        pd.DataFrame({"log10(E/eV)": results["mc_recon_fullcuts"][k]}).to_csv(
            run_dir / "data" / mc_filename, index=False
        )

        # Data reconstructed energies
        dt_filename = f"{array_cfg.array_type}_data_recon_cut{suffix}.csv"
        pd.DataFrame({"log10(E/eV)": results["dt_recon_fullcuts"][k]}).to_csv(
            output_cfg.base_dir / "data" / dt_filename, index=False
        )
        pd.DataFrame({"log10(E/eV)": results["dt_recon_fullcuts"][k]}).to_csv(
            run_dir / "data" / dt_filename, index=False
        )

        # MC thrown energies (no cuts)
        thrown_filename = f"{array_cfg.array_type}_mc_thrown_nocuts{suffix}.csv"
        pd.DataFrame({"log10(E/eV)": results["mc_thrown_nocuts"][k]}).to_csv(
            output_cfg.base_dir / "data" / thrown_filename, index=False
        )
        pd.DataFrame({"log10(E/eV)": results["mc_thrown_nocuts"][k]}).to_csv(
            run_dir / "data" / thrown_filename, index=False
        )

        # MC thrown energies (geom cuts)
        geom_filename = f"{array_cfg.array_type}_mc_thrown_geomcuts{suffix}.csv"
        pd.DataFrame({"log10(E/eV)": results["mc_thrown_geomcuts"][k]}).to_csv(
            output_cfg.base_dir / "data" / geom_filename, index=False
        )
        pd.DataFrame({"log10(E/eV)": results["mc_thrown_geomcuts"][k]}).to_csv(
            run_dir / "data" / geom_filename, index=False
        )

    passed_cuts_filename = f"{array_cfg.array_type}_passed_cuts_data.parquet"
    ak.to_parquet(
        results["passed_cuts_df"],
        output_cfg.base_dir / "data" / passed_cuts_filename
    )
    ak.to_parquet(
        results["passed_cuts_df"],
        run_dir / "data" / passed_cuts_filename
    )

    logger.log_text("Saved processed arrays.")
    logger.log_json(event="processed_arrays_saved")

    logger.log_text("cbprocess pipeline completed successfully.")
    logger.log_json(event="cbprocess_complete")
    logger.close()

    return results