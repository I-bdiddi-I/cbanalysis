"""
Main orchestration for the cbefficiency pipeline.

This pipeline:
    1. Sets up logging and run directories

"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from cbanalysis.utils.data_classes import (
    ArrayConfig,
    EfficiencyFilesConfig,
    EfficiencyProcessingConfig,
    OutputConfig
)
from cbanalysis.utils.logging_utils import RunLogger
from cbanalysis.utils.output_utils import save_data_csv
from cbanalysis.utils.binning import (
    make_energy_bins_from_min_max_size,
    histogram_events,
)
from .efficiency import (
    compute_efficiency_with_error,
    analyze_efficiency_curve,
)
from cbanalysis.utils.plotting import plot_efficiency_curve


# Run directory creation
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


def run_cbefficiency(
        array_cfg: ArrayConfig,
        eff_files_cfg: EfficiencyFilesConfig,
        eff_proc_cfg: EfficiencyProcessingConfig,
        output_cfg: OutputConfig,
        cfg: dict,
        cli_args=None,
):
    # 1. Run directory + logger
    run_dir, logs_dir = _make_run_dir(output_cfg)
