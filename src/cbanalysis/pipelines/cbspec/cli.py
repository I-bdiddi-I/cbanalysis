"""
Command-line interface for the cbspec pipline.

This module provides the entry point for:
    python -m cbspec
    python -m cbspec --config config/default_config.yaml
    python -m cbspec --array-type CBSD

The CLI supports overriding:
    - YAML configuration file
    - array type (TASD or CBSD)
    - MC parquet file path
    - data parquet file path

All arguments are optional -- if omitted, defaults come from YAML file.
"""

import argparse
from pathlib import Path

from .load_config import load_config
from .main import run_pipeline


# CLI argument parser
def pars_args():
    """
    Define and parse command-line arguments.
    :return argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run the cbspec UHECR spectrum pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--array_type",
        type=str,
        choices=["TASD", "CBSD"],
        help="Override array type (TASD or CBSD).",
    )
    parser.add_argument(
        "--mc_file",
        type=str,
        help="Override MC parquet file path.",
    )
    parser.add_argument(
        "--dt_file",
        type=str,
        help="Override data parquet file path.",
    )

    return parser.parse_args()

def main():
    """
    Entry point for the cbspec CLI.

    Steps:
        1. Parse command-line arguments
        2. Load YAML configuration file
        3. Apply CLI overrides
        4. Run the full pipeline
    """
    args = pars_args()

    # Load YAML config → dataclasses
    array_cfg, spectrum_cfg, cuts_cfg, output_cfg, cfg = load_config(args.config)

    # Apply CLI overrides
    if args.array_type is not None:
        array_cfg.array_type = args.array_type

    if args.mc_file is not None:
        array_cfg.mc_file = Path(args.mc_file)

    if args.dt_file is not None:
        array_cfg.dt_file = Path(args.dt_file)

    # Run the full pipeline
    run_pipeline(
        array_cfg=array_cfg,
        spectrum_cfg=spectrum_cfg,
        cuts_cfg=cuts_cfg,
        output_cfg=output_cfg,
        cfg=cfg
    )