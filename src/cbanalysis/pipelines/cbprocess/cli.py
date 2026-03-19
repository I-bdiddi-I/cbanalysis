"""
CLI entry point for the cbprocess pipeline.

Usage:
    $ cbprocess [options]
Options:
    --config path/to/cbprocess.yaml
    --array_type TASD|CBSD
"""

import argparse
from pathlib import Path

from cbanalysis.utils.load_config import load_config
from cbanalysis.pipelines.cbprocess.main import run_cbprocess

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "config" / "cbprocess.yaml"

def parse_args():
    parser = argparse.ArgumentParser(description="Run cbprocess preprocessing pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (defualt: config/cbprocess.yaml)",
    )
    parser.add_argument(
        "--array_type",
        type=str,
        choices=["TASD", "CBSD"],
        help="Override array type (TASD or CBSD).",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    config_path = Path(args.config) if args.config is not None else DEFAULT_CONFIG_PATH

    array_cfg, spectrum_cfg, cuts_cfg, output_cfg, cfg = load_config(config_path)

    if args.array_type is not None:
        array_cfg.array_type = args.array_type

    run_cbprocess(
        array_cfg=array_cfg,
        spectrum_cfg=spectrum_cfg,
        cuts_cfg=cuts_cfg,
        output_cfg=output_cfg,
        cfg=cfg,
    )