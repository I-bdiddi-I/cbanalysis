"""
CLI entry point for cbefficiency pipeline.

Usage:
    $ cbefficiency [options]

Options:
    --config path/to/config.yaml
    --array_type TASD|CBSD
    --periods N
    --energy_binning min:max:bin_size
    --energy_min float
    --energy_max float
    --energy_bin_size float
    --mc_thrown_geomcuts path/to/file.csv
    --mc_thrown_fullcuts path/to/file.csv
"""

import argparse
from pathlib import Path

from cbanalysis.utils.load_config import load_config
from cbanalysis.pipelines.cbefficiency.main import run_cbefficiency

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent[4] / "config" / "cbefficiency.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Run cbefficiency pipeline.")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (defualt: config/cbefficiency.yaml)",
    )

    parser.add_argument(
        "--array_type",
        type=str,
        choices=["TASD", "CBSD"],
        help="Override array type (TASD or CBSD).",
    )

    parser.add_argument(
        "--periods",
        type=int,
        default=None,
        help="Number of time periods to split the data into (default: from YAML)."
    )

    parser.add_argument(
        "--energy_binning",
        type=str,
        default=None,
        help="Unified energy binning specification: min:max:bin_size"
    )

    parser.add_argument(
        "--energy_min",
        type=float,
        default=None,
        help="Minimum log10(E/eV) for energy binning."
    )

    parser.add_argument(
        "--energy_max",
        type=float,
        default=None,
        help="Maximum log10(E/eV) for energy binning."
    )

    parser.add_argument(
        "--energy_bin_size",
        type=float,
        default=None,
        help="Bin size in log10(E/eV) for energy binning."
    )

    parser.add_argument(
        "--mc_thrown_geomcuts",
        type=str,
        default=None,
        help="Override path to MC thrown energies (geom cuts)."
    )

    parser.add_argument(
        "--mc_thrown_fullcuts",
        type=str,
        default=None,
        help="Override path to MC thrown energies (full cuts)."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config) if args.config is not None else DEFAULT_CONFIG_PATH

    array_cfg, eff_files_cfg, eff_proc_cfg, output_cfg, cfg = load_config(config_path)

    # CLI overrides
    if args.array_type is not None:
        array_cfg.array_type = args.array_type

    # Period override
    if args.periods is not None:
        eff_files_cfg.periods = args.periods
        periods = args.periods
    else:
        periods = eff_proc_cfg.periods

    # Unified energy binning override
    if args.energy_binning is not None:
        try:
            en_min_str, en_max_str, bin_size_str = args.energy_binning.split(":")
            eff_proc_cfg.en_min = float(en_min_str)
            eff_proc_cfg.en_max = float(en_max_str)
            eff_proc_cfg.bin_size = float(bin_size_str)
        except ValueError:
            raise ValueError(
                "Invalid format for --energy_binning. Expected format: min:max:bin_size"
            )

    # Separate energy binning overrides
    if args.energy_min is not None:
        eff_proc_cfg.en_min = args.energy_min
    if args.energy_max is not None:
        eff_proc_cfg.en_max = args.energy_max
    if args.energy_bin_size is not None:
        eff_proc_cfg.bin_size = args.energy_bin_size

    # Interactive prompting for file paths and date ranges
    period_ranges = []

    print(f"\nYou selected {periods} period(s).\n")

    # Case 1: User DID specify --periods → Prompt for file paths AND date ranges for each period
    if args.periods is not None:
        print("Please enter the file paths and date ranges for each period.\n")

        for i in range(periods):
            print(f"--- Period {i+1} ---")

            geom_path = input("Path to MC thrown energies (geom cuts): ").strip()
            full_path = input("Path to MC thrown energies (full cuts): ").strip()
            date_range = input("Date range (yymmdd_start:yymmdd_end): ").strip()

            if i == 0:
                # First period overrides the main config
                eff_files_cfg.mc_thrown_geomcuts = Path(geom_path)
                eff_files_cfg.mc_thrown_fullcuts = Path(full_path)

            # Store date ranges
            period_ranges.append(date_range)
            print()

    # Case 2: User did NOT specify --periods → Only prompt for date range → File paths come from YAML or CLI overrides
    else:
        print("Please enter date range for this file.\n")
        date_range = input("Date range (yymmdd_start:yymmdd_end): ").strip()
        period_ranges.append(date_range)
        print()

        # Apply CLI overrides if provided
        if args.mc_thrown_geomcuts is not None:
            eff_files_cfg.mc_thrown_geomcuts = Path(args.mc_thrown_geomcuts)

        if args.mc_thrown_fullcuts is not None:
            eff_files_cfg.mc_thrown_fullcuts = Path(args.mc_thrown_fullcuts)

    # Save period ranges
    cfg["processing"]["period_ranges"] = period_ranges

    run_cbefficiency(
        array_cfg=array_cfg,
        eff_files_cfg=eff_files_cfg,
        eff_proc_cfg=eff_proc_cfg,
        output_cfg=output_cfg,
        cfg=cfg,
        cli_args=args,
    )

if __name__ == "__main__":
    main()