"""
cbspec
------
A modular, physics‑transparent Python package for converting Telescope Array TASD or CBSD (checkerboard)
surface-detector data into an ultra‑high‑energy cosmic‑ray flux and E³J(E) spectrum.

This package provides:
    - YAML-driven configuration
    - Parquet ingestion for MC and data
    - Automatic tree-type detection (resTree vs. tTlfit)
    - TA-style quality cuts
    - Energy binning in log10(E/eV)
    - Aperture and exposure calculation
    - Feldman-Cousins confidence intervals
    - Flux and E³J(E) spectrum calculation
    - Publication-quality plotting
    - Text + JSON logging
    - CLI entry point: `python -m cbspec`

The main entry point for programmatic use is `run_pipeline` in `main.py`
"""

from cbspec.main import run_pipeline

__all__ = [
    "run_pipeline",
]