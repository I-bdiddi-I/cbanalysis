"""
All data ingestion and processing for cbprocess pipeline is handled here.

This module performs the following tasks:
    1. Read MC and data parquet files batch-wise
    2. Detect the tree type automatically:
        - resTree   → TASD standard reconstruction
        - tTlfit    → (need to remember what this is)
    3. Apply FD energy correction and compute:
        - log10(E_recon/eV)
        - log10(E_thrown/eV) for MC
    4. Apply TA-style quality cuts (configurable in YAML).
    5. Accumulate:
        - reconstructed MC log10(E/eV)
        - reconstructed data log10(E/eV)
        - thrown MC log10(E/eV)
    6. Log all steps to text + JSON logs
"""

from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
import pandas as pd

from cbanalysis.utils.data_classes import QualityCuts
from cbanalysis.utils.constants import fd_energy_corr, EeV_corr
from cbanalysis.utils.logging_utils import RunLogger


def apply_quality_cuts(
        df,
        theta_corr,
        s,
        sc,
        dsc,
        ngsd,
        bdist,
        ldf,
        gf,
        cuts: QualityCuts
):
    """
    Apply TA-style quality cuts using thresholds from the YAML configuration file.
    :param df: pandas.DataFrame
               The full parquet batch
    :param theta_corr: float
                       Array-specific zenith-angle correction:
                        - TASD → 0.5°
                        - CBSD → 1.0°
    :param s: int
              Index for selecting the correct reconstruction branch:
                - resTree → 2
                - tTlfit → 1
    :param sc: array-like
               S800 values (branch dependent)
    :param dsc: array-like
                ΔS800 values (branch dependent)
    :param ngsd: array-like
                 Number of good surface detectors
    :param bdist: array-like
                  Detector array border distance in meters
    :param ldf: array-like
                LDF χ² values
    :param gf: array-like
               Geometry χ² values
    :param cuts: QualityCuts
                 Dataclass containing all cut thresholds
    :return df.loc[mask]: pandas.DataFrame
                          Subset of df passing all cuts
    """

    # Reconstructed zenith angles with array-specific correction
    theta = df["theta"].str[s] + theta_corr

    # Fractional S800
    fs800 = dsc / sc

    # Pedestal error
    pderr = df["pderr"].str[s]

    # Boolean mask for all cuts
    mask = (
        (ngsd >= cuts.number_of_good_sd)
        & (theta < cuts.theta_deg)
        & (bdist >= cuts.boarder_dist_m)
        & (gf < cuts.geometry_chi2)
        & (ldf < cuts.ldf_chi2)
        & (pderr < cuts.ped_error)
        & (fs800 < cuts.frac_s800)
    )

    # Returns the filtered DataFrame
    return df.loc[mask]

def process_batch(df, array_type, j_index, comp_df, cuts: QualityCuts, batch_idx, logger: RunLogger):
    """
    Processes parquet data each batch.

    Steps:
    1. Detect tree type (resTree or tTlfit)
    2. Apply FD energy correction
    3. Compute:
        - log10(E_recon/eV)
        - log10(E_thrown/eV) for MC
    4. Apply quality cuts (returns cdata)
    5. Append accepted log energies to comp_df
    6. Log batch progress

    :param df: pandas.DataFrame
               The parquet batch
    :param array_type: str
                       "TASD" or "CBSD"
    :param j_index: int
                    0 → MC file
                    1 → data file.
    :param comp_df: list of pandas.DataFrame
                    Accumulators:
                        comp_df[0] → MC reconstructed log10(E/eV)
                        comp_df[1] → data reconstructed log10(E/eV)
                        comp_df[2] → MC thrown log10(E/eV)
    :param cuts: QualityCuts
                 Quality cut thresholds
    :param batch_idx: int
                      Current batch index
    :param logger: RunLogger
                   Handles text + JSON logging
    :return comp_df: pandas.DataFrame
                     Updated comp_df entries
    :return cdata: pandas.DataFrame
                   Filtered DataFrame of current batch
    """

    logger.log_text(f"Processing batch {batch_idx} for file index {j_index}...")
    logger.log_json(event="batch_start", batch=batch_idx, file_index=j_index)

    # Array-specific zenith-angle correction
    theta_corr = 0.5 if array_type == "TASD" else 1.0

    # Detect tree type and extract variables
    if "energy" in df.columns:
        tree_type = "resTree"
        s = 2 # branch index
        en = df['energy'].str[0] / fd_energy_corr
        sc = df['sc'].str[0]
        dsc = df['dsc'].str[0]
        ngsd = df['nstclust']
        bdist = df['bdist'] * 1000 # km → m
        ldf = df['ldfchi2'].str[0]
        gf = df['gfchi2'].str[2]

    elif "energy_s800_p" in df.columns:
        tree_type = "tTlfit"
        s = 1
        df['energy'] = df['energy_s800_p']
        en = df['energy'] / fd_energy_corr
        sc = df['sc']
        dsc = df['dsc']
        ngsd = df['ngsd']
        bdist = df['bdist']
        ldf = df['ldfchi2pdof']
        gf = df['gfchi2pdof'].str[1]

    else:
        raise ValueError("Unknown tree type: no energy column found")

    # Log the detected tree type
    logger.log_text(f"Detected tree type: {tree_type}")
    logger.log_json(event="tree_type", value=tree_type, batch=batch_idx)

    # MC true energy
    mcen = df["mcenergy"] / fd_energy_corr

    # Compute log10 energies
    df['logen'] = np.log10(en) + EeV_corr
    df['mclogen'] = np.log10(mcen) + EeV_corr

    # Save uncut MC thrown energies (only for j_index == 0 MC file)
    if j_index == 0:
        comp_df[-1] = pd.concat([comp_df[-1], df['mclogen']], ignore_index=True)

    # Apply quality cuts
    cdata = apply_quality_cuts(
        df=df,
        theta_corr=theta_corr,
        s=s,
        sc=sc,
        dsc=dsc,
        ngsd=ngsd,
        bdist=bdist,
        ldf=ldf,
        gf=gf,
        cuts=cuts,
    )

    # Append reconstructed log10(E) from accepted events
    comp_df[j_index] = pd.concat([comp_df[j_index], cdata["logen"]], ignore_index=True)

    # Events accepted this batch
    accepted_now = len(cdata)

    logger.log_text(f"Number of accepted events in current loop: {accepted_now}")
    logger.log_json(event="batch_end", batch=batch_idx, accepted=accepted_now)

    return comp_df[j_index], comp_df[-1], cdata


def set_up_energy_array(infiles, array_type, cuts: QualityCuts, logger: RunLogger):
    """
    Read MC and data parquet files and return:
        mc_array            = MC reconstructed log10(E/eV) np.ndarray
        dt_array            = data reconstructed log10(E/eV) np.ndarray
        mc_thrown_array     = MC thrown log10(E/eV) np.ndarray

    Includes print statements:
        - Input file
        - RecordBatch
        - Accepted events per batch
        - Running total per file

    :param infiles: list of Path
                    [MC_file, data_file]
    :param array_type: str
                       "TASD" or "CBSD"
    :param cuts: QualityCuts
                 Quality cut thresholds
    :param logger: RunLogger
                   Handles text + JSON logging
    :return mc_array: np.ndarray
    :return dt_array: np.ndarray
    :return mc_thrown_array: np.ndarray
    """
    comp_df = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]

    for j, infile in enumerate(infiles):
        logger.log_text(f"Input File: {infile}")
        logger.log_json(event="input_file", file=str(infile), index=j)

        count = 0 # running total of accepted events in this file
        parquet_file = pq.ParquetFile(infile)

        # Iterate through parquet batches
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=160000)):
            df = batch.to_pandas()

            # Process batch
            comp_df[j], comp_df[-1], cdata = process_batch(
                df=df,
                array_type=array_type,
                j_index=j,
                comp_df=comp_df,
                cuts=cuts,
                batch_idx=batch_idx,
                logger=logger,
            )

            # Update running total
            accepted_now = len(cdata)
            count += accepted_now

            logger.log_text(f"Total number of accepted events from {infile}: {count}")
            logger.log_json(event="running_total", file=str(infile), total=count)

    # Convert accumulated DataFrames to numpy arrays
    mc_array = comp_df[0].to_numpy()
    dt_array = comp_df[1].to_numpy()
    mc_thrown_array = comp_df[-1].to_numpy()

    return mc_array, dt_array, mc_thrown_array