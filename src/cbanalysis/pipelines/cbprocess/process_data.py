"""
All data ingestion and processing for cbprocess pipeline is handled here.

This module performs the following tasks:
    1. Reads MC and data parquet files batch-wise
    2. Detects the tree type automatically:
        - resTree   → TASD standard reconstruction
        - tTlfit    → (need to remember what this is)
    3. Applies FD energy correction and compute:
        - log10(E_recon/eV)
        - log10(E_thrown/eV) for MC
    4. Applies TA-style quality cuts (configurable in YAML)
    5. Produces period-split energy arrays (in log10(E/eV)):
        - MC reconstructed (full cuts)
        - DT reconstructed (full cuts)
        - MC thrown (full cuts)
        - MC thrown (no cuts)
        - MC thrown (geom cuts)
    6. Preserves full jagged event table (Awkward) for DT after cuts
    7. Computes date ranges for each period (yymmdd_min, yymmdd_max)
    8. Logs all steps to text + JSON logs
"""


import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import awkward as ak
import pyarrow as pa

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
    theta = df["theta_corr"]

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

def process_batch(df, array_type, j_index, cuts: QualityCuts, batch_idx, logger: RunLogger):
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

    # Compute corrected zenith angle and save it
    df["theta_corr"] = df["theta"].str[s] + theta_corr

    # MC true energies
    mcen = df["mcenergy"] / fd_energy_corr

    # Compute log10 energies
    df['logen'] = np.log10(en) + EeV_corr
    df['mclogen'] = np.log10(mcen) + EeV_corr

    # Save uncut and geom cut MC thrown energies (only for j_index == 0 MC file)
    if j_index == 0:
        thrown_nocuts = df["mclogen"]
        geom_mask = (
                (df["theta_corr"] < cuts.theta_deg) &
                (bdist >= cuts.boarder_dist_m)
        )
        thrown_geomcuts = df.loc[geom_mask, "mclogen"]
    else:
        thrown_nocuts = pd.Series([], dtype=float)
        thrown_geomcuts = pd.Series([], dtype=float)


    # Apply full quality cuts
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

    if j_index == 0:
        # MC thrown energies that pass full cuts
        thrown_fullcuts = df.loc[cdata.index, "mclogen"]
    else:
        thrown_fullcuts = pd.Series([], dtype=float)

    # Append reconstructed log10(E) from accepted events
    #comp_df[j_index] = pd.concat([comp_df[j_index], cdata["logen"]], ignore_index=True)
    recon_fullcuts = cdata["logen"]

    # Events accepted this batch
    accepted_now = len(cdata)
    logger.log_text(f"Number of accepted events in current loop: {accepted_now}")
    logger.log_json(event="batch_end", batch=batch_idx, accepted=accepted_now)

    # Extract yymmdd for period splitting
    dates = df["yymmdd"]
    #print(years)

    return (
        recon_fullcuts,
        thrown_nocuts,
        thrown_fullcuts,
        thrown_geomcuts,
        cdata,
        dates,
    )


def set_up_energy_array(infiles, array_type, cuts: QualityCuts, logger: RunLogger, periods):
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
    :param periods: int
    :return mc_array: np.ndarray
    :return dt_array: np.ndarray
    :return mc_thrown_array: np.ndarray
    """
    # comp_df = [MC recon, DT recon, MC thrown]
    #comp_df = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]

    # Period accumulators
    energy = {
        "mc_recon":  [[] for _ in range(periods)],
        "dt_recon":  [[] for _ in range(periods)],
        "mc_thrown_nocuts": [[] for _ in range(periods)],
        "mc_thrown_fullcuts": [[] for _ in range(periods)],
        "mc_thrown_geomcuts": [[] for _ in range(periods)],
    }

    # Passed cuts data (stored as Awkward)
    passed_cuts = []

    # Period ranges
    period_ranges = [(None, None) for _ in range(periods)]

    # Determine year range for period splitting
    global_date_min = None
    global_date_max = None

    for infile in infiles:
        date_min = None
        date_max = None

        logger.log_text(f"Scanning date range for input file {infile}...")
        parquet_file = pq.ParquetFile(infile)

        for rg in range(parquet_file.num_row_groups):
            yymmdd_rg = parquet_file.read_row_group(rg, columns=["yymmdd"]).to_pandas()["yymmdd"]

            if date_min is None:
                date_min = yymmdd_rg.min()
                date_max = yymmdd_rg.max()
            else:
                date_min = min(date_min, yymmdd_rg.min())
                date_max = max(date_max, yymmdd_rg.max())

            if global_date_min is None:
                global_date_min = yymmdd_rg.min()
                global_date_max = yymmdd_rg.max()
            else:
                global_date_min = min(global_date_min, yymmdd_rg.min())
                global_date_max = max(global_date_max, yymmdd_rg.max())


        total_span = date_max - date_min
        logger.log_text(f"Date range: {date_min} → {date_max} ({total_span // 10000} years)")
        logger.log_json(event=f"total_span_{infile}", value=int(total_span))

    global_span = global_date_max - global_date_min
    logger.log_text(f"Unified date range: {global_date_min} → {global_date_max} ({global_span // 10000} years)")
    logger.log_json(event=f"global_span_", value=int(global_span))


    for j, infile in enumerate(infiles):
        logger.log_text(f"Processing file: {infile}")
        logger.log_json(event="input_file", file=str(infile), index=j)

        parquet_file = pq.ParquetFile(infile)
        count = 0  # running total of accepted events in this file

        # Iterate through parquet batches
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=160000)):
            df = batch.to_pandas()

            # Process batch
            (
                recon_fullcuts,
                thrown_nocuts,
                thrown_fullcuts,
                thrown_geomcuts,
                cdata,
                dates,
            ) = process_batch(
                df=df,
                array_type=array_type,
                j_index=j,
                cuts=cuts,
                batch_idx=batch_idx,
                logger=logger,
            )

            # Period splitting (energies only, per file)
            for k in range(periods):
                start_date = global_date_min + (k * (global_span // periods))
                end_date = (
                    global_date_min + ((k + 1) * (global_span // periods))
                    if k < periods - 1
                    else global_date_max + 1
                )
                #print(start, end)
                mask = (dates >= start_date) & (dates < end_date)

                # MC recon (full cuts), MC thrown (uncut), MC thrown (geom cuts)
                if j == 0:
                    energy["mc_recon"][k].extend(recon_fullcuts.loc[mask].tolist())
                    energy["mc_thrown_nocuts"][k].extend(thrown_nocuts.loc[mask].tolist())
                    energy["mc_thrown_fullcuts"][k].extend(thrown_fullcuts.loc[mask].tolist())
                    energy["mc_thrown_geomcuts"][k].extend(thrown_geomcuts.loc[mask].tolist())
                    #print(mc_recon)

                # DT recon (full cuts)
                if j == 1:
                    energy["dt_recon"][k].extend(recon_fullcuts.loc[mask].tolist())

                    if len(cdata.loc[mask]) > 0:
                        start_data = cdata.loc[mask, "yymmdd"].min()
                        end_data = cdata.loc[mask, "yymmdd"].max()

                        if period_ranges[k][0] is None:
                            period_ranges[k] = (start_data, end_data)
                        else:
                            period_ranges[k] = (
                                min(period_ranges[k][0], start_data),
                                max(period_ranges[k][1], end_data)
                            )

                logger.log_text(
                    f"Period {k+1}: MC={len(energy["mc_recon"][k+1])}, DT={len(energy["dt_recon"][k+1])}"
                )

            # Save all DT that has passed cuts
            if j == 1:
                table = pa.Table.from_pandas(cdata, preserve_index=False)
                passed_cuts.append(ak.from_arrow(table))

            # Update running total
            accepted_now = len(cdata)
            count += accepted_now
            logger.log_text(f"Total number of accepted events from {infile}: {count}")
            logger.log_json(event="running_total", file=str(infile), total=count)

    # Combine passed cut data
    passed_cuts_df = ak.concatenate(passed_cuts) if passed_cuts else ak.Array([])

    return {
        "energy": energy,
        "passed_cuts_df": passed_cuts_df,
        "period_ranges": period_ranges,
    }