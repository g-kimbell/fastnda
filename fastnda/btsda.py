"""Functions for generating test data from BTSDA.

Export data using BTSDA.

Make sure variables have SMALLEST UNITS, as BTSDA text export removes precision.

Make sure the following columns are included in export:
    - Cycle Index
    - Step Index
    - Step Type
    - Time (in ms)
    - Total Time (in ms)
    - Current (in µA)
    - Voltage (in mV)
    - Capacity (in mAs)
    - Energy (in mWs)
    - Date (in hh:mm:ss.ms)

Use the same file name as the .nda or .ndax file, only change extension to csv.

Run the csv through the function btsda_csv_to_parquet(...).

This will create a file with extension .parquet.

This can be used in pytest when included with a .nda/.ndax with the same stem in
the same folder.
"""

import re
from pathlib import Path

import polars as pl

dtypes = {
    "Cycle Index": pl.UInt32,
    "Step Index": pl.UInt32,
    "Step Type": pl.Categorical,
    "Time": pl.Float64,
    "Total Time": pl.Float64,
    "Current(uA)": pl.Float32,
    "Voltage(mV)": pl.Float32,
    "Capacity(mAs)": pl.Float32,
    "Energy(mWs)": pl.Float32,
    "Date": pl.Datetime("ms"),
    "Step Count": pl.UInt32,
}


def _time_str_to_float(time_str: str) -> float:
    """Convert hh:mm:ss.ms to (float64) seconds."""
    h, m, s, ms = re.split(r"[:.]", time_str)
    return float(h) * 3600 + float(m) * 60 + float(s) + float("0." + ms)


def btsda_csv_to_parquet(csv_file: str | Path, out_file: str | Path | None = None) -> pl.DataFrame:
    """Convert csv from BTSDA into Parquet file.

    Export data to CSV using BTSDA, export the "recording layer".

    Make sure variables have SMALLEST UNITS, as BTSDA text export removes precision.

    Make sure the following columns are included in export:
        - Cycle Index
        - Step Index
        - Step Type
        - Time (in ms)
        - Total Time (in ms)
        - Current (in µA)
        - Voltage (in mV)
        - Capacity (in mAs)
        - Energy (in mWs)
        - Date (in hh:mm:ss.ms)
        - Step start and end identification
    """
    csv_file = Path(csv_file)
    out_file = csv_file.with_suffix(".parquet") if out_file is None else Path(out_file)
    df = pl.read_csv(csv_file, infer_schema_length=10000, encoding="cp1252")
    df = df.with_columns(
        pl.col("Time").map_elements(_time_str_to_float, return_dtype=pl.Float64),
        pl.col("Total Time").map_elements(_time_str_to_float, return_dtype=pl.Float64),
        pl.col("Date").str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f", time_unit="ms"),
        pl.col("Step Index").diff().fill_null(0).ne(0).cum_sum().alias("Step"),
        pl.col("Step start and end identification ").eq(0).cast(pl.UInt32).fill_null(0).cum_sum().alias("Step Count"),
    )
    df = df.rename({"Current(µA)": "Current(uA)"})
    aux_cols = [c for c in df.columns if re.match(r"^[TtHV]\d+", c)]
    df = df.select(list(dtypes.keys()) + aux_cols)
    df = df.cast({**dtypes, **dict.fromkeys(aux_cols, pl.Float32)})
    # Brotli seems to have best file size, we don't care about speed
    # Trying to keep repo small with lots of test data
    df.write_parquet(out_file, compression="brotli", compression_level=11)
    return df


def btsda9_xlsx_to_parquet(xlsx_file: str | Path, out_file: str | Path | None = None) -> pl.DataFrame:
    """Convert BTS 9.0 files to parquet."""
    xlsx_file = Path(xlsx_file)
    out_file = xlsx_file.with_suffix(".parquet") if out_file is None else Path(out_file)
    df = pl.read_excel(xlsx_file, infer_schema_length=1000, sheet_name="record")
    if "FlowTimer" in df.columns:
        df = df.rename({"FlowTimer": "Time"})
    df = df.with_columns(
        [
            pl.col("FlowTimer").map_elements(_time_str_to_float, return_dtype=pl.Float64).alias("Total Time"),
            pl.col("RtcTimer").str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f", time_unit="ms").alias("Date"),
            (pl.col("Current(mA)") * 1000).alias("Current(uA)"),
            (pl.col("Voltage(V)") * 1000).alias("Voltage(mV)"),
            (pl.col("CurrStep_Capacity(mAh)") * 3600).alias("Capacity(mAs)"),
            (pl.col("CurrStep_Energy(mWh)") * 3600).alias("Energy(mWs)"),
            pl.col("Step ID").alias("Step Index"),  # Don't know the difference here
            pl.col("Step ID").alias("Step Count"),
            pl.col("Cycle ID").alias("Cycle Index"),
            pl.col("DataSerial").alias("Index"),
            pl.col("StepType").alias("Step Type"),
            pl.col("AuxTemp1(Start Temperature)").alias("T1"),
        ]
    )
    max_df = (
        df.group_by("Step ID")
        .agg(pl.col("Total Time").max().alias("Max Total Time"))
        .sort("Step ID")
        .with_columns(pl.col("Max Total Time").shift(1).fill_null(0))
    )
    df = df.join(max_df, on="Step ID", how="left").with_columns(
        (pl.col("Total Time") - pl.col("Max Total Time")).alias("Time")
    )
    dtypes_here = {k: v for k, v in dtypes.items() if k in df.columns}
    aux_cols = [c for c in df.columns if re.match(r"^[TtHV]\d+", c)]
    df = df.select(list(dtypes_here.keys()) + aux_cols)
    df = df.cast({**dtypes_here, **dict.fromkeys(aux_cols, pl.Float32)})
    df.write_parquet(out_file, compression="brotli", compression_level=11)
    return df


def btsda91_xlsx_to_parquet(xlsx_file: str | Path, out_file: str | Path | None = None) -> pl.DataFrame:
    """Convert BTS 9.1 files to parquet."""
    xlsx_file = Path(xlsx_file)
    out_file = xlsx_file.with_suffix(".parquet") if out_file is None else Path(out_file)
    df = pl.read_excel(xlsx_file, infer_schema_length=1000, sheet_name="record")
    df_step = pl.read_excel(xlsx_file, infer_schema_length=1000, sheet_name="step")
    df = df.join(df_step, on="Step ID")
    df = df.with_columns(
        [
            pl.col("Time(h:m:s:ms:us)").map_elements(_time_str_to_float, return_dtype=pl.Float64).alias("Time"),
            pl.col("Realtime")
            .str.replace_all(".", "", literal=True)
            .str.to_datetime(format="%Y-%m-%d %H:%M:%S:%6f", time_unit="ms")
            .alias("Date"),
            (pl.col("Current(mA)") * 1000).alias("Current(uA)"),
            (pl.col("Voltage(V)") * 1000).alias("Voltage(mV)"),
            (pl.col("Cap(mAh)") * 3600).alias("Capacity(mAs)"),
            (pl.col("Energy(mWh)") * 3600).alias("Energy(mWs)"),
            pl.col("Step ID").alias("Step Count"),
            pl.col("OriStepID").alias("Step Index"),
            pl.col("CycleID").alias("Cycle Index"),
            pl.col("Record ID").alias("Index"),
        ]
    )
    max_df = (
        df.group_by("Step Count")
        .agg(
            pl.col("Time").max().alias("Max Step Time"),
            pl.col("Capacity(mAs)").last().alias("Last Capacity(mAs)"),
            pl.col("Energy(mWs)").last().alias("Last Energy(mWs)"),
        )
        .sort("Step Count")
        .with_columns(
            pl.col("Max Step Time").shift(1).fill_null(0).cum_sum(),
            pl.col("Last Capacity(mAs)").shift(1).fill_null(0),
            pl.col("Last Energy(mWs)").shift(1).fill_null(0),
        )
    )
    df = df.join(max_df, on="Step Count", how="left").with_columns(
        (pl.col("Time") + pl.col("Max Step Time")).alias("Total Time"),
        pl.col("Capacity(mAs)") - pl.col("Last Capacity(mAs)").alias("Capacity(mAs)"),
        pl.col("Energy(mWs)") - pl.col("Last Energy(mWs)").alias("Energy(mWs)"),
    )
    dtypes_here = {k: v for k, v in dtypes.items() if k in df.columns}
    aux_cols = [c for c in df.columns if re.match(r"^[TtHV]\d+", c)]
    df = df.select(list(dtypes_here.keys()) + aux_cols)
    df = df.cast({**dtypes_here, **dict.fromkeys(aux_cols, pl.Float32)})
    df.write_parquet(out_file, compression="brotli", compression_level=11)
    return df
