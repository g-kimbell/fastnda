"""Module to read Neware NDAX files."""

import logging
import re
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
from defusedxml import ElementTree

from fastnda.dicts import (
    aux_chl_type_columns,
    multiplier_dict,
)
from fastnda.utils import _count_changes

logger = logging.getLogger(__name__)


def read_ndax(file: str | Path) -> pl.DataFrame:
    """Read data from a Neware .ndax zipped file.

    Args:
        file: Path to .ndax file to read
        software_cycle_number: Regenerate the cycle number field
        cycle_mode: Selects how the cycle is incremented.
            'chg': (Default) Sets new cycles with a Charge step following a Discharge.
            'dchg': Sets new cycles with a Discharge step following a Charge.
            'auto': Identifies the first non-rest state as the incremental state.

    Returns:
        DataFrame containing all records in the file
        Metadata dictionary with file version and other info

    """
    zf = zipfile.PyZipFile(str(file))

    # Find all auxiliary channel files
    # Auxiliary files files need to be matched to entries in TestInfo.xml
    # Sort by the numbers in the filename, assume same order in TestInfo.xml
    aux_data = []
    for f in zf.namelist():
        m = re.search(r"data_AUX_(\d+)_(\d+)_(\d+)\.ndc", f)
        if m:
            aux_data.append((f, list(map(int, m.groups()))))
        else:
            m = re.search(r".*_(\d+)\.ndc", f)
            if m:
                aux_data.append((f, [int(m.group(1)), 0, 0]))

    # Sort by the three integers
    aux_data.sort(key=lambda x: x[1])
    aux_filenames = [f for f, _ in aux_data]

    # Find all auxiliary channel dicts in TestInfo.xml
    aux_dicts: list[dict] = []
    if aux_filenames:
        try:
            step = zf.read("TestInfo.xml").decode("gb2312")
            test_info = ElementTree.fromstring(step).find("config/TestInfo")
            if test_info is not None:
                aux_dicts.extend(
                    {k: int(v) if v.isdigit() else v for k, v in child.attrib.items()}
                    for child in test_info
                    if "aux" in child.tag.lower()
                )
        except Exception:
            logger.exception("Aux files found, but could not read TestInfo.xml!")

    # ASSUME channel files are in the same order as TestInfo.xml, map filenames to dicts
    if len(aux_dicts) == len(aux_filenames):
        aux_ch_dict = dict(zip(aux_filenames, aux_dicts, strict=True))
    else:
        aux_ch_dict = {}
        logger.critical("Found a different number of aux channels in files and TestInfo.xml!")

    # Extract and parse all of the .ndc files into dataframes in parallel
    files_to_read = ["data.ndc", "data_runInfo.ndc", "data_step.ndc", *aux_filenames]
    dfs = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(extract_and_bytes_to_df, zf, fname): fname for fname in files_to_read}
        for future in as_completed(futures):
            fname, df = future.result()
            if df is not None:
                dfs[fname] = df

    if "data.ndc" not in dfs:
        msg = "File type not yet supported!"
        raise NotImplementedError(msg)

    df = dfs["data.ndc"]

    # 'runInfo' contains times, capacities, energies, and needs to be forward-filled/interpolated
    if "data_runInfo.ndc" in dfs:
        df = df.join(dfs["data_runInfo.ndc"], how="left", on="index")
        df = _data_interpolation(df)

        # 'step' contains cycle count, step index, status for each step
        if "data_step.ndc" in dfs:
            df = df.join(dfs["data_step.ndc"], how="left", on="step_count")

    # Merge the aux data if it exists
    for i, (f, aux_dict) in enumerate(aux_ch_dict.items()):
        if f not in dfs:
            continue
        aux_df = dfs[f]

        # Get aux ID, use -i if not present to avoid conflicts
        aux_id = aux_dict.get("AuxID", -i)

        # If ? column exists, rename name by ChlType (T, t, H)
        if "?" in aux_df.columns and aux_dict.get("ChlType") in aux_chl_type_columns:
            col = aux_chl_type_columns[aux_dict["ChlType"]]
            aux_df = aux_df.rename({"?": f"aux{aux_id}_{col}"})
        else:  # Otherwise just append aux ID to column names
            aux_df = aux_df.rename({col: f"aux{aux_id}_{col}" for col in aux_df.columns if col not in ["index"]})
        df = df.join(aux_df, how="left", on="index")

    return df


def read_ndax_metadata(file: str | Path) -> dict[str, str | float]:
    """Read metadata from VersionInfo.xml and Step.xml in a Neware .ndax file."""
    with zipfile.PyZipFile(str(file)) as zf:
        metadata = {}
        # Read version information
        try:
            version_info = zf.read("VersionInfo.xml").decode("gb2312")
            config = ElementTree.fromstring(version_info).find("config/ZwjVersion")
            if config is not None:
                metadata["Server version"] = config.attrib["SvrVer"]
                metadata["Client version"] = config.attrib["CurrClientVer"]
                metadata["Control unit version"] = config.attrib["ZwjVersion"]
                metadata["Tester version"] = config.attrib["MainXwjVer"]
        except Exception:
            logger.exception("Failed to read VersionInfo.xml")

        # Read active mass
        try:
            step = zf.read("Step.xml").decode("gb2312")
            scq = ElementTree.fromstring(step).find("config/Head_Info/SCQ")
            if scq is not None:
                active_mass = float(scq.attrib["Value"])
                metadata["Active mass"] = active_mass / 1000
        except Exception:
            logger.exception("Failed to read Step.xml")

    return metadata


def extract_and_bytes_to_df(zf: zipfile.ZipFile, filename: str) -> tuple[str, pl.DataFrame | None]:
    """Extract .ndc from a zipfile and reads it into a DataFrame."""
    if filename in zf.namelist():
        buf = zf.read(filename)
        return filename, read_ndc(buf)
    return filename, None


def _data_interpolation(df: pl.DataFrame) -> pl.DataFrame:
    """Forward fill and interpolate missing data in the DataFrame."""
    # Get time by forward filling differences
    df = (
        df.with_columns(
            [
                pl.col("step_time_s").is_null().alias("nan_mask"),
                pl.col("step_time_s").is_not_null().cum_sum().shift(1).fill_null(0).alias("group_idx"),
                pl.col(
                    "dt",
                    "step_count",
                    "step_time_s",
                    "unix_time_s",
                    "charge_capacity_mAh",
                    "discharge_capacity_mAh",
                    "charge_energy_mWh",
                    "discharge_energy_mWh",
                ).fill_null(strategy="forward"),
            ]
        )
        .with_columns(
            [
                (pl.col("dt").cum_sum().over("group_idx") * (pl.col("nan_mask"))).alias("cdt"),
                ((pl.col("dt") * pl.col("current_mA") / 3600).cum_sum().over("group_idx") * pl.col("nan_mask")).alias(
                    "inc_capacity"
                ),
                (
                    (pl.col("dt") * pl.col("voltage_V") * pl.col("current_mA") / 3600).cum_sum().over("group_idx")
                    * pl.col("nan_mask")
                ).alias("inc_energy"),
            ]
        )
        .with_columns(
            [
                (pl.col("step_time_s") + pl.col("cdt")).alias("step_time_s"),
                (pl.col("unix_time_s") + pl.col("cdt")).alias("unix_time_s"),
                (pl.col("charge_capacity_mAh").abs() + pl.col("inc_capacity").clip(lower_bound=0)).alias(
                    "charge_capacity_mAh"
                ),
                (pl.col("discharge_capacity_mAh").abs() - pl.col("inc_capacity").clip(upper_bound=0)).alias(
                    "discharge_capacity_mAh"
                ),
                (pl.col("charge_energy_mWh").abs() + pl.col("inc_energy").clip(lower_bound=0)).alias(
                    "charge_energy_mWh"
                ),
                (pl.col("discharge_energy_mWh").abs() - pl.col("inc_energy").clip(upper_bound=0)).alias(
                    "discharge_energy_mWh"
                ),
            ]
        )
        .drop(["nan_mask", "group_idx", "cdt", "inc_capacity", "inc_energy", "dt"])
    )

    # Sanity checks
    if (df["unix_time_s"].diff() < 0).any():
        logger.warning(
            "IMPORTANT: This ndax has negative jumps in the 'timestamp' column! "
            "This can sometimes happen in the ndax file itself. "
            "Use the 'Time' column for analysis.",
        )

    return df


def read_ndc(buf: bytes) -> pl.DataFrame:
    """Read electrochemical data from a Neware ndc binary file.

    Args:
        buf: Bytes object for the .ndc file to read
    Returns:
        DataFrame containing all records in the file

    """
    # Get ndc file version and filetype
    ndc_filetype = int(buf[0])
    ndc_version = int(buf[2])
    reader = NDC_READERS.get((ndc_version, ndc_filetype))
    if reader is None:
        msg = f"ndc version {ndc_version} filetype {ndc_filetype} is not yet supported!"
        raise NotImplementedError(msg) from None
    logger.info("Reading ndc version %d filetype %d", ndc_version, ndc_filetype)
    return reader(buf)


def _read_ndc_2_filetype_1(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("_pad1", "V8"),  # 0-7
            ("index", np.uint32),  # 8-11
            ("cycle_count", np.uint32),  # 12-15
            ("step_index", np.uint8),  # 16
            ("status", np.uint8),  # 17
            ("_pad2", "V5"),  # 18-22
            ("step_time_s", np.uint64),  # 23-30
            ("voltage_V", np.int32),  # 31-34
            ("current_mA", np.int32),  # 35-38
            ("_pad3", "V4"),  # 39-42
            ("charge_capacity_mAh", np.int64),  # 43-50
            ("discharge_capacity_mAh", np.int64),  # 51-58
            ("charge_energy_mWh", np.int64),  # 59-66
            ("discharge_energy_mWh", np.int64),  # 67-74
            ("Y", np.uint16),  # 75-76
            ("M", np.uint8),  # 77
            ("D", np.uint8),  # 78
            ("h", np.uint8),  # 79
            ("m", np.uint8),  # 80
            ("s", np.uint8),  # 81
            ("range", np.int32),  # 82-85
            ("_pad4", "V8"),  # 86-93
        ]
    )
    return (
        _bytes_to_df(buf, dtype, 5, 37, record_size=512, file_header_size=512)
        .with_columns(
            [
                pl.col("cycle_count") + 1,
                pl.col("step_time_s").cast(pl.Float64) * 1e-3,
                pl.col("voltage_V").cast(pl.Float32) * 1e-4,
                pl.col("range").replace_strict(multiplier_dict, return_dtype=pl.Float64).alias("multiplier"),
                pl.datetime(pl.col("Y"), pl.col("M"), pl.col("D"), pl.col("h"), pl.col("m"), pl.col("s")).alias(
                    "timestamp"
                ),
                _count_changes(pl.col("step_index")).alias("step_count"),
            ]
        )
        .with_columns(
            [
                pl.col("current_mA") * pl.col("multiplier"),
                (
                    pl.col(
                        ["charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh"],
                    )
                    * pl.col("multiplier")
                    / 3600
                ).cast(pl.Float32),
            ]
        )
        .drop(["Y", "M", "D", "h", "m", "s"])
    )


def _read_ndc_2_filetype_5(buf: bytes) -> pl.DataFrame:
    # This dtype is missing humudity % column - does not exist in current test data
    dtype = np.dtype(
        [
            ("_pad2", "V8"),  # 4-7
            ("index", np.uint32),  # 8-11
            ("_pad3", "V19"),  # 12 - 30
            ("voltage_V", np.int32),  # 31-34
            ("_pad4", "V6"),  # 35-40
            ("temperature_degC", np.int16),  # 41-42
            ("temperature_setpoint_degC", np.int16),  # 43-44
            ("_pad5", "V49"),  # 45-93
        ]
    )
    return _bytes_to_df(buf, dtype, 5, 37, record_size=512, file_header_size=512).with_columns(
        pl.col("voltage_V").cast(pl.Float32) / 10000,
        pl.col("temperature_degC").cast(pl.Float32) * 0.1,
        pl.col("temperature_setpoint_degC").cast(pl.Float32) * 0.1,
    )


def _read_ndc_5_filetype_1(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("_pad1", "V8"),  # 0-7
            ("index", np.uint32),  # 8-11
            ("cycle_count", np.uint32),  # 12-15
            ("step_index", np.uint8),  # 16
            ("status", np.uint8),  # 17
            ("_pad2", "V5"),  # 18-22
            ("step_time_s", np.uint64),  # 23-30
            ("voltage_V", np.int32),  # 31-34
            ("current_mA", np.int32),  # 35-38
            ("_pad3", "V4"),  # 39-42
            ("charge_capacity_mAh", np.int64),  # 43-50
            ("discharge_capacity_mAh", np.int64),  # 51-58
            ("charge_energy_mWh", np.int64),  # 59-66
            ("discharge_energy_mWh", np.int64),  # 67-74
            ("Y", np.uint16),  # 75-76
            ("M", np.uint8),  # 77
            ("D", np.uint8),  # 78
            ("h", np.uint8),  # 79
            ("m", np.uint8),  # 80
            ("s", np.uint8),  # 81
            ("range", np.int32),  # 82-85
            ("_pad4", "V1"),  # 86
        ]
    )
    return (
        _bytes_to_df(buf, dtype, 125, 56)
        .with_columns(
            [
                pl.col("cycle_count") + 1,
                pl.col("step_time_s").cast(pl.Float64) * 1e-3,
                pl.col("voltage_V").cast(pl.Float32) * 1e-4,
                pl.col("range").replace_strict(multiplier_dict, return_dtype=pl.Float64).alias("multiplier"),
                pl.datetime(pl.col("Y"), pl.col("M"), pl.col("D"), pl.col("h"), pl.col("m"), pl.col("s")).alias(
                    "timestamp"
                ),
                _count_changes(pl.col("step_index")).alias("step_count"),
            ]
        )
        .with_columns(
            [
                pl.col("current_mA") * pl.col("multiplier"),
                (
                    pl.col(
                        ["charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh"],
                    )
                    * pl.col("multiplier")
                    / 3600
                ).cast(pl.Float32),
            ]
        )
        .drop(["Y", "M", "D", "h", "m", "s"])
    )


def _read_ndc_5_filetype_5(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("_pad2", "V8"),  # 4-7
            ("index", np.uint32),  # 8-11
            ("_pad3", "V19"),  # 12 - 30
            ("voltage_V", np.int32),  # 31-34
            ("_pad4", "V6"),  # 35-40
            ("temperature_degC", np.int16),  # 41-42
            ("temperature_setpoint_degC", np.int16),  # 43-44
            ("_pad5", "V42"),  # 45-86
        ]
    )
    df = _bytes_to_df(buf, dtype, 125, 56).with_columns(
        pl.col("voltage_V").cast(pl.Float32) * 1e-4,
        pl.col("temperature_degC").cast(pl.Float32) * 0.1,
        pl.col("temperature_setpoint_degC").cast(pl.Float32) * 0.1,
    )
    # Drop empty columns
    cols_to_drop = [
        col
        for col in ["voltage_V", "temperature_degC", "temperature_setpoint_degC"]
        if df.filter(pl.col(col) != 0).is_empty()
    ]
    return df.select(pl.exclude(cols_to_drop))


def _read_ndc_11_filetype_1(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("voltage_V", "<f4"),
            ("current_mA", "<f4"),
        ]
    )
    return _bytes_to_df(buf, dtype, 132, 4).with_columns(
        [
            pl.col("voltage_V") * 1e-4,  # 0.1mV -> V
        ]
    )


def _read_ndc_11_filetype_5(buf: bytes) -> pl.DataFrame:
    header = 4096

    if buf[header + 132 : header + 133] == b"\x65":
        dtype = np.dtype(
            [
                ("mask", "<i1"),
                ("voltage_V", "<f4"),
                ("temperature_degC", "<i2"),
            ]
        )
        df = _bytes_to_df(buf, dtype, 132, 2, mask=101).with_columns(
            [
                pl.col("voltage_V") * 1e-4,  # 0.1 mV -> V
                pl.col("temperature_degC").cast(pl.Float32) * 0.1,  # 0.1'C -> 'C
                pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("index"),
            ]
        )
        # Drop empty columns
        cols_to_drop = [col for col in ["voltage_V", "temperature_degC"] if df.filter(pl.col(col) != 0).is_empty()]
        return df.select(pl.exclude(cols_to_drop))

    if buf[header + 132 : header + 133] == b"\x74":
        dtype = np.dtype(
            [
                ("_pad1", "V1"),
                ("index", "<i4"),
                ("Aux", "<i1"),
                ("_pad2", "V29"),
                ("temperature_degC", "<i2"),
                ("_pad3", "V51"),
            ]
        )
        return (
            _bytes_to_df(buf, dtype, 132, 4)
            .with_columns(
                [
                    pl.col("temperature_degC").cast(pl.Float32) * 0.1,  # 0.1'C -> 'C
                ]
            )
            .drop("Aux")
        )  # Aux channel inferred from TestInfo.xml

    msg = "Unknown file structure for ndc version 11 filetype 5."
    raise NotImplementedError(msg)


def _read_ndc_11_filetype_7(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("cycle_count", "<i4"),
            ("step_index", "<i4"),
            ("_pad1", "V16"),
            ("status", "<i1"),
            ("_pad2", "V12"),
        ]
    )
    return _bytes_to_df(buf, dtype, 132, 5).with_columns(
        [
            pl.col("cycle_count") + 1,
            pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("step_count"),
        ]
    )


def _read_ndc_11_filetype_18(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("step_time_s", "<i4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V8"),
            ("dt", "<i4"),
            ("unix_time_s", "<i4"),
            ("step_count", "<i4"),
            ("index", "<i4"),
            ("uts_ms", "<i2"),
        ]
    )
    return (
        _bytes_to_df(buf, dtype, 132, 16)
        .with_columns(
            [
                pl.col("step_time_s", "dt").cast(pl.Float64) / 1000,  # ms -> s
                pl.col("charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh")
                / 3600,  # mAs|mWs -> mAh|mWh
                (pl.col("unix_time_s") + pl.col("uts_ms") / 1000).alias("unix_time_s"),
                _count_changes(pl.col("step_count")).alias("step_count"),
            ]
        )
        .drop("uts_ms")
        .unique(subset="index", keep="last")  # Assume same rule as 14 - not sure without data
    )


def _read_ndc_14_filetype_1(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("voltage_V", "<f4"),
            ("current_mA", "<f4"),
        ]
    )
    return _bytes_to_df(buf, dtype, 132, 4).with_columns(
        [
            pl.col("current_mA") * 1000,
        ]
    )


def _read_ndc_14_filetype_5(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("?", "<f4"),  # Column name is assigned later from TestInfo.xml
        ]
    )
    return _bytes_to_df(buf, dtype, 132, 4).with_columns(
        [
            pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("index"),
        ]
    )


def _read_ndc_14_filetype_7(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("cycle_count", "<i4"),
            ("step_index", "<i4"),
            ("_pad1", "V16"),
            ("status", "<i1"),
            ("_pad2", "V12"),
        ]
    )
    return _bytes_to_df(buf, dtype, 132, 5).with_columns(
        [
            pl.col("cycle_count") + 1,
            pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("step_count"),
        ]
    )


def _read_ndc_14_filetype_18(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("step_time_s", "<i4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V8"),
            ("dt", "<i4"),
            ("unix_time_s", "<i4"),
            ("step_count", "<i4"),
            ("index", "<i4"),
            ("uts_ms", "<i2"),
            ("_pad3", "V8"),
        ]
    )
    return (
        _bytes_to_df(buf, dtype, 132, 4)
        .with_columns(
            [
                pl.col("step_time_s", "dt").cast(pl.Float64) / 1000,  # ms -> s
                pl.col("charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh")
                * 1000,  # Ah|Wh -> mAh|mWh
                (pl.col("unix_time_s") + pl.col("uts_ms") / 1000).alias("unix_time_s"),
                pl.col("step_count").diff().fill_null(1).abs().gt(0).cum_sum().alias("step_count"),
            ]
        )
        .drop("uts_ms")
        .unique(subset="index", keep="last")  # Seems like this file type keeps the last of duplicates, 17 is different
    )


def _read_ndc_16_filetype_1(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("voltage_V", "<f4"),
            ("current_mA", "<f4"),
        ]
    )
    return _bytes_to_df(buf, dtype, 132, 4).with_columns(
        [
            pl.col("voltage_V") / 10000,
            pl.col("current_mA"),
        ]
    )


def _read_ndc_16_filetype_5(buf: bytes) -> pl.DataFrame:
    header = 4096
    if buf[header + 132 : header + 133] == b"\x65":
        dtype = np.dtype(
            [
                ("mask", "<i1"),
                ("voltage_V", "<f4"),
                ("temperature_degC", "<i2"),
            ]
        )
        df = _bytes_to_df(buf, dtype, 132, 2, mask=101).with_columns(
            [
                pl.col("voltage_V") / 10000,  # 0.1 mV -> V
                pl.col("temperature_degC").cast(pl.Float32) * 0.1,  # 0.1'C -> 'C
                pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("index"),
            ]
        )
        # Drop empty columns
        cols_to_drop = [col for col in ["voltage_V", "temperature_degC"] if df.filter(pl.col(col) != 0).is_empty()]
        return df.select(pl.exclude(cols_to_drop))
    msg = "Unknown file structure for ndc version 16 filetype 5."
    raise NotImplementedError(msg)


def _read_ndc_16_filetype_7(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("cycle_count", "<i4"),
            ("step_index", "<i4"),
            ("_pad1", "V16"),
            ("status", "<i1"),
            ("_pad2", "V8"),
            ("index", "<i4"),
            ("_pad3", "V63"),
        ]
    )
    return _bytes_to_df(buf, dtype, 132, 64).with_columns(
        [
            pl.col("cycle_count") + 1,
            _count_changes(pl.col("step_index")).alias("step_count"),
        ]
    )


def _read_ndc_16_filetype_18(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("step_time_s", "<i4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V8"),
            ("dt", "<i4"),
            ("unix_time_s", "<i4"),
            ("step_count", "<i4"),
            ("index", "<i4"),
            ("uts_ms", "<i2"),
            ("_pad3", "V53"),
        ]
    )
    df = _bytes_to_df(buf, dtype, 132, 64)
    return df.with_columns(
        [
            pl.col("step_time_s", "dt").cast(pl.Float64) / 1000,
            (
                pl.col("charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh")
                / 3600
            ).cast(pl.Float32),  # mAs|mWs -> mAh|mWh
            (pl.col("unix_time_s") + pl.col("uts_ms") / 1000).alias("unix_time_s"),
        ]
    ).drop("uts_ms")


def _read_ndc_17_filetype_1(buf: bytes) -> pl.DataFrame:
    return _read_ndc_14_filetype_1(buf)


def _read_ndc_17_filetype_7(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("cycle_count", "<i4"),
            ("step_index", "<i4"),
            ("_pad1", "V16"),
            ("status", "<i1"),
            ("_pad2", "V8"),
            ("step_count", "<i4"),
            ("_pad3", "V63"),
        ]
    )
    return _bytes_to_df(buf, dtype, 132, 64).with_columns(
        [
            pl.col("cycle_count") + 1,
            _count_changes(pl.col("step_index")).alias("step_count"),
        ]
    )


def _read_ndc_17_filetype_18(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("step_time_s", "<i4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V8"),
            ("dt", "<i4"),
            ("unix_time_s", "<i4"),
            ("step_count", "<i4"),
            ("index", "<i4"),
            ("uts_ms", "<i2"),
            ("_pad3", "V53"),
        ]
    )
    return (
        _bytes_to_df(buf, dtype, 132, 64)
        .with_columns(
            [
                pl.col("step_time_s", "dt").cast(pl.Float64) / 1000,
                (
                    pl.col("charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh")
                    * 1000
                ).cast(pl.Float32),  # Ah|Wh -> mAh|mWh
                (pl.col("unix_time_s") + pl.col("uts_ms") / 1000).alias("unix_time_s"),
            ]
        )
        .drop("uts_ms")
        .unique(subset="index", keep="first")  # Seems like this file type keeps the last of duplicates, 14 is different
    )


def _bytes_to_df(
    buf: bytes,
    dtype: np.dtype,
    record_header_size: int,
    record_footer_size: int,
    record_size: int = 4096,
    file_header_size: int = 4096,
    mask: int | None = None,
) -> pl.DataFrame:
    """Read bytes into a polars DataFrame.

    Args:
        buf: Bytes object containing the binary data.
        dtype: Numpy dtype describing the record structure.
        record_header_size: Size of the record header in bytes.
        record_footer_size: Size of the record footer in bytes.
        record_size: Total size of a single record in bytes.
        file_header_size: Size of the file header in bytes.
        mask (optional): Mask to filter, assumes a column named
            "mask" and keeps rows where "mask" equals this value.

    Returns:
        DataFrame containing the records.

    """
    # Read entire file into 1 byte array nrecords x record_size
    num_records = (len(buf) - file_header_size) // record_size
    arr = np.frombuffer(buf[file_header_size:], dtype=np.int8).reshape((num_records, record_size))
    # Slice the header and footer
    arr = arr[:, record_header_size:-record_footer_size]
    # Remove padding columns
    useful_cols = [name for name in dtype.names if not name.startswith("_")]
    dtype_no_pad = dtype[useful_cols]
    arr = arr.view(dtype=dtype_no_pad)
    # Flatten
    arr = arr.reshape(-1)

    # If a mask is provided, filter the array
    if mask is not None and "mask" in arr.dtype.names:
        arr = arr[arr["mask"] == mask]
        return pl.DataFrame(arr).drop("mask")

    # If runInfo file, remove 0 index rows
    if "index" in arr.dtype.names:
        arr = arr[arr["index"] != 0]
        return pl.DataFrame(arr)

    # If step file, remove 0 step index rows
    if "step_index" in arr.dtype.names:
        arr = arr[arr["step_index"] != 0]
        return pl.DataFrame(arr)

    # If data file, remove 0.0 voltage rows and add Index column
    if "voltage_V" in arr.dtype.names:
        arr = arr[arr["voltage_V"] != 0]
        return pl.DataFrame(arr).with_columns(
            [
                pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("index"),
            ]
        )

    return pl.DataFrame(arr)


# Map NDC (version, filetype) to handler functions
NDC_READERS: dict[tuple[int, int], Callable[[bytes], pl.DataFrame]] = {
    (2, 1): _read_ndc_2_filetype_1,
    (2, 5): _read_ndc_2_filetype_5,
    (5, 1): _read_ndc_5_filetype_1,
    (5, 5): _read_ndc_5_filetype_5,
    (11, 1): _read_ndc_11_filetype_1,
    (11, 5): _read_ndc_11_filetype_5,
    (11, 7): _read_ndc_11_filetype_7,
    (11, 18): _read_ndc_11_filetype_18,
    (14, 1): _read_ndc_14_filetype_1,
    (14, 5): _read_ndc_14_filetype_5,
    (14, 7): _read_ndc_14_filetype_7,
    (14, 18): _read_ndc_14_filetype_18,
    (16, 1): _read_ndc_16_filetype_1,
    (16, 5): _read_ndc_16_filetype_5,
    (16, 7): _read_ndc_16_filetype_7,
    (16, 18): _read_ndc_16_filetype_18,
    (17, 1): _read_ndc_17_filetype_1,
    (17, 5): _read_ndc_14_filetype_5,
    (17, 7): _read_ndc_17_filetype_7,
    (17, 18): _read_ndc_17_filetype_18,
}
