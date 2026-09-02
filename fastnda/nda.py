# Copyright © 2026, Empa.
"""Module to read Neware NDA files."""

import datetime
import logging
import mmap
import warnings
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import numpy as np
import polars as pl

from fastnda.nda_meta import _read_nda_test_info
from fastnda.utils import (
    UnverifiedFormatWarning,
    _add_total_time,
    _count_changes,
    _drop_empty,
    _range_to_mult,
    _step_sign,
)

logger = logging.getLogger(__name__)


def read_nda(file: str | Path) -> pl.DataFrame:
    """Read data from a Neware .nda binary file.

    Args:
        file: Path of .nda file to read

    Returns:
        DataFrame containing all records in the file

    """
    file = Path(file)
    with file.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        if mm.read(6) != b"NEWARE":
            msg = f"{file} does not appear to be a Neware file."
            raise ValueError(msg)
        # Parse binary data to dataframe
        df = _read_nda(mm)

    # Drop duplicate indexes and sort
    df = df.unique(subset="index")
    return df.sort(by="index")


_START_TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y.%m.%d %H:%M:%S")


def _read_nda_start_time_s(mm: mmap.mmap, nda_version: int) -> float | None:
    """Read the test-level start_time and parse it to unix seconds (UTC)."""
    start_time = _read_nda_test_info(mm, nda_version).get("start_time")
    if not start_time:
        return None
    for fmt in _START_TIME_FORMATS:
        with suppress(ValueError):
            return datetime.datetime.strptime(start_time, fmt).replace(tzinfo=datetime.timezone.utc).timestamp()
    return None


def _add_derived_unix_time(df: pl.DataFrame, start_time_s: float | None) -> pl.DataFrame:
    """Derive unix_time_s from the test start time plus test time.

    Best guess for formats that do not include unix time.
    Can drift from wall clock if the test was paused or resumed.
    """
    df = _add_total_time(df)
    if start_time_s is None:
        logger.info("No test start time found, cannot derive unix_time_s.")
        return df
    return df.with_columns((pl.col("total_time_s") + start_time_s).alias("unix_time_s"))


def _find_header(mm: mmap.mmap, header: bytes | int) -> int:
    """Get header index."""
    if isinstance(header, int):
        return header
    header_idx = mm.find(header)
    if header_idx == -1:
        msg = "Could not find start of data section."
        raise EOFError(msg)
    return header_idx


def _get_arr_from_nda(
    mm: mmap.mmap,
    header: bytes | int,
    record_len: int,
    data_len: int = 0,
) -> np.ndarray:
    """Read records from an nda file, stopping after data_len bytes if it is non-zero."""
    header_idx = _find_header(mm, header)
    available = min(data_len, len(mm) - header_idx) if data_len else len(mm) - header_idx
    num_records = available // record_len
    return np.frombuffer(mm, dtype=np.uint8, count=num_records * record_len, offset=header_idx).reshape(
        (num_records, record_len)
    )


def _view_arr(
    arr: np.ndarray,
    dtype: np.dtype,
) -> pl.DataFrame:
    """Get polars dataframe from array, dropping padding columns."""
    assert dtype.names is not None  # noqa: S101
    dtype_no_pad = dtype[[name for name in dtype.names if not name.startswith("_")]]
    arr = arr.view(dtype_no_pad).ravel()
    return pl.DataFrame(arr)


def _mask_arr(
    arr: np.ndarray,
    dtype: np.dtype,
    mask: int,
) -> pl.DataFrame:
    """Get polars dataframe from array, filtered on an identifier value."""
    return _view_arr(arr, dtype).filter(pl.col("identifier") == mask).drop("identifier")


def _nda_head_main(mm: mmap.mmap, *, pos_offset: int = 64, pos64: bool = False) -> tuple[int, int]:
    """Read the {begin, length} pointer to the main data block from the header."""
    size = 8 if pos64 else 4
    begin = int.from_bytes(mm[pos_offset : pos_offset + size], "little")
    length = int.from_bytes(mm[pos_offset + size : pos_offset + 2 * size], "little")
    return begin, length


def _nda_multiplier(mm: mmap.mmap, record_range: pl.Expr | None = None) -> pl.Expr:
    """Multiplier applied to current, capacity and energy. Fixed with optional per-row range."""
    block_begin = int.from_bytes(mm[16:20], "little")
    fixed_range = int.from_bytes(mm[block_begin + 26 : block_begin + 30], "little", signed=True)
    # If no range column, just use fixed range
    if record_range is None:
        return _range_to_mult(pl.lit(fixed_range, pl.Int32))
    # If range column, replace 0s with fixed value
    return _range_to_mult(record_range.replace(0, fixed_range))


# Byte that cannot occur in an aux column name, used to split the pivot's output names
_AUX_PIVOT_SEP = "\x00"


def _aux_pivot_name(col: str, value_cols: list[str]) -> str:
    """Name a pivoted aux column aux{channel}_{measurement}, e.g. aux1_voltage_V."""
    # Polars names a pivot of one value column by the channel
    # If there is more than one column, it names them name+sep+channel
    if len(value_cols) == 1:
        name, channel = value_cols[0], col
    else:
        name, _, channel = col.partition(_AUX_PIVOT_SEP)
    return f"aux{channel}_{name.removeprefix('aux_')}"


def _aux_column_order(channels: list[int], value_cols: list[str]) -> list[str]:
    """Merge aux column names, grouped by channel then measurement."""
    return [f"aux{channel}_{col.removeprefix('aux_')}" for channel in channels for col in value_cols]


def _merge_aux(
    df: pl.DataFrame,
    aux_df: pl.DataFrame,
) -> pl.DataFrame:
    """Merge aux left into data, renaming columns if aux channel in data."""
    if aux_df.is_empty():
        return df
    if "aux" not in aux_df.columns:
        return df.join(aux_df.unique(subset=["index"]), on="index", how="left")
    value_cols = [col for col in aux_df.columns if col not in {"index", "aux"}]
    if not value_cols:
        return df
    channels = aux_df["aux"].unique().sort().to_list()
    # Shortcut to a horizontal join if possible (pivot is slow)
    if len(channels) == 1 and len(aux_df) == len(df) and aux_df["index"].equals(df["index"]):
        names = dict(zip(value_cols, _aux_column_order(channels, value_cols), strict=True))
        return pl.concat([df, aux_df.select(value_cols).rename(names)], how="horizontal")
    # Full pivot and merge, slow but may be necessary
    aux_df = aux_df.unique(subset=["index", "aux"])
    aux_df = aux_df.pivot(index="index", on="aux", values=value_cols, separator=_AUX_PIVOT_SEP)
    aux_df.columns = [col if col == "index" else _aux_pivot_name(col, value_cols) for col in aux_df.columns]
    aux_df = aux_df.select("index", *_aux_column_order(channels, value_cols))
    return df.join(aux_df, on="index", how="left")


def _read_nda(mm: mmap.mmap) -> pl.DataFrame:
    """Figure out nda version and pass to correct reader."""
    nda_version = int(mm[14])
    reader = _NDA_READERS.get(nda_version)
    if reader is None:
        msg = f"nda version {nda_version} is not yet supported!"
        raise NotImplementedError(msg) from None
    if nda_version not in _CONFIRMED_NDA_VERSIONS:
        warnings.warn(
            f"nda_version {nda_version} has not been verified against real Neware data - results may be "
            "incorrect. If you can, please share a sample file at "
            "https://github.com/empaeconversion/fastnda/issues so we can confirm this format.",
            UnverifiedFormatWarning,
            stacklevel=2,
        )
    logger.debug("Reading nda version %d", nda_version)
    return reader(mm)


def _read_nda_1(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 1."""
    header_idx, data_len = _nda_head_main(mm)
    arr = _get_arr_from_nda(mm, header=header_idx, record_len=38, data_len=data_len)
    dtype = np.dtype(
        [
            ("index", "<u4"),
            ("cycle_count", "<u4"),  # 2 step/loop counters, handled by _count_changes
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("step_time_s", "<u4"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad1", "V8"),  # nIR, iTemp - unpopulated in all files checked
            ("capacity_mAh", "<i8"),
        ]
    )
    # nda1 uses a legacy step type enum
    # Remap to the standard codes used by STEP_TYPE_MAP
    # The remap is incomplete, needs more test data
    step_remap = {2: 3, 4: 2, 5: 4}
    multiplier = _nda_multiplier(mm)
    start_time_s = _read_nda_start_time_s(mm, 1)
    df = (
        _view_arr(arr, dtype)
        .filter(pl.col("index") != 0)
        .with_columns(
            [
                pl.col("step_time_s").cast(pl.Float64),
                pl.col("voltage_V").cast(pl.Float32) / 10000,
                pl.col("current_mA") * multiplier,
                pl.col("step_type").replace(step_remap),
                _count_changes(pl.col("step_index")).alias("step_count"),
                _count_changes(pl.col("cycle_count")),
            ]
        )
        .with_columns(  # need the step type remapped
            (
                pl.col("capacity_mAh").cast(pl.Float64)
                * multiplier
                * _step_sign(pl.col("step_type"), pl.col("current_mA"))
                / 3600
            ).cast(pl.Float32),
        )
    )
    return _add_derived_unix_time(df, start_time_s)


def _read_nda_2(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 2 (deprecated by Neware - unreachable in BTSDA)."""
    header_idx, data_len = _nda_head_main(mm, pos_offset=64)
    arr = _get_arr_from_nda(mm, header=header_idx, record_len=39, data_len=data_len)
    dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("_pad2", "V4"),  # Unknown
            ("cycle_count", "<u4"),
            ("step_type", "<u1"),
            ("step_index", "<u1"),
            ("step_time_s", "<u4"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad3", "V8"),  # Unknown
            ("capacity_mAh", "<i8"),
        ]
    )
    multiplier = _nda_multiplier(mm)
    return (
        _view_arr(arr, dtype)
        .filter(pl.col("identifier") == 0)
        .drop("identifier")
        .with_columns(
            [
                pl.int_range(1, pl.len() + 1).alias("index"),
                pl.col("step_time_s").cast(pl.Float64),
                pl.col("voltage_V").cast(pl.Float32) / 10000,
                pl.col("current_mA") * multiplier,
                (
                    pl.col("capacity_mAh").cast(pl.Float64)
                    * multiplier
                    * _step_sign(pl.col("step_type"), pl.col("current_mA"))
                    / 3600
                ).cast(pl.Float32),
                _count_changes(pl.col("step_index")).alias("step_count"),
            ]
        )
    )


def _read_nda_3(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 3 (file version 3, 4)."""
    header_idx, data_len = _nda_head_main(mm)
    arr = _get_arr_from_nda(mm, header=header_idx, record_len=43, data_len=data_len)
    dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("step_time_s", "<u4"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad1", "V8"),  # nIR, iTemp
            ("capacity_mAh", "<i8"),
            ("_pad2", "V4"),  # dwCRC32
        ]
    )
    multiplier = _nda_multiplier(mm)
    start_time_s = _read_nda_start_time_s(mm, 3)
    df = (
        _view_arr(arr, dtype)
        .filter(pl.col("identifier").is_in([0, 85]))
        .drop("identifier")
        .with_columns(
            [
                _count_changes(pl.col("cycle_count")),
                pl.col("step_time_s").cast(pl.Float64),
                pl.col("voltage_V").cast(pl.Float32) / 10000,
                pl.col("current_mA") * multiplier,
                (
                    pl.col("capacity_mAh").cast(pl.Float64)
                    * multiplier
                    * _step_sign(pl.col("step_type"), pl.col("current_mA"))
                    / 3600
                ).cast(pl.Float32),
                _count_changes(pl.col("step_index")).alias("step_count"),
            ]
        )
    )
    return _add_derived_unix_time(df, start_time_s)


def _read_nda_5(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 5 (file versions 5, 6, 7, 8).

    Identifier is 0 for versions 5, 6, 8, or 85 for 7.
    Raw cycle_count is 0-indexed for 5, 7, 1-indexed for 6, 8.
    """
    header_idx, data_len = _nda_head_main(mm)
    arr = _get_arr_from_nda(mm, header=header_idx, record_len=59, data_len=data_len)
    dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("step_time_s", "<u4"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad2", "V8"),
            ("capacity_mAh", "<i8"),
            ("energy_mWh", "<i8"),
            ("unix_time_s", "<u8"),
            ("_pad3", "V4"),  # Possibly a checksum
        ]
    )
    multiplier = _nda_multiplier(mm)
    nda_version = int(mm[14])
    cycle_offset = 1 if nda_version in (5, 7) else 0
    return (
        _view_arr(arr, dtype)
        .filter(pl.col("identifier").is_in([0, 85]))
        .drop("identifier")
        .with_columns(
            [
                pl.col("cycle_count") + cycle_offset,
                pl.col("step_time_s").cast(pl.Float64),
                pl.col("voltage_V").cast(pl.Float32) / 10000,
                pl.col("current_mA") * multiplier,
                (
                    pl.col(["capacity_mAh", "energy_mWh"]).cast(pl.Float64)
                    * multiplier
                    * _step_sign(pl.col("step_type"), pl.col("current_mA"))
                    / 3600
                ).cast(pl.Float32),
                _count_changes(pl.col("step_index")).alias("step_count"),
            ]
        )
    )


def _read_nda_9(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 9."""
    header_idx, data_len = _nda_head_main(mm)
    arr = _get_arr_from_nda(mm, header=header_idx, record_len=60, data_len=data_len)
    dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("_pad0", "V1"),  # btAuxChlID
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("step_time_s", "<u4"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad1", "V8"),  # nIR, iTemp
            ("capacity_mAh", "<i8"),
            ("energy_mWh", "<i8"),
            ("unix_time_s", "<u8"),
            ("_pad2", "V4"),  # dwCRC32
        ]
    )
    multiplier = _nda_multiplier(mm)
    return _mask_arr(arr, dtype, 85).with_columns(
        [
            pl.col("cycle_count") + 1,
            pl.col("step_time_s").cast(pl.Float64),
            pl.col("voltage_V").cast(pl.Float32) / 10000,
            pl.col("current_mA") * multiplier,
            (
                pl.col(["capacity_mAh", "energy_mWh"]).cast(pl.Float64)
                * multiplier
                * _step_sign(pl.col("step_type"), pl.col("current_mA"))
                / 3600
            ).cast(pl.Float32),
            _count_changes(pl.col("step_index")).alias("step_count"),
        ]
    )


def _read_nda_10(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 10."""
    header_idx, data_len = _nda_head_main(mm)
    arr = _get_arr_from_nda(mm, header=header_idx, record_len=64, data_len=data_len)
    dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("_pad0", "V1"),  # btAuxChlID
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("step_time_s", "<u8"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad1", "V8"),  # nIR, iTemp
            ("capacity_mAh", "<i8"),
            ("energy_mWh", "<i8"),
            ("unix_time_s", "<u8"),
            ("_pad2", "V4"),  # dwCRC32
        ]
    )
    multiplier = _nda_multiplier(mm)
    return _mask_arr(arr, dtype, 85).with_columns(
        [
            _count_changes(pl.col("cycle_count")),
            pl.col("step_time_s").cast(pl.Float64),
            pl.col("voltage_V").cast(pl.Float32) / 10000,
            pl.col("current_mA") * multiplier,
            (
                pl.col(["capacity_mAh", "energy_mWh"]).cast(pl.Float64)
                * multiplier
                * _step_sign(pl.col("step_type"), pl.col("current_mA"))
                / 3600
            ).cast(pl.Float32),
            _count_changes(pl.col("step_index")).alias("step_count"),
        ]
    )


def _read_nda_11(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda struct 11 (file versions 11, 12, 13, 15, 18)."""
    header_idx, data_len = _nda_head_main(mm)
    arr = _get_arr_from_nda(mm, header=header_idx, record_len=69, data_len=data_len)
    dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("_pad1", "V1"),  # aux ID
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u2"),
            ("step_type", "<u1"),
            ("step_time_s", "<u8"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad3", "V8"),  # aux channels
            ("capacity_mAh", "<i8"),
            ("energy_mWh", "<i8"),
            ("unix_time_s", "<u8"),
            ("range", "<i4"),
            ("_pad2", "V4"),  # dwCRC32
        ]
    )
    multiplier = _nda_multiplier(mm, pl.col("range"))
    data_df = (
        _mask_arr(arr, dtype, 85)
        .with_columns(
            [
                pl.col("cycle_count") + 1,
                pl.col("step_time_s").cast(pl.Float64) / 1000,
                pl.col("voltage_V").cast(pl.Float32) / 10000,
                pl.col("current_mA") * multiplier,
                (
                    pl.col("capacity_mAh").cast(pl.Float64)
                    * multiplier
                    * _step_sign(pl.col("step_type"), pl.col("current_mA"))
                    / 3600
                ).cast(pl.Float32),
                (
                    pl.col("energy_mWh").cast(pl.Float64)
                    * multiplier
                    * _step_sign(pl.col("step_type"), pl.col("current_mA"))
                    / 3600
                ).cast(pl.Float32),
                _count_changes(pl.col("step_index")).alias("step_count"),
            ]
        )
        .drop("range")
    )

    aux_begin, aux_len = _nda_head_main(mm, pos_offset=72)
    aux_df = pl.DataFrame(schema={"index": pl.UInt32})
    if aux_len:
        aux_arr = _get_arr_from_nda(mm, header=aux_begin, record_len=69, data_len=aux_len)
        aux_dtype = np.dtype(
            [
                ("identifier", "<u1"),
                ("aux", "<u1"),
                ("index", "<u4"),
                ("_pad1", "V15"),
                ("aux_voltage_V", "<i4"),  # best guess, offset unconfirmed - always 0 in test data
                ("_pad2", "V8"),
                ("aux_temperature_degC", "<i2"),
                ("_pad3", "V34"),
            ]
        )
        aux_df = _mask_arr(aux_arr, aux_dtype, 101).with_columns(
            [
                pl.col("aux_temperature_degC").cast(pl.Float32) / 10,
                pl.col("aux_voltage_V").cast(pl.Float32) / 10000,  # best guess
            ]
        )
        aux_df = _drop_empty(aux_df, ["aux_voltage_V", "aux_temperature_degC"])
    return _merge_aux(data_df, aux_df)


def _read_nda_14(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 14 (file versions 14, 16, 17, 20, 22, 23, 24)."""
    header_idx, data_len = _nda_head_main(mm)
    arr = _get_arr_from_nda(mm, header=header_idx, record_len=86, data_len=data_len)
    data_dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("_pad1", "V1"),
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u2"),
            ("step_type", "<u1"),
            ("step_count", "<u1"),
            ("step_time_s", "<u8"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad3", "V8"),
            ("charge_capacity_mAh", "<i8"),
            ("discharge_capacity_mAh", "<i8"),
            ("charge_energy_mWh", "<i8"),
            ("discharge_energy_mWh", "<i8"),
            ("unix_time_s", "<u8"),
            ("range", "<i4"),
            ("_pad5", "V4"),
        ]
    )
    multiplier = _nda_multiplier(mm, pl.col("range"))
    mult_cols = ["charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh"]
    data_df = (
        _mask_arr(arr, data_dtype, 85)
        .with_columns(
            [
                pl.col("cycle_count") + 1,
                pl.col("step_time_s").cast(pl.Float64) / 1000,
                pl.col("voltage_V").cast(pl.Float32) / 10000,
                _count_changes(pl.col("step_index"), pl.col("step_count")).alias("step_count"),
                pl.col("current_mA") * multiplier,
                (pl.col(mult_cols).cast(pl.Float64) * multiplier.cast(pl.Float64) / 3600).cast(pl.Float32),
            ]
        )
        .drop("range")
    )

    aux_begin, aux_len = _nda_head_main(mm, pos_offset=72)
    aux_df = pl.DataFrame(schema={"index": pl.UInt32})
    if aux_len:
        aux_arr = _get_arr_from_nda(mm, header=aux_begin, record_len=86, data_len=aux_len)
        aux_dtype = np.dtype(
            [
                ("identifier", "<u1"),
                ("aux", "<u1"),
                ("index", "<u4"),
                ("_pad1", "V16"),
                ("aux_voltage_V", "<i4"),
                ("_pad2", "V8"),
                ("aux_temperature_degC", "<i2"),
                ("_pad3", "V50"),
            ]
        )
        aux_df = _mask_arr(aux_arr, aux_dtype, 101).with_columns(
            [
                pl.col("aux_temperature_degC").cast(pl.Float32) / 10,  # 0.1'C -> 'C
                pl.col("aux_voltage_V").cast(pl.Float32) / 10000,  # 0.1 mV -> V
            ]
        )
        aux_df = _drop_empty(aux_df, ["aux_voltage_V", "aux_temperature_degC"])
    return _merge_aux(data_df, aux_df)


def _read_nda_19(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 19."""
    header_idx, _data_len = _nda_head_main(mm)
    arr = _get_arr_from_nda(mm, header=header_idx, record_len=68)
    dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("_pad0", "V3"),  # btSubDevID, btChannelID, btAuxChlID
            ("_pad0b", "V4"),  # dwTestID
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u2"),
            ("step_type", "<u1"),
            ("_pad1", "V1"),  # btWorkType
            ("_pad2", "V1"),  # btStepChgCount
            ("_pad3", "V3"),  # btReserved
            ("step_time_s", "<u4"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad4", "V8"),  # nIR, iTemp
            ("charge_capacity_mAh", "<i4"),
            ("discharge_capacity_mAh", "<i4"),
            ("charge_energy_mWh", "<i4"),
            ("discharge_energy_mWh", "<i4"),
            ("unix_time_s", "<u4"),
            ("_pad5", "V4"),  # dwCRC32
        ]
    )
    multiplier = _nda_multiplier(mm)
    mult_cols = ["charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh"]
    return _mask_arr(arr, dtype, 85).with_columns(
        [
            pl.col("cycle_count") + 1,
            pl.col("step_time_s").cast(pl.Float64),
            pl.col("voltage_V").cast(pl.Float32) / 10000,
            pl.col("current_mA") * multiplier,
            (pl.col(mult_cols).cast(pl.Float64) * multiplier / 3600).cast(pl.Float32),
            _count_changes(pl.col("step_index")).alias("step_count"),
        ]
    )


def _read_nda_25(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 25 (file versions 25, 27)."""
    header_idx, data_len = _nda_head_main(mm)
    arr = _get_arr_from_nda(mm, header=header_idx, record_len=70, data_len=data_len)
    dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("_pad0", "V1"),  # btAuxChlID
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u2"),
            ("step_type", "<u1"),
            ("step_time_s", "<u8"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad1", "V8"),  # nIR, iTemp
            ("capacity_mAh", "<i8"),
            ("energy_mWh", "<i8"),
            ("unix_time_s", "<u8"),
            ("range", "<i4"),
            ("_pad2", "V1"),  # btStepChgCount
            ("_pad3", "V4"),  # dwCRC32
        ]
    )
    multiplier = _nda_multiplier(mm, pl.col("range"))
    return (
        _mask_arr(arr, dtype, 85)
        .with_columns(
            [
                _count_changes(pl.col("cycle_count")),
                pl.col("step_time_s").cast(pl.Float64) / 1000,
                pl.col("voltage_V").cast(pl.Float32) / 10000,
                _count_changes(pl.col("step_index")).alias("step_count"),
                pl.col("current_mA") * multiplier,
                (
                    pl.col("capacity_mAh").cast(pl.Float64)
                    * multiplier
                    * _step_sign(pl.col("step_type"), pl.col("current_mA"))
                    / 3600
                ).cast(pl.Float32),
                (
                    pl.col("energy_mWh").cast(pl.Float64)
                    * multiplier
                    * _step_sign(pl.col("step_type"), pl.col("current_mA"))
                    / 3600
                ).cast(pl.Float32),
            ]
        )
        .drop("range")
    )


def _read_nda_29(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 26, 29."""
    arr = _get_arr_from_nda(mm, b"\x55\x00\x01\x00\x00\x00", 86)
    data_dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("_pad1", "V1"),
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u2"),
            ("step_type", "<u1"),
            ("step_count", "<u1"),  # Records jumps
            ("step_time_s", "<u8"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad3", "V8"),
            ("charge_capacity_mAh", "<i8"),
            ("discharge_capacity_mAh", "<i8"),
            ("charge_energy_mWh", "<i8"),
            ("discharge_energy_mWh", "<i8"),
            ("Y", "<u2"),
            ("M", "<u1"),
            ("D", "<u1"),
            ("h", "<u1"),
            ("m", "<u1"),
            ("s", "<u1"),
            ("_pad4", "V1"),
            ("range", "<i4"),
            ("_pad5", "V4"),
        ]
    )
    multiplier = _nda_multiplier(mm, pl.col("range"))
    mult_cols = ["charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh"]
    data_df = (
        _mask_arr(arr, data_dtype, 85)
        .with_columns(
            [
                pl.col("cycle_count") + 1,
                pl.col("step_time_s").cast(pl.Float64) / 1000,
                pl.col("voltage_V").cast(pl.Float32) / 10000,
                pl.datetime(pl.col("Y"), pl.col("M"), pl.col("D"), pl.col("h"), pl.col("m"), pl.col("s")).alias(
                    "timestamp"
                ),
                _count_changes(pl.col("step_count")).alias("step_count"),
                pl.col("current_mA") * multiplier,
                (pl.col(mult_cols).cast(pl.Float64) * multiplier.cast(pl.Float64) / 3600).cast(pl.Float32),
            ]
        )
        .with_columns((pl.col("timestamp").cast(pl.Float64) * 1e-6).alias("unix_time_s"))
        .drop(["Y", "M", "D", "h", "m", "s", "range"])
    )

    aux_dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("aux", "<u1"),
            ("index", "<u4"),
            ("_pad2", "V16"),
            ("aux_voltage_V", "<i4"),
            ("_pad3", "V8"),
            ("aux_temperature_degC", "<i2"),
            ("_pad4", "V50"),
        ]
    )
    aux_df = _mask_arr(arr, aux_dtype, 101).with_columns(
        [
            pl.col("aux_temperature_degC").cast(pl.Float32) / 10,  # 0.1'C -> 'C
            pl.col("aux_voltage_V").cast(pl.Float32) / 10000,  # 0.1 mV -> V
        ]
    )
    return _merge_aux(data_df, aux_df)


# Identifier byte in record, pos 4 in BTS9.0, pos 0 in BTS9.1
_BTS9_IDENTIFIER = 85


def _bts9_data_blocks(mm: mmap.mmap) -> list[tuple[int, int]]:
    """Read the {begin, length} pointers to every data block of a BTS9 file.

    Most files hold a single block, some files have multiple blocks each with
    its own 1024-byte NEWARE header.
    """
    begin, total_length = _nda_head_main(mm, pos_offset=82, pos64=True)
    if begin != 0 or mm[1024:1030] != b"NEWARE":
        return [(begin, total_length)]
    # A header with no begin summarises all blocks
    blocks: list[tuple[int, int]] = []
    head = 1024
    seen: set[int] = set()
    while head not in seen and mm[head : head + 6] == b"NEWARE":
        seen.add(head)
        block_begin, block_length = _nda_head_main(mm, pos_offset=head + 82, pos64=True)
        if block_length:
            blocks.append((block_begin, block_length))
        # A header's last pointer ends on the next header or EOF
        foot_begin, foot_length = _nda_head_main(mm, pos_offset=head + 242, pos64=True)
        head = foot_begin + foot_length
    chained_length = sum(length for _, length in blocks)
    if chained_length != total_length:
        warnings.warn(
            f"BTS9 section chain covers {chained_length} bytes but the header reports "
            f"{total_length} - some records may be missing.",
            UnverifiedFormatWarning,
            stacklevel=2,
        )
    return blocks


def _bts9_record_len(mm: mmap.mmap, blocks: list[tuple[int, int]]) -> int:
    """Guess the BTS9.0 record length, which is not recorded in the header."""
    begin, first_len = blocks[0]
    block_end = begin + first_len if first_len else len(mm)
    # Records open with a constant 4-byte tag (maybe device type?) followed by an identifier
    tag = mm[begin : begin + 4] + bytes([_BTS9_IDENTIFIER])
    next_record = mm.find(tag, begin + 1, block_end)
    # If no next record, it is one record
    record_len = next_record - begin if next_record != -1 else block_end - begin
    if record_len <= 0:
        msg = f"Could not find a BTS9.0 record at {begin}."
        raise EOFError(msg)
    # Correct length should integer divide every data block
    if any(length % record_len for _, length in blocks):
        msg = f"BTS9.0 record length {record_len} does not divide every data block."
        raise ValueError(msg)
    return record_len


def _df_from_blocks(
    mm: mmap.mmap,
    blocks: list[tuple[int, int]],
    record_len: int,
    dtype: np.dtype,
    mask: int,
) -> pl.DataFrame:
    """Merge data block arrays into one dataframe, concat in polars for speed."""
    dfs = [
        _mask_arr(_get_arr_from_nda(mm, header=begin, record_len=record_len, data_len=length), dtype, mask)
        for begin, length in blocks
    ]
    return pl.concat(dfs, how="vertical", rechunk=False)


def _read_nda_129(mm: mmap.mmap) -> pl.DataFrame:
    """Read the data blocks shared by nda 129 and nda 130 BTS9.0."""
    blocks = _bts9_data_blocks(mm)
    record_len = _bts9_record_len(mm, blocks)
    dtype = np.dtype(
        [
            ("_pad1", "V4"),  # btDevType, btDevID, btUnitID, btChlID
            ("identifier", "<u1"),
            ("_pad2", "V4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("_pad3", "V5"),
            ("index", "<u4"),
            ("_pad4", "V8"),
            ("step_time_s", "<u8"),  # microseconds
            ("voltage_V", "<f4"),
            ("current_mA", "<f4"),
            ("_pad5", "V8"),  # fInterRes, fTempture
            ("charge_capacity_mAh", "<f4"),  # mA.s
            ("charge_energy_mWh", "<f4"),  # mW.s
            ("discharge_capacity_mAh", "<f4"),  # mA.s
            ("discharge_energy_mWh", "<f4"),  # mW.s
            ("unix_time_s", "<u8"),  # microseconds
            ("_pad6", f"V{record_len - 76}"),  # dwCurStepRange, dwLogCode, dwCRC32
        ]
    )
    # Files may or may not put a negative sign on discharge
    # abs those cols and let main calculate net capacity/energy
    mult_cols = ["charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh"]
    return _df_from_blocks(mm, blocks, record_len, dtype, _BTS9_IDENTIFIER).with_columns(
        [
            pl.col(mult_cols).abs() / 3600,
            (pl.col("unix_time_s").cast(pl.Float64) / 1e6).alias("unix_time_s"),
            (pl.col("step_time_s").cast(pl.Float64) / 1e6).alias("step_time_s"),
            _count_changes(pl.col("step_index")).alias("step_count"),
        ]
    )


def _read_nda_130(mm: mmap.mmap) -> pl.DataFrame:
    """Figure out whether BTS9.0 or BTS9.1 and pass to correct function."""
    begin = _bts9_data_blocks(mm)[0][0]
    # Both carry identifier 85, at byte 4 of a BTS9.0 record and byte 0 of a BTS9.1 record
    if mm[begin + 4] == _BTS9_IDENTIFIER:
        return _read_nda_129(mm)
    if mm[begin] == _BTS9_IDENTIFIER:
        return _read_nda_130_91(mm)
    msg = f"nda 130 data block at {begin} does not match BTS9.0 or BTS9.1"
    raise NotImplementedError(msg)


def _read_nda_130_91(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 130 BTS9.1."""
    # Search forward from the first record for the next identifier to get the record length
    blocks = _bts9_data_blocks(mm)
    begin = blocks[0][0]
    identifier_bytes = mm[begin : begin + 2]
    identifier_int = int.from_bytes(identifier_bytes, byteorder="little", signed=False)
    record_len = mm.find(identifier_bytes, begin + 2) - begin

    # In BTS9.1, data and aux are in the same rows
    dtype_list = [
        ("identifier", "<u2"),
        ("step_index", "<u1"),
        ("step_type", "<u1"),
        ("_pad2", "V4"),
        ("index", "<u4"),
        ("total_time_s", "<u4"),
        ("time_ns", "<u4"),
        ("current_mA", "<f4"),
        ("voltage_V", "<f4"),
        ("capacity_mAs", "<f4"),
        ("energy_mWs", "<f4"),
        ("cycle_count", "<u4"),
        ("_pad3", "V4"),  # Data here, looks like <f4 doesn't match anything in ref
        ("unix_time_s", "<u4"),
        ("uts_ns", "<u4"),
    ]
    if record_len >= 56:
        dtype_list += [("aux_temperature_degC", "<f4")]
    if record_len > 56:
        dtype_list.append(("_pad4", f"V{record_len - 56}"))
    data_dtype = np.dtype(dtype_list)

    data_df = _df_from_blocks(mm, blocks, record_len, data_dtype, identifier_int).with_columns(
        [
            pl.col("capacity_mAs").clip(lower_bound=0).alias("charge_capacity_mAh") / 3600,
            pl.col("capacity_mAs").clip(upper_bound=0).abs().alias("discharge_capacity_mAh") / 3600,
            pl.col("energy_mWs").clip(lower_bound=0).alias("charge_energy_mWh") / 3600,
            pl.col("energy_mWs").clip(upper_bound=0).abs().alias("discharge_energy_mWh") / 3600,
            (pl.col("total_time_s") + pl.col("time_ns") / 1e9).cast(pl.Float64),
            (pl.col("unix_time_s") + pl.col("uts_ns") / 1e9).alias("unix_time_s"),
            pl.col("cycle_count") + 1,
            _count_changes(pl.col("step_index")).alias("step_count"),
        ]
    )
    # Need to calculate step times - not included in this NDA
    max_df = (
        data_df.group_by("step_count")
        .agg(pl.col("total_time_s").max().alias("max_total_time_s"))
        .sort("step_count")
        .with_columns(pl.col("max_total_time_s").shift(1).fill_null(0))
    )

    data_df = data_df.join(max_df, on="step_count", how="left").with_columns(
        (pl.col("total_time_s") - pl.col("max_total_time_s")).alias("step_time_s")
    )
    return data_df.drop(["uts_ns", "energy_mWs", "capacity_mAs", "time_ns", "max_total_time_s"])


# NDA FileVer code -> struct type reader
_NDA_READERS: dict[int, Callable[[mmap.mmap], pl.DataFrame]] = {
    1: _read_nda_1,
    2: _read_nda_2,  # Deprecated by Neware
    3: _read_nda_3,
    4: _read_nda_3,
    5: _read_nda_5,
    6: _read_nda_5,
    7: _read_nda_5,
    8: _read_nda_5,
    9: _read_nda_9,
    10: _read_nda_10,
    11: _read_nda_11,
    12: _read_nda_11,
    13: _read_nda_11,
    14: _read_nda_14,
    15: _read_nda_11,
    16: _read_nda_14,
    17: _read_nda_14,
    18: _read_nda_11,
    19: _read_nda_19,
    20: _read_nda_14,
    # 21: Missing in Neware
    22: _read_nda_14,
    23: _read_nda_14,
    24: _read_nda_14,
    25: _read_nda_25,
    26: _read_nda_29,
    27: _read_nda_25,
    28: _read_nda_29,
    29: _read_nda_29,
    129: _read_nda_129,  # Deprecated by Neware
    130: _read_nda_130,  # Variable length
}

# Reader functions confirmed against real data
_CONFIRMED_READER_NAMES = frozenset({"_read_nda_5", "_read_nda_14", "_read_nda_29", "_read_nda_130"})
_CONFIRMED_NDA_VERSIONS = frozenset(
    version for version, reader in _NDA_READERS.items() if reader.__name__ in _CONFIRMED_READER_NAMES
)
