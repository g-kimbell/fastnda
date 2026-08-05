# Copyright © 2026, Empa.
"""Private module to read Neware main NDC files.

Do not use these methods directly, they may change any time without warning.
"""

import numpy as np
import polars as pl

from fastnda._ndc.ndc_utils import bytes_to_df
from fastnda.dicts import MULTIPLIER_MAP
from fastnda.utils import _count_changes


def read_ndc_main_1(buf: bytes) -> pl.DataFrame:
    """Read ndc version 1 filetype 1. Also used for version 3."""
    dtype = np.dtype(
        [
            ("_pad1", "V8"),
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("_pad2", "V2"),
            ("step_time_s", "<u8"),
            ("voltage_V", "<u4"),
            ("current_mA", "<u4"),
            ("_pad3", "V4"),
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
            ("range", "<i4"),
        ]
    )
    return (
        bytes_to_df(buf, dtype, data_start_ind=5, record_size=512, use_bitmask=False)
        .with_columns(
            [
                pl.col("cycle_count") + 1,
                pl.col("step_time_s").cast(pl.Float64) * 1e-3,
                pl.col("voltage_V").cast(pl.Float32) * 1e-4,
                pl.col("range").replace_strict(MULTIPLIER_MAP, return_dtype=pl.Float64).alias("multiplier"),
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
                (pl.col("timestamp").cast(pl.Float64) / 1e6).alias("unix_time_s"),
            ]
        )
        .drop(["Y", "M", "D", "h", "m", "s"])
    )


def read_ndc_main_2(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("_pad1", "V8"),
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("_pad2", "V5"),
            ("step_time_s", "<u8"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad3", "V4"),
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
            ("range", "<i4"),
            ("_pad4", "V8"),
        ]
    )
    return (
        bytes_to_df(buf, dtype, data_start_ind=5, record_size=512, use_bitmask=False)
        .with_columns(
            [
                pl.col("cycle_count") + 1,
                pl.col("step_time_s").cast(pl.Float64) * 1e-3,
                pl.col("voltage_V").cast(pl.Float32) * 1e-4,
                pl.col("range").replace_strict(MULTIPLIER_MAP, return_dtype=pl.Float64).alias("multiplier"),
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
                (pl.col("timestamp").cast(pl.Float64) / 1e6).alias("unix_time_s"),
            ]
        )
        .drop(["Y", "M", "D", "h", "m", "s"])
    )


def read_ndc_main_5(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("_pad1", "V1"),
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("_pad2", "V5"),
            ("step_time_s", "<u8"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad3", "V4"),
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
            ("range", "<i4"),
            ("_pad4", "V8"),
        ]
    )
    return (
        bytes_to_df(buf, dtype)
        .with_columns(
            [
                pl.col("cycle_count") + 1,
                pl.col("step_time_s").cast(pl.Float64) * 1e-3,
                pl.col("voltage_V").cast(pl.Float32) * 1e-4,
                pl.col("range").replace_strict(MULTIPLIER_MAP, return_dtype=pl.Float64).alias("multiplier"),
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
                (pl.col("timestamp").cast(pl.Float64) / 1e6).alias("unix_time_s"),
            ]
        )
        .drop(["Y", "M", "D", "h", "m", "s"])
    )


def read_ndc_main_6(buf: bytes) -> pl.DataFrame:
    """Read ndc version 6 filetype 1.

    v6's main record contains the capacity/energy/step fields.
    Later versions split out into a separate runinfo file (filetype 18).
    """
    dtype = np.dtype(
        [
            ("step_time_s", "<u4"),
            ("_pad1", "V1"),
            ("voltage_V", "<f4"),
            ("current_mA", "<f4"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V1"),
            ("unix_time_s", "<u4"),
            ("step_count", "<u4"),
        ]
    )
    return bytes_to_df(buf, dtype, add_index=True).with_columns(
        [
            pl.col("step_time_s").cast(pl.Float64) / 1000,  # ms -> s
            pl.col("current_mA") * 1000,  # A -> mA
            pl.col("charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh")
            * 1000,  # Ah|Wh -> mAh|mWh
        ]
    )


# ndc7 column code: (column name, numpy typestr)
_NDC7_FIELDS: dict[int, tuple[str, str]] = {
    1: ("_pad1", "V1"),  # DataFlag
    6: ("index", "<u4"),  # TestSn
    7: ("cycle_count", "<u4"),  # Cycle
    8: ("step_index", "<u1"),  # StepID
    9: ("step_type", "<u1"),  # StepType
    10: ("_pad10", "V1"),  # StepChgCount
    11: ("_pad11", "V4"),  # WorkType
    12: ("step_time_s", "<u8"),  # TestTime
    13: ("voltage_V", "<i4"),  # Volt
    14: ("current_mA", "<i4"),  # Curr
    15: ("_pad15", "V2"),  # InterRes
    16: ("temperature_degC", "<i2"),  # Tempature
    17: ("charge_capacity_mAh", "<i8"),  # CCap
    18: ("discharge_capacity_mAh", "<i8"),  # DCCap
    19: ("charge_energy_mWh", "<i8"),  # CEng
    20: ("discharge_energy_mWh", "<i8"),  # DCEng
    28: ("_pad28", "V4"),  # CurStepRange
    29: ("total_time_s", "<u8"),  # TotalTime
    51: ("_pad51", "V8"),  # TotalCap
    52: ("_pad52", "V8"),  # TotalEng
    53: ("atime_ms", "<u2"),  # ATimeMs
}

# ATime (code 21) is several fields packed together (y/m/d/H/M/S), needs a special case
_NDC7_ATIME_CODE = 21
_NDC7_ATIME_FIELDS: list[tuple[str, str]] = [
    ("atime_year", "<u2"),
    ("atime_month", "<u1"),
    ("atime_day", "<u1"),
    ("atime_hour", "<u1"),
    ("atime_minute", "<u1"),
    ("atime_second", "<u1"),
]

_NDC7_SCALE_EXPRS: dict[str, pl.Expr] = {
    "step_time_s": pl.col("step_time_s").cast(pl.Float64) / 1000,
    "total_time_s": pl.col("total_time_s").cast(pl.Float64) / 1000,
    "voltage_V": pl.col("voltage_V").cast(pl.Float32) / 10000,
    "temperature_degC": pl.col("temperature_degC").cast(pl.Float32) * 0.1,
    "charge_capacity_mAh": (pl.col("charge_capacity_mAh").cast(pl.Float64) / 3600).cast(pl.Float32),
    "discharge_capacity_mAh": (pl.col("discharge_capacity_mAh").cast(pl.Float64) / 3600).cast(pl.Float32),
    "charge_energy_mWh": (pl.col("charge_energy_mWh").cast(pl.Float64) / 3600).cast(pl.Float32),
    "discharge_energy_mWh": (pl.col("discharge_energy_mWh").cast(pl.Float64) / 3600).cast(pl.Float32),
}


def _ndc7_schema(buf: bytes) -> np.dtype:
    """Build ND7 dtype from the column code list at the top of the file."""
    n_data_type = np.frombuffer(buf, dtype="<i4", count=40, offset=13)
    fields: list[tuple[str, str]] = []
    for code in n_data_type.tolist():
        if code == 0:
            break
        if code == _NDC7_ATIME_CODE:
            fields.extend(_NDC7_ATIME_FIELDS)
        elif code in _NDC7_FIELDS:
            fields.append(_NDC7_FIELDS[code])
        else:
            # Unknown field means all following offsets are wrong, cannot cleanly skip
            msg = f"ndc version 7: unrecognized field type code {code}"
            raise NotImplementedError(msg)
    return np.dtype(fields)


def read_ndc_main_7(buf: bytes) -> pl.DataFrame:
    """Read ndc version 7 filetype 1 (also used for filetype 5, aux)."""
    dtype = _ndc7_schema(buf)
    df = bytes_to_df(buf, dtype, add_index="index" not in dtype.names)
    exprs = [expr for name, expr in _NDC7_SCALE_EXPRS.items() if name in df.columns]
    if "cycle_count" in df.columns:
        exprs.append(pl.col("cycle_count") + 1)
    if exprs:
        df = df.with_columns(exprs)
    if "step_index" in df.columns:
        df = df.with_columns(_count_changes(pl.col("step_index")).alias("step_count"))
    if "atime_year" in df.columns:
        unix_time_s = (
            pl.datetime(
                pl.col("atime_year"),
                pl.col("atime_month"),
                pl.col("atime_day"),
                pl.col("atime_hour"),
                pl.col("atime_minute"),
                pl.col("atime_second"),
            ).cast(pl.Float64)
            / 1e6
        )
        drop_cols = ["atime_year", "atime_month", "atime_day", "atime_hour", "atime_minute", "atime_second"]
        if "atime_ms" in df.columns:
            unix_time_s = unix_time_s + pl.col("atime_ms") / 1000
            drop_cols.append("atime_ms")
        df = df.with_columns(unix_time_s.alias("unix_time_s")).drop(drop_cols)
    return df


def read_ndc_main_11(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("voltage_V", "<f4"),
            ("current_mA", "<f4"),
        ]
    )
    return bytes_to_df(buf, dtype, add_index=True).with_columns(
        [
            pl.col("voltage_V") * 1e-4,  # 0.1mV -> V
        ]
    )


def read_ndc_main_14(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("voltage_V", "<f4"),
            ("current_mA", "<f4"),
        ]
    )
    return bytes_to_df(buf, dtype, add_index=True).with_columns(
        [
            pl.col("current_mA") * 1000,
        ]
    )


def read_ndc_main_16(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("voltage_V", "<f4"),
            ("current_mA", "<f4"),
        ]
    )
    return bytes_to_df(buf, dtype, add_index=True).with_columns(
        [
            pl.col("voltage_V") / 10000,
            pl.col("current_mA"),
        ]
    )
