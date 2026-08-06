# Copyright © 2026, Empa.
"""Private module to read Neware main NDC files.

Do not use these methods directly, they may change any time without warning.
"""

import warnings

import numpy as np
import polars as pl

from fastnda._ndc.ndc_utils import bytes_to_df
from fastnda.utils import _count_changes, _range_to_mult


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
                _range_to_mult(pl.col("range")).alias("multiplier"),
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
                _range_to_mult(pl.col("range")).alias("multiplier"),
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
                _range_to_mult(pl.col("range")).alias("multiplier"),
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


# ndax 15 - fixed 36-byte record prefix + customisable columns after
# Fixed columns
_NDC15_PREFIX_FIELDS: list[tuple[str, str, int]] = [
    ("step_index", "<u1", 2),
    ("step_type_raw", "<u1", 3),  # different to STEP_TYPE_MAP, needs remapping
    ("index", "<u4", 8),
    ("total_time_s", "<u4", 12),
    ("total_time_ns", "<u4", 16),
    ("current_mA", "<f4", 20),
    ("voltage_V", "<f4", 24),
    ("fixed_cap_mAs", "<f4", 28),  # ignored if dynamic ccap/dcap are present
    ("fixed_eng_mWs", "<f4", 32),  # ignored if dynamic ccap/dcap are present
]

# Customisable columns
_NDC15_CUSTOM_ITEM_DTYPE = np.dtype([("axis_type", "<i4"), ("value_type", "u1"), ("pos", "<i4")])
# axis_type -> sub-fields
_NDC15_DYNAMIC_FIELDS: dict[int, list[tuple[str, str]]] = {
    15: [("unix_time_s", "<u4"), ("unix_time_ns", "<u4")],
    65: [("step_time_s", "<u4"), ("step_time_ns", "<u4")],
    5: [("dyn_ccap_mAs", "<f4")],
    6: [("dyn_dcap_mAs", "<f4")],
    8: [("dyn_ceng_mWs", "<f4")],
    9: [("dyn_deng_mWs", "<f4")],
}

# BTS9 step types to BTS8/fastnda step types defined in dicts.py
_NDC15_STEP_TYPE_MAP: dict[int, int] = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    16: 16,
    17: 17,
    18: 18,
    19: 19,
    20: 20,
    21: 27,
    22: 26,
}


def _ndc15_custom_axis_items(buf: bytes) -> dict[int, int]:
    """Parse the custom column definition into {col_type: byte_position}."""
    items = np.frombuffer(buf, dtype=_NDC15_CUSTOM_ITEM_DTYPE, count=300, offset=520)
    # terminated at the first all-zero entry, or unterminated if all 300 slots are populated
    terminators = np.flatnonzero((items["axis_type"] == 0) & (items["value_type"] == 0))
    end = terminators[0] if terminators.size else len(items)
    return dict(zip(items["axis_type"][:end].tolist(), items["pos"][:end].tolist(), strict=True))


def _ndc15_schema(buf: bytes) -> np.dtype:
    """Build a numpy dtype for main: fixed + custom columns."""
    # Total size in bytes of one row
    itemsize = int(np.frombuffer(buf, dtype="<u4", count=1, offset=516)[0])
    # Get custom columns
    axis_items = _ndc15_custom_axis_items(buf)
    # Copy fixed columns and extend with custom column details
    fields = _NDC15_PREFIX_FIELDS.copy()
    for axis_type, start_pos in axis_items.items():
        dynamic_fields = _NDC15_DYNAMIC_FIELDS.get(axis_type)
        if dynamic_fields is None:
            warnings.warn(f"Axis type {axis_type} in NDC15 schema not understood.", stacklevel=2)
            continue
        pos = start_pos
        for name, fmt in dynamic_fields:
            fields.append((name, fmt, pos))
            pos += np.dtype(fmt).itemsize
    names, formats, offsets = zip(*fields, strict=True)
    return np.dtype({"names": list(names), "formats": list(formats), "offsets": list(offsets), "itemsize": itemsize})


def read_ndc_main_15(buf: bytes) -> pl.DataFrame:
    """Read ndc version 15 filetype 1. See _ndc15_schema for the per-file record layout."""
    dtype = _ndc15_schema(buf)
    df = bytes_to_df(buf, dtype, add_index="index" not in dtype.names)

    # Build exprs for with_columns, map step_type, calc total_time and step_count
    exprs = [
        pl.col("step_type_raw")
        .replace_strict(_NDC15_STEP_TYPE_MAP, default=0, return_dtype=pl.UInt8)
        .alias("step_type"),
        (pl.col("total_time_s") + pl.col("total_time_ns") * 1e-9).alias("total_time_s"),
        _count_changes(pl.col("step_index")).alias("step_count"),
    ]

    # If extra time columns are present, calculate
    if "step_time_s" in df.columns:
        exprs.append((pl.col("step_time_s") + pl.col("step_time_ns") * 1e-9).alias("step_time_s"))
    if "unix_time_s" in df.columns:
        exprs.append((pl.col("unix_time_s") + pl.col("unix_time_ns") * 1e-9).alias("unix_time_s"))

    # If capacity and energy are split into two cols, combine
    if "dyn_ccap_mAs" in df.columns and "dyn_dcap_mAs" in df.columns:
        exprs.append((pl.col("dyn_ccap_mAs") + pl.col("dyn_dcap_mAs")).alias("fixed_cap_mAs"))
    if "dyn_ceng_mWs" in df.columns and "dyn_deng_mWs" in df.columns:
        exprs.append((pl.col("dyn_ceng_mWs") + pl.col("dyn_deng_mWs")).alias("fixed_eng_mWs"))

    # Run expressions, we need rescale "fixed_x_y" in the next step
    df = df.with_columns(exprs)
    exprs = []

    # Raw step type scaling
    _NDC15_POS = {1, 3, 7, 9, 11, 16, 17, 18, 21}
    _NDC15_NEG = {2, 8, 10, 19, 20, 22}
    _NDC15_SIM = 17

    # Don't abs on SIM
    is_absed = pl.col("step_type_raw").ne(_NDC15_SIM)
    # Multiply by 1/3600 for charge/SIM or -1/3600 for discharge
    mult = (
        pl.when(pl.col("step_type_raw").is_in(_NDC15_POS))
        .then(1 / 3600)
        .otherwise(pl.when(pl.col("step_type_raw").is_in(_NDC15_NEG)).then(-1 / 3600).otherwise(0.0))
    )

    exprs.append(
        (pl.when(is_absed).then(pl.col("fixed_cap_mAs").abs()).otherwise(pl.col("fixed_cap_mAs")) * mult).alias(
            "capacity_mAh"
        )
    )
    exprs.append(
        (pl.when(is_absed).then(pl.col("fixed_eng_mWs").abs()).otherwise(pl.col("fixed_eng_mWs")) * mult).alias(
            "energy_mWh"
        )
    )

    df = df.with_columns(exprs)

    drop_cols = [
        "step_type_raw",
        "fixed_cap_mAs",
        "fixed_eng_mWs",
        "dyn_ccap_mAs",
        "dyn_dcap_mAs",
        "dyn_ceng_mWs",
        "dyn_deng_mWs",
        "unix_time_ns",
        "total_time_ns",
        "step_time_ns",
    ]
    return df.drop([c for c in drop_cols if c in df.columns])
