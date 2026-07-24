"""Module to read Neware main NDC files."""

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
