"""Module to read Neware 'runInfo' NDC files."""

import numpy as np
import polars as pl

from fastnda._ndc.ndc_utils import bytes_to_df
from fastnda.utils import _count_changes


def read_ndc_runinfo_1(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("step_time_s", "<u4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V5"),
            ("dt", "<i4"),
            ("unix_time_s", "<u4"),
            ("step_count", "<u4"),
            ("index", "<u4"),
        ]
    )
    return (
        bytes_to_df(buf, dtype)
        .with_columns(
            [
                pl.col("step_time_s", "dt").cast(pl.Float64) / 1000,  # ms -> s
                pl.col("charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh")
                * 1000,  # Ah|Wh -> mAh|mWh
                _count_changes(pl.col("step_count")).alias("step_count"),
            ]
        )
        .unique(subset="index", keep="first")
    )


def read_ndc_runinfo_2(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("step_time_s", "<u4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V8"),
            ("dt", "<i4"),
            ("unix_time_s", "<u4"),
            ("step_count", "<u4"),
            ("index", "<u4"),
        ]
    )
    return (
        bytes_to_df(buf, dtype)
        .with_columns(
            [
                pl.col("step_time_s", "dt").cast(pl.Float64) / 1000,  # ms -> s
                pl.col("charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh")
                / 3600,  # mAs|mWs -> mAh|mWh
                _count_changes(pl.col("step_count")).alias("step_count"),
            ]
        )
        .unique(subset="index", keep="first")
    )


def read_ndc_runinfo_11(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("step_time_s", "<u4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V8"),
            ("dt", "<i4"),
            ("unix_time_s", "<u4"),
            ("step_count", "<u4"),
            ("index", "<u4"),
            ("uts_ms", "<u2"),
        ]
    )
    return (
        bytes_to_df(buf, dtype)
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
        .unique(subset="index", keep="first")
    )


def read_ndc_runinfo_12(buf: bytes) -> pl.DataFrame:
    """v12 uses the same struct (NdcRunInfo) as v11, but v12 is "Plus"-family so needs v14's scaling."""
    dtype = np.dtype(
        [
            ("step_time_s", "<u4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V8"),
            ("dt", "<i4"),
            ("unix_time_s", "<u4"),
            ("step_count", "<u4"),
            ("index", "<u4"),
            ("uts_ms", "<u2"),
        ]
    )
    return (
        bytes_to_df(buf, dtype)
        .with_columns(
            [
                pl.col("step_time_s", "dt").cast(pl.Float64) / 1000,  # ms -> s
                pl.col("charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh")
                * 1000,  # Ah|Wh -> mAh|mWh
                (pl.col("unix_time_s") + pl.col("uts_ms") / 1000).alias("unix_time_s"),
                _count_changes(pl.col("step_count")).alias("step_count"),
            ]
        )
        .drop("uts_ms")
        .unique(subset="index", keep="first")
    )


def read_ndc_runinfo_13(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("step_time_s", "<u4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V8"),
            ("dt", "<i4"),
            ("unix_time_s", "<u4"),
            ("step_count", "<u4"),
            ("index", "<u4"),
            ("uts_ms", "<u2"),
            ("_pad3", "V8"),
        ]
    )
    return (
        bytes_to_df(buf, dtype)
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
        .unique(subset="index", keep="first")
    )


def read_ndc_runinfo_14(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("step_time_s", "<u4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V8"),
            ("dt", "<i4"),
            ("unix_time_s", "<u4"),
            ("step_count", "<u4"),
            ("index", "<u4"),
            ("uts_ms", "<i2"),
            ("_pad3", "V8"),
        ]
    )
    return (
        bytes_to_df(buf, dtype)
        .with_columns(
            [
                pl.col("step_time_s", "dt").cast(pl.Float64) / 1000,  # ms -> s
                pl.col("charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh")
                * 1000,  # Ah|Wh -> mAh|mWh
                (pl.col("unix_time_s") + pl.col("uts_ms") / 1000).alias("unix_time_s"),
                _count_changes(pl.col("step_count")).alias("step_count"),
            ]
        )
        .drop("uts_ms")
        .unique(subset="index", keep="first")
    )


def read_ndc_runinfo_16(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("step_time_s", "<u4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V8"),
            ("dt", "<i4"),
            ("unix_time_s", "<u4"),
            ("step_count", "<u4"),
            ("index", "<u4"),
            ("uts_ms", "<u2"),
            ("_pad3", "V53"),
        ]
    )
    return (
        bytes_to_df(buf, dtype)
        .with_columns(
            [
                pl.col("step_time_s", "dt").cast(pl.Float64) / 1000,
                (
                    pl.col("charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh")
                    / 3600
                ).cast(pl.Float32),  # mAs|mWs -> mAh|mWh
                (pl.col("unix_time_s") + pl.col("uts_ms") / 1000).alias("unix_time_s"),
                _count_changes(pl.col("step_count")).alias("step_count"),
            ]
        )
        .drop("uts_ms")
        .unique(subset="index", keep="first")
    )


def read_ndc_runinfo_17(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("step_time_s", "<u4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V8"),
            ("dt", "<i4"),
            ("unix_time_s", "<u4"),
            ("step_count", "<u4"),
            ("index", "<u4"),
            ("uts_ms", "<u2"),
            ("_pad3", "V53"),
        ]
    )
    return (
        bytes_to_df(buf, dtype)
        .with_columns(
            [
                pl.col("step_time_s", "dt").cast(pl.Float64) / 1000,
                (
                    pl.col("charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh")
                    * 1000
                ).cast(pl.Float32),  # Ah|Wh -> mAh|mWh
                (pl.col("unix_time_s") + pl.col("uts_ms") / 1000).alias("unix_time_s"),
                _count_changes(pl.col("step_count")).alias("step_count"),
            ]
        )
        .drop("uts_ms")
        .unique(subset="index", keep="first")
    )
