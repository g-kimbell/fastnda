# Copyright © 2026, Empa.
"""Convert outputs to different formats."""

from collections.abc import Mapping
from types import MappingProxyType

import polars as pl

BDF_COL_MAP: Mapping[str, str] = MappingProxyType(
    {
        "index": "record_index",
        "voltage_V": "voltage_volt",
        "current_mA": "current_ampere",
        "unix_time_s": "unix_time_second",
        "step_time_s": "step_time_second",
        "total_time_s": "test_time_second",
        "cycle_count": "cycle_count",
        "step_count": "step_count",
        "step_index": "step_id",
        "step_type": "step_type",
        "capacity_mAh": "step_net_capacity_ah",
        "energy_mWh": "step_net_energy_wh",
    }
)

BDF_MULTIPLIER_MAP: Mapping[str, float] = MappingProxyType(
    {
        "current_ampere": 1e-3,
        "step_net_capacity_ah": 1e-3,
        "step_net_energy_wh": 1e-3,
    }
)

BDF_PREF_LABEL_MAP: Mapping[str, str] = MappingProxyType(
    {
        "record_index": "Record Index / 1",
        "voltage_volt": "Voltage / V",
        "current_ampere": "Current / A",
        "unix_time_second": "Unix Time / s",
        "step_time_second": "Step Time / s",
        "test_time_second": "Test Time / s",
        "cycle_count": "Cycle Count / 1",
        "step_count": "Step Count / 1",
        "step_id": "Step ID",
        "step_type": "Step Type",
        "step_net_capacity_ah": "Step Net Capacity / Ah",
        "step_net_energy_wh": "Step Net Energy / Wh",
    }
)


def to_bdf(df: pl.DataFrame) -> pl.DataFrame:
    """Convert fastnda columnds to BDF machine readable columns."""
    col_map = {k: v for k, v in BDF_COL_MAP.items() if k in df.columns}
    df = df.rename(col_map)

    # Dynamically rename any aux columns
    rename_map = {}
    for col in df.columns:
        if col.endswith("_V"):
            rename_map[col] = col[:-2] + "_volt"
        elif col.endswith("_degC"):
            rename_map[col] = col[:-5] + "_celsius"
    if rename_map:
        df = df.rename(rename_map)

    return df.with_columns(pl.col(k) * v for k, v in BDF_MULTIPLIER_MAP.items() if k in df.columns)


def to_bdf_pref(df: pl.DataFrame) -> pl.DataFrame:
    """Convert fastnda columnds to BDF preferred label columns."""
    df = to_bdf(df)
    return df.rename({k: v for k, v in BDF_PREF_LABEL_MAP.items() if k in df.columns})
