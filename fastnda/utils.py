# Copyright © 2026, Empa.
"""Utility functions for processing Neware data."""

import logging
from typing import Literal

import polars as pl

from fastnda.dicts import CHARGE_DISCHARGE_MAP

logger = logging.getLogger(__name__)


def _generate_cycle_number(
    df: pl.DataFrame,
    cycle_mode: Literal["chg", "dchg", "auto"] = "chg",
) -> pl.DataFrame:
    """Generate a cycle number to match Neware.

    cycle_mode: Selects how the cycle is incremented.
            'chg': (Default) Cycle incremented by a charge step following a discharge.
            'dchg': Cycle incremented by a discharge step following a charge.
            'auto': Identifies the first non-rest state as the incremental state.
    """
    if cycle_mode not in {"chg", "dchg", "auto"}:
        msg = "Cycle_Mode %s not recognized. Supported options are 'chg', 'dchg', and 'auto'."
        raise KeyError(msg, cycle_mode)

    # Check if any df
    if len(df) == 0:
        return df

    if df.select(pl.col("step_type").is_in({16, 17, 25}).any()).item():
        logger.warning(
            "Data contains Pulse, SIM, or Ramp steps. "
            "This might give unexpected cycle numbers with 'chg' 'dchg' or 'auto' mode. "
            "Consider using 'raw' cycle mode instead."
        )

    # Auto: find the first non rest cycle
    if cycle_mode == "auto":
        cycle_mode = _id_first_state(df)

    # Increment when 0->1 (chg) or 1->0 (dchg)
    target_diff = 1 if cycle_mode == "chg" else -1
    return df.with_columns(
        pl.col("step_type")
        .replace_strict(CHARGE_DISCHARGE_MAP, default=None)
        .forward_fill()
        .diff()
        .eq(target_diff)
        .cum_sum()
        .fill_null(0)
        .add(1)
        .cast(pl.UInt32)
        .alias("cycle_count")
    )


def _step_sign(step_type: pl.Expr, current_col: pl.Expr) -> pl.Expr:
    """Sign of capacity/energy from step type.

    Falls back to the current sign for step types not in CHARGE_DISCHARGE_MAP.
    """
    return (step_type.replace_strict(CHARGE_DISCHARGE_MAP, default=None) * 2 - 1).fill_null(current_col.sign())


def _count_changes(series: pl.Expr) -> pl.Expr:
    """Enumerate the number of value changes in a series."""
    return series.diff().fill_null(1).abs().gt(0).cum_sum()


def _id_first_state(df: pl.DataFrame) -> Literal["chg", "dchg"]:
    """Identify the first non-rest state in the DataFrame."""
    # Filter on non-rest keys, check first row
    filtered = df.filter(pl.col("step_type").is_in(CHARGE_DISCHARGE_MAP)).head(1)
    if not filtered.is_empty() and CHARGE_DISCHARGE_MAP[filtered[0, "step_type"]] == 1:
        return "chg"
    return "dchg"


def _range_to_mult(series: pl.Expr) -> pl.Expr:
    """Generate multiplier from current range."""
    a = series.abs()
    return (
        pl.when((series == 0) | (series < -2e9)).then(1e-1)
        .when(series < 0).then(
            pl.when(a < 10).then(1e-5)
            .when(a < 100).then(1e-4)
            .when(a < 1000).then(1e-3)
            .when(a < 1000000).then(1e-2)
            .when(a < 10000000).then(1e-8)
            .when(a < 100000000).then(1e-7)
            .otherwise(1e-6)
        )
        .otherwise(
            pl.when(a < 10).then(1e-4)
            .when(a < 100).then(1e-3)
            .when(a < 1000).then(1e-2)
            .otherwise(1e-1)
        )
    )  # fmt: skip


def _drop_empty(df: pl.DataFrame, cols: list[str] | None) -> pl.DataFrame:
    """Drop empty columns."""
    # Drop empty columns
    if cols is None:
        cols = df.columns
    cols_to_drop = [c for c in cols if df.filter(pl.col(c) != 0).is_empty()]
    return df.select(pl.exclude(cols_to_drop))


class UnverifiedFormatWarning(UserWarning):
    """Raised when reading an nda_version which hasn't been tested against real data."""
