# Copyright © 2026, Empa.
"""Main module for reading Neware NDA and NDAX files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


def read(
    file: str | Path,
    cycle_mode: Literal["chg", "dchg", "auto", "raw"] = "chg",
    columns: Literal["default", "bdf", "bdf-pref"] = "default",
    *,
    raw_categories: bool = False,
) -> pl.DataFrame:
    """Read Neware nda or ndax binary file into polars DataFrame.

    Args:
        file: Path of .nda or .ndax file to read
        cycle_mode: Selects how the cycle is incremented.
            'chg': (Default) Cycle incremented by a charge step following a discharge.
            'dchg': Cycle incremented by a discharge step following a charge.
            'auto': Identifies the first non-rest state as the incremental state.
            'raw': Leaves cycles as it is found in the Neware file.
        columns: Selects how to format the output columns
            'default': fastnda columns, e.g. 'voltage_V', 'current_mA'
            'bdf': battery-data-format columns 'machine-readable' columns
                e.g. 'voltage_volt', 'current_ampere'
                battery-data-format is still in development, these column names
                may change without a major version bump
            'bdf-pref': battery-data-format 'preferred label' columns
                e.g. 'Voltage / V', 'Current / A'
                battery-data-format is still in development, these column names
                may change without a major version bump
        raw_categories: Return `step_type` column as integer codes.

    Returns:
        DataFrame containing all records in the file

    """
    # Read file and generate DataFrame
    file = Path(file)
    if file.suffix == ".nda":
        from fastnda.nda import read_nda

        df = read_nda(file)
    elif file.suffix == ".ndax":
        from fastnda.ndax import read_ndax

        df = read_ndax(file)
    else:
        msg = "File type not supported!"
        raise ValueError(msg)

    # Generate cycle number if requested or missing
    if "cycle_count" not in df.columns and cycle_mode == "raw":
        logger.warning("Raw cycle column missing for this file type, using 'auto'.")
        cycle_mode = "auto"
    if cycle_mode in {"chg", "dchg", "auto"}:
        cycle_mode = cast("Literal['chg', 'dchg', 'auto']", cycle_mode)
        from fastnda.utils import _generate_cycle_number

        df = _generate_cycle_number(df, cycle_mode)
    import polars as pl

    from fastnda.dicts import DTYPE_MAP, STEP_TYPE_MAP

    if "total_time_s" not in df.columns:
        from fastnda.utils import _add_total_time

        df = _add_total_time(df)

    # Round time to us, step_type -> categories, merge charge/discharge capacity/energy
    cols = []
    if "step_time_s" in df.columns:
        cols.append(pl.col("step_time_s").round(6))
    if "total_time_s" in df.columns:
        cols.append(pl.col("total_time_s").round(6))
    if "unix_time_s" in df.columns:
        cols.append(pl.col("unix_time_s").round(6))
    if not raw_categories:
        cols.append(pl.col("step_type").replace_strict(STEP_TYPE_MAP, default=None, return_dtype=pl.Categorical))
    if "capacity_mAh" not in df.columns:
        cols += [
            (pl.col("charge_capacity_mAh") - pl.col("discharge_capacity_mAh")).alias("capacity_mAh"),
            (pl.col("charge_energy_mWh") - pl.col("discharge_energy_mWh")).alias("energy_mWh"),
        ]
    df = df.with_columns(cols)

    # Ensure columns have correct data types
    dtype_map = dict(DTYPE_MAP)
    if raw_categories:
        dtype_map["step_type"] = pl.UInt8
    df = df.with_columns([pl.col(name).cast(dtype_map[name]) for name in df.columns if name in dtype_map])

    # Reorder columns
    non_aux_columns = [name for name in DTYPE_MAP if name in df.columns]
    aux_columns = [name for name in df.columns if name.startswith("aux")]
    df = df.select(non_aux_columns + aux_columns)

    # Output with desired column-style
    if columns == "default":
        return df
    if columns == "bdf":
        from fastnda.formats import to_bdf

        return to_bdf(df)
    if columns == "bdf-pref":
        from fastnda.formats import to_bdf_pref

        return to_bdf_pref(df)
    logger.warning("Column type %s not understood, using default.", columns)
    return df


def read_metadata(file: str | Path) -> dict[str, str | float]:
    """Read metadata from a Neware .nda or .ndax file.

    Args:
        file: Path of .nda or .ndax file

    Returns:
        Dictionary containing metadata

    """
    file = Path(file)
    suffix = file.suffix.lower()
    if suffix == ".nda":
        from fastnda.nda_meta import read_nda_metadata

        return read_nda_metadata(file)
    if suffix == ".ndax":
        from fastnda.ndax import read_ndax_metadata

        return read_ndax_metadata(file)
    msg = "File type not supported!"
    raise ValueError(msg)
