"""Main module for reading Neware NDA and NDAX files."""

import logging
from pathlib import Path
from typing import Literal, cast

import polars as pl

from fastnda.dicts import dtype_dict
from fastnda.nda import read_nda, read_nda_metadata
from fastnda.ndax import read_ndax, read_ndax_metadata
from fastnda.utils import _generate_cycle_number, state_dict

logger = logging.getLogger(__name__)


def read(file: str | Path, cycle_mode: Literal["chg", "dchg", "auto", "raw"] = "chg") -> pl.DataFrame:
    """Read electrochemical data from an Neware nda or ndax binary file.

    Args:
        file: Path of .nda or .ndax file to read
        cycle_mode: Selects how the cycle is incremented.
            'chg': (Default) Cycle incremented by a charge step following a discharge.
            'dchg': Cycle incremented by a discharge step following a charge.
            'auto': Identifies the first non-rest state as the incremental state.
            'raw': Leaves cycles as it is found in the Neware file.

    Returns:
        DataFrame containing all records in the file

    """
    # Read file and generate DataFrame
    file = Path(file)
    if file.suffix == ".nda":
        df = read_nda(file)
    elif file.suffix == ".ndax":
        df = read_ndax(file)
    else:
        msg = "File type not supported!"
        raise ValueError(msg)

    # Generate cycle number if requested
    if cycle_mode in {"chg", "dchg", "auto"}:
        cycle_mode = cast("Literal['chg', 'dchg', 'auto']", cycle_mode)
        df = _generate_cycle_number(df, cycle_mode)

    # round time to ms, step_type -> categories, uts -> Timestamp
    cols = [
        pl.col("step_time_s").round(3),
        pl.col("step_type").replace_strict(state_dict, default=None).alias("step_type"),
        (pl.col("charge_capacity_mAh") - pl.col("discharge_capacity_mAh")).alias("capacity_mAh"),
        (pl.col("charge_energy_mWh") - pl.col("discharge_energy_mWh")).alias("energy_mWh"),
    ]
    df = df.with_columns(cols)

    if "total_time_s" not in df.columns:
        max_df = (
            df.group_by("step_count")
            .agg(pl.col("step_time_s").max().alias("max_step_time_s"))
            .sort("step_count")
            .with_columns(pl.col("max_step_time_s").shift(1).fill_null(0).cum_sum())
        )
        df = df.join(max_df, on="step_count", how="left").with_columns(
            (pl.col("step_time_s") + pl.col("max_step_time_s")).alias("total_time_s")
        )

    # Ensure columns have correct data types
    df = df.with_columns([pl.col(name).cast(dtype_dict[name]) for name in df.columns if name in dtype_dict])

    # Reorder columns
    non_aux_columns = [name for name in dtype_dict if name in df.columns]
    aux_columns = [name for name in df.columns if name.startswith("aux")]
    df = df.select(non_aux_columns + aux_columns)

    return df


def read_metadata(file: str | Path) -> dict[str, str | float]:
    """Read metadata from a Neware .nda or .ndax file.

    Args:
        file: Path of .nda or .ndax file

    Returns:
        Dictionary containing metadata

    """
    file = Path(file)
    if file.suffix == ".nda":
        return read_nda_metadata(file)
    if file.suffix == ".ndax":
        return read_ndax_metadata(file)
    msg = "File type not supported!"
    raise ValueError(msg)
