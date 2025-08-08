"""Main module for reading Neware NDA and NDAX files."""

import logging
from pathlib import Path
from typing import Literal

import polars as pl

from fastnda.dicts import dtype_dict
from fastnda.nda import read_nda
from fastnda.ndax import read_ndax
from fastnda.utils import _generate_cycle_number, state_dict

logger = logging.getLogger(__name__)


def read(
    file: str | Path, software_cycle_number: bool = True, cycle_mode: Literal["chg", "dchg", "auto"] = "chg"
) -> tuple[pl.DataFrame, dict[str, str | float]]:
    """Read electrochemical data from an Neware nda or ndax binary file.

    Args:
        file: Path of .nda or .ndax file to read
        software_cycle_number: Regenerate the cycle number to match
            Neware's 'Charge First' circular statistic setting
        cycle_mode: Selects how the cycle is incremented.
            'chg': (Default) Sets new cycles with a Charge step following a Discharge.
            'dchg': Sets new cycles with a Discharge step following a Charge.
            'auto': Identifies the first non-rest state as the incremental state.

    Returns:
        DataFrame containing all records in the file

    """
    # Read file and generate DataFrame
    file = Path(file)
    if file.suffix == ".nda":
        df, metadata = read_nda(file)
    elif file.suffix == ".ndax":
        df, metadata = read_ndax(file)
    else:
        msg = "File type not supported!"
        raise ValueError(msg)

    # Generate cycle number if requested
    if software_cycle_number:
        df = _generate_cycle_number(df, cycle_mode)

    # round time to ms, Status -> categories, uts -> Timestamp
    cols = [
        pl.col("step_time_s").round(3),
        pl.col("status").replace_strict(state_dict, default=None).alias("status"),
    ]
    if "unix_time_s" in df.columns:
        cols += [pl.from_epoch(pl.col("unix_time_s"), time_unit="s").alias("timestamp")]
    df = df.with_columns(cols)

    # Ensure columns have correct data types
    df = df.with_columns([pl.col(name).cast(dtype_dict[name]) for name in df.columns if name in dtype_dict])

    # Reorder columns
    non_aux_columns = [name for name in dtype_dict if name in df.columns]
    aux_columns = [name for name in df.columns if name.startswith("aux")]
    df = df.select(non_aux_columns + aux_columns)

    return df, metadata
