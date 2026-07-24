"""Module to read Neware 'step' NDC files."""

import numpy as np
import polars as pl

from fastnda._ndc.ndc_utils import bytes_to_df


def read_ndc_step_6(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("cycle_count", "<u4"),
            ("step_index", "<u4"),
            ("_pad1", "V16"),
            ("step_type", "<u1"),
            ("_pad2", "V12"),
        ]
    )
    return bytes_to_df(buf, dtype).with_columns(
        [
            pl.col("cycle_count") + 1,
            pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("step_count"),
        ]
    )


def read_ndc_step_16(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("cycle_count", "<u4"),
            ("step_index", "<u4"),
            ("_pad1", "V16"),
            ("step_type", "<u1"),
            ("_pad2", "V8"),
            ("index", "<u4"),
            ("_pad3", "V63"),
        ]
    )
    return bytes_to_df(buf, dtype).with_columns(
        [
            pl.col("cycle_count") + 1,
            pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("step_count"),
        ]
    )
