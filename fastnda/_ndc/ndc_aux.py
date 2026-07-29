# Copyright © 2026, Empa.
"""Private module to read Neware 'aux' NDC files.

Do not use these methods directly, they may change any time without warning.
"""

import numpy as np
import polars as pl

from fastnda._ndc.ndc_utils import bytes_to_df


def read_ndc_aux_2(buf: bytes) -> pl.DataFrame:
    # This dtype is missing humudity % column - does not exist in current test data
    dtype = np.dtype(
        [
            ("_pad2", "V8"),
            ("index", "<u4"),
            ("_pad3", "V19"),
            ("voltage_V", "<i4"),
            ("_pad4", "V6"),
            ("temperature_degC", "<i2"),
            ("temperature_setpoint_degC", "<i2"),
            ("_pad5", "V49"),
        ]
    )
    df = bytes_to_df(buf, dtype, data_start_ind=5, record_size=512, use_bitmask=False).with_columns(
        pl.col("voltage_V").cast(pl.Float32) / 10000,
        pl.col("temperature_degC").cast(pl.Float32) * 0.1,
        pl.col("temperature_setpoint_degC").cast(pl.Float32) * 0.1,
    )
    # Drop empty columns
    cols_to_drop = [
        col
        for col in ["voltage_V", "temperature_degC", "temperature_setpoint_degC"]
        if df.filter(pl.col(col) != 0).is_empty()
    ]
    return df.select(pl.exclude(cols_to_drop))


def read_ndc_aux_5(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("_pad2", "V1"),
            ("index", "<u4"),
            ("_pad3", "V19"),
            ("voltage_V", "<i4"),
            ("_pad4", "V6"),
            ("temperature_degC", "<i2"),
            ("temperature_setpoint_degC", "<i2"),
            ("_pad5", "V49"),
        ]
    )
    df = bytes_to_df(buf, dtype).with_columns(
        pl.col("voltage_V").cast(pl.Float32) * 1e-4,
        pl.col("temperature_degC").cast(pl.Float32) * 0.1,
        pl.col("temperature_setpoint_degC").cast(pl.Float32) * 0.1,
    )
    # Drop empty columns
    cols_to_drop = [
        col
        for col in ["voltage_V", "temperature_degC", "temperature_setpoint_degC"]
        if df.filter(pl.col(col) != 0).is_empty()
    ]
    return df.select(pl.exclude(cols_to_drop))


def read_ndc_aux_6(buf: bytes) -> pl.DataFrame:
    dtype = np.dtype(
        [
            ("?", "<f4"),  # Column name is assigned later from TestInfo.xml
        ]
    )
    return bytes_to_df(buf, dtype, add_index=True).with_columns(
        [
            pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("index"),
        ]
    )


def read_ndc_aux_11(buf: bytes) -> pl.DataFrame:
    header = 4096
    identifier = buf[header + 132 : header + 133]
    if identifier == b"\x65":
        dtype = np.dtype(
            [
                ("_mask", "<i1"),
                ("voltage_V", "<f4"),
                ("temperature_degC", "<i2"),
            ]
        )
        df = bytes_to_df(buf, dtype).with_columns(
            [
                pl.col("voltage_V") * 1e-4,  # 0.1 mV -> V
                pl.col("temperature_degC").cast(pl.Float32) * 0.1,  # 0.1'C -> 'C
                pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("index"),
            ]
        )
        # Drop empty columns
        cols_to_drop = [col for col in ["voltage_V", "temperature_degC"] if df.filter(pl.col(col) != 0).is_empty()]
        return df.select(pl.exclude(cols_to_drop))

    if identifier == b"\x74":
        dtype = np.dtype(
            [
                ("_pad1", "V1"),
                ("index", "<u4"),
                ("Aux", "<i1"),
                ("_pad2", "V29"),
                ("temperature_degC", "<i2"),
                ("_pad3", "V51"),
            ]
        )
        return (
            bytes_to_df(buf, dtype)
            .with_columns(
                [
                    pl.col("temperature_degC").cast(pl.Float32) * 0.1,  # 0.1'C -> 'C
                ]
            )
            .drop("Aux")
        )  # Aux channel inferred from TestInfo.xml

    msg = "Unknown file structure for ndc version 11 filetype 5."
    raise NotImplementedError(msg)


def read_ndc_aux_16(buf: bytes) -> pl.DataFrame:
    header = 4096
    if buf[header + 132 : header + 133] == b"\x65":
        dtype = np.dtype(
            [
                ("_mask", "<i1"),
                ("voltage_V", "<f4"),
                ("temperature_degC", "<i2"),
            ]
        )
        df = bytes_to_df(buf, dtype).with_columns(
            [
                pl.col("voltage_V") / 10000,  # 0.1 mV -> V
                pl.col("temperature_degC").cast(pl.Float32) * 0.1,  # 0.1'C -> 'C
                pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("index"),
            ]
        )
        # Drop empty columns
        cols_to_drop = [col for col in ["voltage_V", "temperature_degC"] if df.filter(pl.col(col) != 0).is_empty()]
        return df.select(pl.exclude(cols_to_drop))
    msg = "Unknown file structure for ndc version 16 filetype 5."
    raise NotImplementedError(msg)
