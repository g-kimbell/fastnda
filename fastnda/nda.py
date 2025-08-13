"""Module to read Neware NDA files."""

import logging
import mmap
import struct
from pathlib import Path

import numpy as np
import polars as pl

from fastnda.dicts import multiplier_dict
from fastnda.utils import _count_changes

logger = logging.getLogger(__name__)


def read_nda(file: str | Path) -> pl.DataFrame:
    """Read data from a Neware .nda binary file.

    Args:
        file: Path of .nda file to read
        software_cycle_number: Generate the cycle number field
            to match old versions of BTSDA
        cycle_mode: Selects how the cycle is incremented.
            'chg': (Default) Sets new cycles with a Charge step following a Discharge.
            'dchg': Sets new cycles with a Discharge step following a Charge.
            'auto': Identifies the first non-rest state as the incremental state.

    Returns:
        DataFrame containing all records in the file
        Metadata dictionary with file version and other info

    """
    file = Path(file)
    with file.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        if mm.read(6) != b"NEWARE":
            msg = f"{file} does not appear to be a Neware file."
            raise ValueError(msg)
        # Get the file version
        nda_version = int(mm[14])
        logger.info("Reading nda version %s", nda_version)

        # Try to find server and client version info
        version_loc = mm.find(b"BTSServer")
        if version_loc != -1:
            mm.seek(version_loc)
            server = mm.read(50).strip(b"\x00").decode()
            logger.info("Server version: %s", server)

            mm.seek(50, 1)
            client = mm.read(50).strip(b"\x00").decode()
            logger.info("Client version: %s", client)
        else:
            logger.info("BTS version not found!")

        # version specific settings
        if nda_version == 29:
            logger.info("Reading nda version 29")
            df, aux_df = _read_nda_29(mm)
        elif nda_version == 130:
            if mm[1024:1025] == b"\x55":  # It is BTS 9.1
                logger.info("Reading nda version 130 BTS9.1")
                df, aux_df = _read_nda_130_91(mm)
            else:
                logger.info("Reading nda version 130 BTS9.0")
                df, aux_df = _read_nda_130_90(mm)
        else:
            msg = f"nda version {nda_version} is not yet supported!"
            raise NotImplementedError(msg)

    # Drop duplicate indexes and sort
    df = df.unique(subset="index")
    df = df.sort(by="index")

    # Join temperature data
    if not aux_df.is_empty():
        if "aux" in aux_df.columns:
            aux_df = aux_df.unique(subset=["index", "aux"])
            aux_df = aux_df.pivot(index="index", on="aux", separator="")
            # Rename - add number to aux prefix e.g. aux1_voltage_volt
            aux_df.columns = [f"aux{col[-1]}_{col[4:-1]}" if col != "index" else "index" for col in aux_df.columns]
        else:
            aux_df = aux_df.unique(subset=["index"])
        df = df.join(aux_df, on="index", how="left")

    return df


def read_nda_metadata(file: str | Path) -> dict[str, str | float]:
    """Read metadata from a Neware .nda file."""
    file = Path(file)
    with file.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        if mm.read(6) != b"NEWARE":
            msg = f"{file} does not appear to be a Neware file."
            raise ValueError(msg)
        metadata = {}
        # Get the file version
        metadata["nda_version"] = int(mm[14])

        # Try to find server and client version info
        version_loc = mm.find(b"BTSServer")
        if version_loc != -1:
            mm.seek(version_loc)
            server = mm.read(50).strip(b"\x00").decode()
            metadata["server"] = server

            mm.seek(50, 1)
            client = mm.read(50).strip(b"\x00").decode()
            metadata["client"] = client
        else:
            logger.info("BTS version not found!")

    return metadata


def _read_nda_29(mm: mmap.mmap) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read nda version 29, return data and aux DataFrames."""
    mm_size = mm.size()

    # Get the active mass
    [active_mass] = struct.unpack("<I", mm[152:156])
    logger.info("Active mass: %s mg", active_mass / 1000)

    try:
        remarks = mm[2317:2417].decode("ASCII")
        # Clean null characters
        remarks = remarks.replace(chr(0), "").strip()
        logger.info("Remarks: %s", remarks)
    except UnicodeDecodeError:
        logger.warning("Converting remark bytes into ASCII failed")
        remarks = ""

    # Identify the beginning of the data section
    record_len = 86
    identifier = b"\x00\x00\x00\x00\x55\x00"
    header = mm.find(identifier)
    if header == -1:
        msg = "File does not contain any valid records."
        raise EOFError(msg)
    while (
        ((mm[header + 4 + record_len] != 85) | (not _valid_record(mm[header + 4 : header + 4 + record_len])))
        if header + 4 + record_len < mm_size
        else False
    ):
        header = mm.find(identifier, header + 4)
    mm.seek(header + 4)

    # Read data records
    num_records = (len(mm) - header - 4) // record_len
    arr = np.frombuffer(mm[header + 4 :], dtype=np.int8).reshape((num_records, record_len))
    # Remove rows where last 4 bytes are zero
    mask = (arr[:, 82:].view(np.int32) == 0).flatten()
    arr = arr[mask]

    # Split into two arrays, one for data and one for aux

    # Data array - first byte is \x55
    data_mask = arr[:, 0] == 85
    data_dtype = np.dtype(
        [
            ("_pad1", "V2"),  # 0-1
            ("index", np.uint32),  # 2-5
            ("cycle_count", np.uint32),  # 6-9
            ("step_index", np.uint16),  # 10-11
            ("status", np.uint8),  # 12
            ("step_count", np.uint8),  # 13 (records jumps)
            ("step_time_s", np.uint64),  # 14-21
            ("voltage_V", np.int32),  # 22-25
            ("current_mA", np.int32),  # 26-29
            ("_pad3", "V8"),  # 30-37
            ("charge_capacity_mAh", np.int64),  # 38-45
            ("discharge_capacity_mAh", np.int64),  # 46-53
            ("charge_energy_mWh", np.int64),  # 54-61
            ("discharge_energy_mWh", np.int64),  # 62-69
            ("Y", np.uint16),  # 70-71
            ("M", np.uint8),  # 72
            ("D", np.uint8),  # 73
            ("h", np.uint8),  # 74
            ("m", np.uint8),  # 75
            ("s", np.uint8),  # 76
            ("_pad4", "V1"),  # 77
            ("range", np.int32),  # 78-81
            ("_pad5", "V4"),  # 82-85
        ]
    )
    assert data_dtype.names is not None  # noqa: S101
    data_dtype_no_pad = data_dtype[[name for name in data_dtype.names if not name.startswith("_")]]
    data_arr = arr[data_mask].view(data_dtype_no_pad).flatten()
    data_df = pl.DataFrame(data_arr)
    data_df = (
        data_df.with_columns(
            [
                pl.col("cycle_count") + 1,
                pl.col("step_time_s").cast(pl.Float32) / 1000,
                pl.col("voltage_V").cast(pl.Float32) / 10000,
                pl.col("range").replace_strict(multiplier_dict, return_dtype=pl.Float64).alias("multiplier"),
                pl.datetime(pl.col("Y"), pl.col("M"), pl.col("D"), pl.col("h"), pl.col("m"), pl.col("s")).alias(
                    "timestamp"
                ),
                _count_changes(pl.col("step_count")).alias("step_count"),
            ]
        )
        .with_columns(
            [
                pl.col("current_mA") * pl.col("multiplier"),
                (
                    pl.col(
                        ["charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh"],
                    ).cast(pl.Float64)
                    * pl.col("multiplier").cast(pl.Float64)
                    / 3600
                ).cast(pl.Float32),
                (pl.col("timestamp").cast(pl.Float64) * 1e-6).alias("unix_time_s"),
            ]
        )
        .drop(["Y", "M", "D", "h", "m", "s", "multiplier", "range"])
    )

    # Aux array - first byte is \x65
    aux_mask = arr[:, 0] == 101
    aux_dtype = np.dtype(
        [
            ("_pad1", "V1"),  # 0
            ("aux", np.uint8),  # 1
            ("index", np.uint32),  # 2-5
            ("_pad2", "V16"),  # 6-21
            ("aux_voltage_volt", np.int32),  # 22-25
            ("_pad3", "V8"),  # 26-33
            ("aux_temperature_degC", np.int16),  # 34-35
            ("_pad4", "V50"),  # 36-81
        ]
    )
    assert aux_dtype.names is not None  # noqa: S101
    aux_dtype_no_pad = aux_dtype[[name for name in aux_dtype.names if not name.startswith("_")]]
    aux_arr = arr[aux_mask].view(aux_dtype_no_pad).flatten()
    aux_df = pl.DataFrame(aux_arr)
    aux_df = aux_df.with_columns(
        [
            pl.col("aux_temperature_degC").cast(pl.Float32) / 10,  # 0.1'C -> 'C
            pl.col("aux_voltage_volt").cast(pl.Float32) / 10000,  # 0.1 mV -> V
        ]
    )

    return data_df, aux_df


def _read_nda_130_91(mm: mmap.mmap) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read nda version 130 BTS9.1, return data and aux DataFrames."""
    record_len = mm.find(mm[1024:1026], 1026) - 1024  # Get record length
    _read_footer(mm)  # Log metadata
    num_records = (len(mm) - 2048) // record_len

    # Read data
    arr = np.frombuffer(mm[1024 : 1024 + num_records * record_len], dtype=np.int8).reshape((num_records, record_len))

    # In BTS9.1, data and aux are in the same rows
    mask = (arr[:, 0] == 85) & (arr[:, 8:12].view(np.uint32) != 0).flatten()
    dtype_list = [
        ("_pad1", "V2"),  # 0-1
        ("step_index", np.uint8),  # 2
        ("status", np.uint8),  # 3
        ("cycle_count", np.uint32),  # 4-7
        ("index", np.uint32),  # 8-11
        ("step_time_s", np.uint32),  # 12-15
        ("time_ns", np.uint32),  # 16-19
        ("current_mA", np.float32),  # 20-23
        ("voltage_V", np.float32),  # 24-27
        ("capacity_mAs", np.float32),  # 28-31
        ("energy_mWs", np.float32),  # 32-35
        ("_pad3", "V8"),  # 36-43
        ("unix_time_s", np.uint32),  # 44-47
        ("uts_ns", np.uint32),  # 48-51
    ]
    if record_len > 52:
        dtype_list.append(("_pad4", f"V{record_len - 52}"))
    data_dtype = np.dtype(dtype_list)
    assert data_dtype.names is not None  # noqa: S101
    data_dtype_no_pad = data_dtype[[name for name in data_dtype.names if not name.startswith("_")]]

    # Mask, view, flatten, recalculate some columns
    data_arr = arr[mask].view(data_dtype_no_pad)
    data_arr = data_arr.flatten()
    data_df = pl.DataFrame(data_arr)
    data_df = data_df.with_columns(
        [
            pl.col("capacity_mAs").clip(lower_bound=0).alias("charge_capacity_mAh") / 3600,
            pl.col("capacity_mAs").clip(upper_bound=0).abs().alias("discharge_capacity_mAh") / 3600,
            pl.col("energy_mWs").clip(lower_bound=0).alias("charge_energy_mWh") / 3600,
            pl.col("energy_mWs").clip(upper_bound=0).abs().alias("discharge_energy_mWh") / 3600,
            (pl.col("step_time_s") + pl.col("time_ns") / 1e9).cast(pl.Float32).alias("step_time_s"),
            (pl.col("unix_time_s") + pl.col("uts_ns") / 1e9).alias("unix_time_s"),
            _count_changes(pl.col("step_index")).alias("step_count"),
        ]
    )
    data_df = data_df.drop(["unix_time_s", "uts_ns", "energy_mWs", "capacity_mAs", "time_ns"])

    # If the record length is 56, then there is an additional temperature column
    # Read into separate DataFrame and merge later for compatibility with other versions
    if record_len == 56:
        aux_dtype = np.dtype(
            [
                ("_pad1", "V8"),  # 0-7
                ("index", np.uint32),  # 8-11
                ("_pad2", "V40"),  # 12-51
                ("aux_temperature_degC", np.float32),  # 52-55
            ]
        )
        assert aux_dtype.names is not None  # noqa: S101
        aux_dtype_no_pad = aux_dtype[[name for name in aux_dtype.names if not name.startswith("_")]]
        aux_arr = arr[mask].view(aux_dtype_no_pad)
        aux_arr = aux_arr.flatten()
        aux_df = pl.DataFrame(aux_arr)
    else:
        aux_df = pl.DataFrame()

    return data_df, aux_df


def _read_nda_130_90(mm: mmap.mmap) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read nda version 130 BTS9.0, return data and aux DataFrames."""
    record_len = 88
    _read_footer(mm)  # Log metadata
    num_records = (len(mm) - 2048) // record_len

    # Read data
    arr = np.frombuffer(mm[1024 : 1024 + num_records * record_len], dtype=np.int8).reshape((num_records, record_len))

    # Data and aux stored in different rows
    data_mask = np.all(arr[:, :6] == arr[0, :6], axis=1).flatten()
    aux_mask = (arr[:, 1:5].view(np.int32) == 101).flatten()

    data_dtype = np.dtype(
        [
            ("_pad1", "V9"),  # 0-8
            ("step_index", np.uint8),  # 9
            ("status", np.uint8),  # 10
            ("_pad2", "V5"),  # 11-15
            ("index", np.uint32),  # 16-19
            ("_pad3", "V8"),  # 20-27
            ("step_time_s", np.uint64),  # 28-35
            ("voltage_V", np.float32),  # 36-39
            ("current_mA", np.float32),  # 40-43
            ("_pad4", "V8"),  # 44-51
            ("charge_capacity_mAh", np.float32),  # 52-55
            ("charge_energy_mWh", np.float32),  # 56-59
            ("discharge_capacity_mAh", np.float32),  # 60-63
            ("discharge_energy_mWh", np.float32),  # 64-67
            ("unix_time_s", np.uint64),  # 68-75
            ("_pad5", "V12"),  # 76-87
        ]
    )
    assert data_dtype.names is not None  # noqa: S101
    data_dtype_no_pad = data_dtype[[name for name in data_dtype.names if not name.startswith("_")]]
    data_arr = arr[data_mask].view(data_dtype_no_pad)
    data_arr = data_arr.flatten()
    data_df = pl.DataFrame(data_arr)
    data_df = data_df.with_columns(
        [
            pl.col("unix_time_s").cast(pl.Float64) / 1e6,  # us -> s
            (pl.col("step_time_s") / 1e6).cast(pl.Float32),  # us -> s
            pl.col(["charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh"])
            / 3600,
            _count_changes(pl.col("step_index")).alias("step_count"),
        ]
    )

    aux_dtype = np.dtype(
        [
            ("_pad1", "V5"),  # 0-4
            ("aux", np.uint8),  # 5
            ("index", np.uint32),  # 6-9
            ("_pad2", "V16"),  # 10-25
            ("aux_voltage_volt", np.int32),  # 26-29
            ("_pad3", "V8"),  # 30-37
            ("aux_temperature_degC", np.int16),  # 38-41
            ("_pad4", "V48"),  # 42-87
        ]
    )
    assert aux_dtype.names is not None  # noqa: S101
    aux_dtype_no_pad = aux_dtype[[name for name in aux_dtype.names if not name.startswith("_")]]
    aux_arr = arr[aux_mask].view(aux_dtype_no_pad)
    aux_arr = aux_arr.flatten()
    aux_df = pl.DataFrame(aux_arr)
    aux_df = aux_df.with_columns(
        [
            pl.col("aux_temperature_degC").cast(pl.Float32) / 10,  # 0.1'C -> 'C
            pl.col("aux_voltage_volt").cast(pl.Float32) / 10000,  # 0.1 mV -> V
        ]
    )

    return data_df, aux_df


def _read_footer(mm: mmap.mmap) -> None:
    # Identify footer
    footer = mm.rfind(b"\x06\x00\xf0\x1d\x81\x00\x03\x00\x61\x90\x71\x90\x02\x7f\xff\x00", 1024)
    if footer:
        mm.seek(footer + 16)
        buf = mm.read(499)

        # Get the active mass
        [active_mass] = struct.unpack("<d", buf[-8:])
        logger.info("Active mass: %s mg", active_mass)

        # Get the remarks
        remarks = buf[363:491].decode("ASCII")

        # Clean null characters
        remarks = remarks.replace(chr(0), "").strip()
        logger.info("Remarks: %s", remarks)


def _valid_record(buf: bytes) -> bool:
    """Identify a valid record."""
    # Check for a non-zero Status
    [Status] = struct.unpack("<B", buf[12:13])
    return Status != 0
