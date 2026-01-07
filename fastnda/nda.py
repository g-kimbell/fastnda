"""Module to read Neware NDA files."""

import logging
import mmap
import struct
from pathlib import Path

import numpy as np
import polars as pl

from fastnda.dicts import MULTIPLIER_MAP
from fastnda.utils import _count_changes

logger = logging.getLogger(__name__)


def read_nda(file: str | Path) -> pl.DataFrame:
    """Read data from a Neware .nda binary file.

    Args:
        file: Path of .nda file to read

    Returns:
        DataFrame containing all records in the file

    """
    file = Path(file)
    with file.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        if mm.read(6) != b"NEWARE":
            msg = f"{file} does not appear to be a Neware file."
            raise ValueError(msg)
        # Get the NDA version
        nda_version = int(mm[14])

        # Reading depends on the NDA version
        if nda_version == 8:
            logger.info("Reading NDA version 8")
            df = _read_nda_8(mm)
            aux_df = pl.DataFrame()
        elif nda_version == 29:
            logger.info("Reading NDA version 29")
            df, aux_df = _read_nda_29(mm)
        elif nda_version == 130:
            if mm[1024:1025] == b"\x55":
                logger.info("Reading NDA version 130 BTS9.1")
                df, aux_df = _read_nda_130_91(mm)
            else:
                logger.info("Reading NDA version 130 BTS9.0")
                df, aux_df = _read_nda_130_90(mm)
        else:
            msg = f"NDA version {nda_version} is not yet supported!"
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


def read_nda_metadata(file: str | Path) -> dict[str, str | int | float]:
    """Read metadata from a Neware .nda file.

    Args:
        file: Path of .nda file to read

    Returns:
        Dictionary containing metadata

    """
    file = Path(file)
    with file.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    if mm.read(6) != b"NEWARE":
        msg = f"{file} does not appear to be a Neware file."
        raise ValueError(msg)

    metadata: dict[str, int | str | float] = {}

    # Get the file version
    metadata["nda_version"] = int(mm[14])

    # Try to find server and client version info
    version_loc = mm.find(b"BTSServer")
    if version_loc != -1:
        mm.seek(version_loc)
        server = mm.read(50).strip(b"\x00").decode()
        metadata["server_version"] = server

        mm.seek(50, 1)
        client = mm.read(50).strip(b"\x00").decode()
        metadata["client_version"] = client
    else:
        logger.info("BTS version not found!")

    # NDA 29 specific fields
    if metadata["nda_version"] == 29:
        metadata["active_mass_mg"] = int.from_bytes(mm[152:156], "little") / 1000
        metadata["remarks"] = mm[2317:2417].decode("ASCII", errors="ignore").replace(chr(0), "").strip()

    # NDA 130 specific fields
    elif metadata["nda_version"] == 130:
        # Identify footer
        footer = mm.rfind(b"\x06\x00\xf0\x1d\x81\x00\x03\x00\x61\x90\x71\x90\x02\x7f\xff\x00", 1024)
        if footer:
            mm.seek(footer + 16)
            buf = mm.read(499)
            metadata["active_mass_mg"] = struct.unpack("<d", buf[-8:])[0]
            metadata["remarks"] = buf[363:491].decode("ASCII").replace(chr(0), "").strip()

    return metadata


def _read_nda_8(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 8, return data and aux DataFrames."""
    # Identify the beginning of the data section - first byte 255 and index = 1
    record_len = 59
    identifier = b"\xff\x01\x00\x00\x00"
    header = mm.find(identifier)
    if header == -1:
        msg = "Could not find start of data section."
        raise EOFError(msg)
    header = header + record_len
    num_records = (len(mm) - header) // record_len
    end = len(mm) - (len(mm) - header) % record_len
    arr = np.frombuffer(mm[header:end], dtype=np.int8).reshape((num_records, record_len))

    data_dtype = np.dtype(
        [
            ("_pad1", "V1"),
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("step_time_s", "<u4"),
            ("voltage_V", "<i4"),  # /10000
            ("current_mA", "<i4"),  # /10000
            ("_pad2", "V8"),
            ("capacity_mAh", "<i8"),  # /3600000
            ("energy_mWh", "<i8"),  # /3600000
            ("unix_time_s", "<u8"),
            ("_pad3", "V4"),  # Possibly a checksum
        ]
    )
    assert data_dtype.names is not None  # noqa: S101
    data_dtype_no_pad = data_dtype[[name for name in data_dtype.names if not name.startswith("_")]]
    data_arr = arr.view(data_dtype_no_pad).flatten()
    return pl.DataFrame(data_arr).with_columns(
        [
            pl.col("step_time_s").cast(pl.Float32),
            pl.col("voltage_V").cast(pl.Float32) / 10000,
            pl.col("current_mA").cast(pl.Float32) / 10000,
            (pl.col("capacity_mAh").cast(pl.Float64) * pl.col("current_mA").sign()) / 3600000,
            (pl.col("energy_mWh").cast(pl.Float64) * pl.col("current_mA").sign()) / 3600000,
            _count_changes(pl.col("step_index")).alias("step_count"),
        ]
    )


def _read_nda_29(mm: mmap.mmap) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read nda version 29, return data and aux DataFrames."""
    # Identify the beginning of the data section - first byte 85 and index = 1
    identifier = b"\x55\x00\x01\x00\x00\x00"
    header = mm.find(identifier)
    if header == -1:
        msg = "Could not find start of data section."
        raise EOFError(msg)

    # Read data records
    record_len = 86
    num_records = (len(mm) - header) // record_len
    arr = np.frombuffer(mm[header:], dtype=np.int8).reshape((num_records, record_len))

    # Remove rows where last 4 bytes are zero
    mask = (arr[:, 82:].view(np.int32) == 0).flatten()
    arr = arr[mask]

    # Split into two arrays, one for data and one for aux

    # Data array - first byte is \x55
    data_mask = arr[:, 0] == 85
    data_dtype = np.dtype(
        [
            ("_pad1", "V2"),
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u2"),
            ("step_type", "<u1"),
            ("step_count", "<u1"),  # Records jumps
            ("step_time_s", "<u8"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad3", "V8"),
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
            ("_pad4", "V1"),
            ("range", "<i4"),
            ("_pad5", "V4"),
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
                pl.col("range").replace_strict(MULTIPLIER_MAP, return_dtype=pl.Float64).alias("multiplier"),
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
            ("_pad1", "V1"),
            ("aux", "<u1"),
            ("index", "<u4"),
            ("_pad2", "V16"),
            ("aux_voltage_volt", "<i4"),
            ("_pad3", "V8"),
            ("aux_temperature_degC", "<i2"),
            ("_pad4", "V50"),
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
    num_records = (len(mm) - 2048) // record_len

    # Read data
    arr = np.frombuffer(mm[1024 : 1024 + num_records * record_len], dtype=np.int8).reshape((num_records, record_len))

    # In BTS9.1, data and aux are in the same rows
    mask = (arr[:, 0] == 85) & (arr[:, 8:12].view(np.uint32) != 0).flatten()
    dtype_list = [
        ("_pad1", "V2"),
        ("step_index", "<u1"),
        ("step_type", "<u1"),
        ("_pad2", "V4"),
        ("index", "<u4"),
        ("total_time_s", "<u4"),
        ("time_ns", "<u4"),
        ("current_mA", "<f4"),
        ("voltage_V", "<f4"),
        ("capacity_mAs", "<f4"),
        ("energy_mWs", "<f4"),
        ("cycle_count", "<u4"),
        ("_pad3", "V4"),
        ("unix_time_s", "<u4"),
        ("uts_ns", "<u4"),
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
            (pl.col("total_time_s") + pl.col("time_ns") / 1e9).cast(pl.Float32),
            (pl.col("unix_time_s") + pl.col("uts_ns") / 1e9).alias("unix_time_s"),
            pl.col("cycle_count") + 1,
            _count_changes(pl.col("step_index")).alias("step_count"),
        ]
    )
    # Need to calculate step times - not included in this NDA
    max_df = (
        data_df.group_by("step_count")
        .agg(pl.col("total_time_s").max().alias("max_total_time_s"))
        .sort("step_count")
        .with_columns(pl.col("max_total_time_s").shift(1).fill_null(0))
    )

    data_df = data_df.join(max_df, on="step_count", how="left").with_columns(
        (pl.col("total_time_s") - pl.col("max_total_time_s")).alias("step_time_s")
    )
    data_df = data_df.drop(["uts_ns", "energy_mWs", "capacity_mAs", "time_ns", "max_total_time_s"])

    # If the record length is 56, then there is an additional temperature column
    # Read into separate DataFrame and merge later for compatibility with other versions
    if record_len == 56:
        aux_dtype = np.dtype(
            [
                ("_pad1", "V8"),
                ("index", "<u4"),
                ("_pad2", "V40"),
                ("aux_temperature_degC", "<f4"),
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
    num_records = (len(mm) - 2048) // record_len

    # Read data
    arr = np.frombuffer(mm[1024 : 1024 + num_records * record_len], dtype=np.int8).reshape((num_records, record_len))

    # Data and aux stored in different rows
    data_mask = np.all(arr[:, :6] == arr[0, :6], axis=1).flatten()
    aux_mask = (arr[:, 1:5].view(np.int32) == 101).flatten()

    data_dtype = np.dtype(
        [
            ("_pad1", "V9"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("_pad2", "V5"),
            ("index", "<u4"),
            ("_pad3", "V8"),
            ("step_time_s", "<u8"),
            ("voltage_V", "<f4"),
            ("current_mA", "<f4"),
            ("_pad4", "V16"),
            ("capacity_mAh", "<f4"),
            ("energy_mWh", "<f4"),
            ("unix_time_s", "<u8"),
            ("_pad5", "V12"),
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
            pl.col(["capacity_mAh", "energy_mWh"]) / 3600,
            _count_changes(pl.col("step_index")).alias("step_count"),
        ]
    )

    aux_dtype = np.dtype(
        [
            ("_pad1", "V5"),
            ("aux", "<u1"),
            ("index", "<u4"),
            ("_pad2", "V16"),
            ("aux_voltage_volt", "<i4"),
            ("_pad3", "V8"),
            ("aux_temperature_degC", "<i2"),
            ("_pad4", "V48"),
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
