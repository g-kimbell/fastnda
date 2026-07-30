# Copyright © 2026, Empa.
"""Module to read Neware NDA files."""

import datetime
import logging
import mmap
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from fastnda.dicts import MULTIPLIER_MAP
from fastnda.utils import UnverifiedFormatWarning, _count_changes

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
        # Parse binary data to dataframe
        df = _read_nda(mm)

    # Drop duplicate indexes and sort
    df = df.unique(subset="index")
    return df.sort(by="index")


# Active mass in ug is stored at a fixed offset in the 2048-byte NDA header
# It is separate to the test info records, which have relative positions
# Tuple is (minimum version, byte osset)
_ACTIVE_MASS_OFFSETS: list[tuple[int, int]] = [
    (130, 330),
    (9, 152),
    (8, 144),
    (1, 80),
]


def _nda_active_mass_mg(mm: mmap.mmap, nda_version: int) -> float:
    """Read active material mass (mg) from the NDA header, for an nda_version 1-29 file."""
    offset = next(off for min_ver, off in _ACTIVE_MASS_OFFSETS if nda_version >= min_ver)
    return int.from_bytes(mm[offset : offset + 4], "little") / 1000


# Field name: (byte offset, byte length), relative to the start of the test info record
_TEST_INFO_FIELDS_V1: dict[str, tuple[int, int]] = {
    "start_time": (36, 20),
    "creator": (76, 15),
    "sn": (91, 20),
    "remarks": (111, 100),
}
_TEST_INFO_FIELDS_V11 = {
    **_TEST_INFO_FIELDS_V1,
    "start_time": (37, 20),
    "creator": (77, 60),
    "sn": (137, 90),
    "remarks": (227, 100),
}
_TEST_INFO_FIELDS_V17 = {**_TEST_INFO_FIELDS_V11, "barcode": (343, 40)}
_TEST_INFO_FIELDS_V29 = {**_TEST_INFO_FIELDS_V17, "test_name": (383, 60), "step_name": (443, 60)}

# (minimum version, test info record size in bytes, field layout for that record)
_TEST_INFO_LAYOUTS: list[tuple[int, int, dict[str, tuple[int, int]]]] = [
    (29, 503, _TEST_INFO_FIELDS_V29),
    (17, 383, _TEST_INFO_FIELDS_V17),
    (16, 343, _TEST_INFO_FIELDS_V11),  # v16 adds a 'parallel channel' field, offsets unaffected
    (11, 327, _TEST_INFO_FIELDS_V11),
    (1, 211, _TEST_INFO_FIELDS_V1),
]


def _read_nda_test_info(mm: mmap.mmap, nda_version: int) -> dict[str, str]:
    """Read the test info record from an nda_version 1-29 file.

    The header stores the test info begin byte and length at fixed offsets 24/28.
    The test info record struct changes with file version.
    Uses last test info record found.
    """
    struct_size, fields = next((size, f) for min_ver, size, f in _TEST_INFO_LAYOUTS if nda_version >= min_ver)

    test_begin = int.from_bytes(mm[24:28], "little")
    test_len = int.from_bytes(mm[28:32], "little")
    count = test_len // struct_size if test_begin and test_len else 0
    if count == 0:
        return {}
    record_offset = test_begin + (count - 1) * struct_size
    record = mm[record_offset : record_offset + struct_size]

    # Fixed-size null-terminated C strings, only return up to the first null, after that is garbled
    return {
        name: record[offset : offset + length].split(b"\x00", 1)[0].decode("gb2312", errors="ignore").strip()
        for name, (offset, length) in fields.items()
    }


def _read_nda_version_info(mm: mmap.mmap) -> dict[str, str]:
    """Read BTS server/client version strings, if present near the start of the file."""
    metadata: dict[str, str] = {}
    version_loc = mm.find(b"BTSServer")
    if version_loc != -1:
        mm.seek(version_loc)
        metadata["server_version"] = mm.read(50).strip(b"\x00").decode()
        mm.seek(50, 1)
        metadata["client_version"] = mm.read(50).strip(b"\x00").decode()
    else:
        xwj = mm.find(b"BTS_XWJ", 0, 1024)
        if xwj != -1:
            end = mm.find(b"\x00", xwj, 1024)
            if end != -1:
                metadata["server_version"] = mm[xwj:end].decode().strip()
        else:
            logger.info("BTS version not found!")
    return metadata


def _decode_text(raw: bytes) -> str:
    """Decode a fixed-width, null-terminated field."""
    return raw.split(b"\x00", 1)[0].decode("gb2312", errors="ignore").strip()


def _decode_u32(raw: bytes) -> int:
    """Decode a little-endian uint32 field."""
    return int.from_bytes(raw, "little")


def _decode_datetime_us(raw: bytes) -> str | None:
    """Decode a little-endian, microseconds-since-epoch uint64 field, or None if zero/out of range."""
    micros = int.from_bytes(raw, "little")
    if not micros:
        return None
    try:
        dt = datetime.datetime.fromtimestamp(micros / 1e6, tz=datetime.timezone.utc)
    except (OSError, OverflowError, ValueError):
        return None
    return dt.isoformat(timespec="milliseconds")


def _decode_hex(raw: bytes) -> str | None:
    """Decode a raw byte span as hex, or None if all-zero."""
    return raw.hex(" ") if any(raw) else None


# dtype name -> (byte length, decode function)
_FIELD_DECODERS: dict[str, tuple[int, Callable[[bytes], str | int | None]]] = {
    "text32": (32, _decode_text),
    "text64": (64, _decode_text),
    "u32": (4, _decode_u32),
    "datetime_us": (8, _decode_datetime_us),
    "hex21": (21, _decode_hex),
}


def _read_fields(record: bytes, fields: dict[str, tuple[int, str]]) -> dict[str, str | int]:
    """Decode a set of name -> (byte offset, dtype) fields from a record, relative to its start."""
    metadata: dict[str, str | int] = {}
    for name, (offset, dtype) in fields.items():
        length, decode = _FIELD_DECODERS[dtype]
        value = decode(record[offset : offset + length])
        if value is not None:
            metadata[name] = value
    return metadata


# Pack test info: name -> (byte offset, dtype), relative to the start of one record.
# Keys are best guesses based on current test data.
_PACK_TEST_INFO_FIELDS: dict[str, tuple[int, str]] = {
    "start_step_id": (4, "u32"),
    "creator": (8, "text32"),
    "sn": (40, "text32"),
    "UNKNOWN_1": (72, "text32"),  # desc? empty in test data
    "UNKNOWN_2": (104, "text64"),  # step_file_name? empty in test data
    "UNKNOWN_3": (168, "text64"),  # step_name? empty in test data
    "UNKNOWN_4": (232, "text32"),  # battery_model? empty in test data
    "remarks": (264, "text64"),
    "start_time": (432, "datetime_us"),
    "stop_time": (440, "datetime_us"),
}

# Older records (swjVer < 8) seem to have a sligthly different layout
_PACK_TEST_INFO_FIELDS_OLD: dict[str, tuple[int, str]] = {
    "UNKNOWN_19": (4, "u32"),  # too large for a step ID; constant across tests, maybe device-level
    "UNKNOWN_5": (8, "u32"),
    "creator": (12, "text32"),
    "sn": (44, "text32"),
    "UNKNOWN_6": (76, "text32"),  # desc? empty in test data
    "UNKNOWN_7": (108, "text64"),  # step_file_name? empty in test data
    "UNKNOWN_8": (172, "text64"),  # step_name? empty in test data
    "UNKNOWN_9": (236, "text32"),  # battery_model? empty in test data
    "remarks": (268, "text64"),
    "test_id": (396, "u32"),
    "start_time": (400, "datetime_us"),
    "stop_time": (408, "datetime_us"),
    "num_datapoints": (416, "u32"),
}


def _decode_pstring(data: bytes, pos: int) -> tuple[str, int]:
    """Decode one Pascal-style string (1-byte length prefix, no terminator); return (text, next_pos)."""
    length = data[pos]
    raw = data[pos + 1 : pos + 1 + length]
    return raw.decode("gb2312", errors="ignore"), pos + 1 + length


def _read_pack_test_info_old_extension(record: bytes) -> dict[str, str]:
    """Read records from 'pack test info' block in swjVer < 8."""
    try:
        pos = 420
        _bts_version, pos = _decode_pstring(record, pos)  # redundant with the file-wide bts_version search
        guid, pos = _decode_pstring(record, pos)
        guid_repeat, pos = _decode_pstring(record, pos)
        device_ip, pos = _decode_pstring(record, pos)
        unknown_5, pos = _decode_pstring(record, pos)  # unknown, "[org] - dedicated use" in examples
        unknown_6, pos = _decode_pstring(record, pos)  # unknown
        unknown_7, pos = _decode_pstring(record, pos)  # zero-length in real samples
        unknown_8, pos = _decode_pstring(record, pos)  # unknown
        pos += 4  # 4 fixed unknown bytes
        server_ip, _pos = _decode_pstring(record, pos)
    except IndexError:
        return {}
    return {
        "guid": guid,
        "guid2": guid_repeat,
        "device_ip": device_ip,
        "server_ip": server_ip,
        "UNKNOWN_10": unknown_5,
        "UNKNOWN_11": unknown_6,
        "UNKNOWN_12": unknown_7,
        "UNKNOWN_13": unknown_8,
    }


# Pack test info tail specific to swjVer >= 8, offsets relative to start of record.
_PACK_TEST_INFO_FIELDS_NEW_TAIL: dict[str, tuple[int, str]] = {
    "UNKNOWN_14": (328, "text32"),
    "server_ip": (360, "text64"),
    "test_id": (424, "u32"),  # Not very confident
    "num_datapoints": (428, "u32"),
    "UNKNOWN_15": (448, "u32"),  # zero in every sample seen
    "UNKNOWN_16": (452, "u32"),  # decodes to a nonsense 2010-era date, not a real timestamp
    "UNKNOWN_17": (456, "u32"),  # constant 1 in every sample seen
    "UNKNOWN_18": (460, "hex21"),  # rest of the record, resembles a pattern also seen in piLogEx
}

# piLogEx, doesn't seem to be present in the older BTS9.0.3 file.
# IP is probably device or middle machine, server_ip is usually 127.0.0.1
_PILOGEX_FIELDS: dict[str, tuple[int, str]] = {
    "device_ip": (223, "text32"),
    "hostname": (255, "text32"),
}


def _read_nda_130_log_ex(mm: mmap.mmap) -> dict[str, str | int]:
    """Read the 'log ex' block from an nda_version 130 file.

    The head info 9022 header stores a u64 {begin, length} pointer to this block
    at fixed offsets 242/250.
    """
    log_ex_begin = int.from_bytes(mm[242:250], "little")
    log_ex_len = int.from_bytes(mm[250:258], "little")
    if not log_ex_begin or not log_ex_len or log_ex_begin + log_ex_len > len(mm):
        return {}
    record = mm[log_ex_begin : log_ex_begin + log_ex_len]
    return _read_fields(record, _PILOGEX_FIELDS)


def _read_nda_130_test_info(mm: mmap.mmap) -> dict[str, str | int]:
    """Read records from 'pack test info' from an nda_version 130 file.

    The header stores a u64 {begin, length} pointer to this block at fixed offsets 34/42.
    """
    test_begin = int.from_bytes(mm[34:42], "little")
    test_len = int.from_bytes(mm[42:50], "little")
    if not test_begin or not test_len:
        return {}
    record = mm[test_begin : test_begin + test_len]
    swj_ver = int.from_bytes(record[0:2], "little")
    is_old = swj_ver < 8
    fields = _PACK_TEST_INFO_FIELDS_OLD if is_old else _PACK_TEST_INFO_FIELDS
    metadata = _read_fields(record, fields)
    if is_old:
        metadata.update(_read_pack_test_info_old_extension(record))
    else:
        metadata.update(_read_fields(record, _PACK_TEST_INFO_FIELDS_NEW_TAIL))
    return metadata


def _read_nda_130_metadata(mm: mmap.mmap) -> dict[str, str | float | int]:
    """Read metadata specific to nda_version 130 (BTS9.0/9.1)."""
    metadata: dict[str, str | float | int] = {}
    subver = int(mm[1024])
    if subver == 85:
        metadata["bts_version"] = "9.1"
        ver = mm.find(b"9.1.")
    elif subver == 18:
        metadata["bts_version"] = "9.0"
        ver = mm.find(b"9.0.")
    else:
        ver = -1
    if ver != -1:
        end = mm.find(b"\x00", ver)
        if end != 1:
            metadata["bts_version"] = mm[ver:end].decode()

    metadata["active_mass_mg"] = _nda_active_mass_mg(mm, 130)
    metadata.update(_read_nda_130_test_info(mm))
    metadata.update(_read_nda_130_log_ex(mm))
    return metadata


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
    nda_version = int(mm[14])
    metadata["nda_version"] = nda_version
    metadata.update(_read_nda_version_info(mm))

    # NDA 1-29 fields: header stores {nBegin, nLen} pointers to a device-info block and a test-info block
    if 1 <= nda_version <= 29:
        metadata["active_mass_mg"] = _nda_active_mass_mg(mm, nda_version)
        metadata.update(_read_nda_test_info(mm, nda_version))

    # NDA 130 specific fields
    elif nda_version == 130:
        warnings.warn(
            (
                "read_metadata for NDA 130 (BTS9) has not been thoroughly tested due to lack of test data. "
                "There may be wrong fields or 'UNKNOWN_x' keys present. "
                "If you can, please share a sample file at "
                "https://github.com/empaeconversion/fastnda/issues so we can improve reading this format."
            ),
            stacklevel=3,
        )
        metadata.update(_read_nda_130_metadata(mm))

    # Drop empty UNKNOWN_x fields
    def _unknown_is_empty(value: str | float) -> bool:
        return not value.strip("\x00").strip() if isinstance(value, str) else not value

    return {k: v for k, v in metadata.items() if not (k.startswith("UNKNOWN_") and _unknown_is_empty(v))}


def _find_header(mm: mmap.mmap, header: bytes | int) -> int:
    """Get header index."""
    if isinstance(header, int):
        return header
    header_idx = mm.find(header)
    if header_idx == -1:
        msg = "Could not find start of data section."
        raise EOFError(msg)
    return header_idx


def _get_arr_from_nda(
    mm: mmap.mmap,
    header: bytes | int,
    record_len: int,
) -> np.ndarray:
    """Read an nda file."""
    header_idx = _find_header(mm, header)
    num_records = (len(mm) - header_idx) // record_len
    return np.frombuffer(mm, dtype=np.uint8, count=num_records * record_len, offset=header_idx).reshape(
        (num_records, record_len)
    )


def _view_arr(
    arr: np.ndarray,
    dtype: np.dtype,
) -> pl.DataFrame:
    """Get polars dataframe from array, dropping padding columns."""
    assert dtype.names is not None  # noqa: S101
    dtype_no_pad = dtype[[name for name in dtype.names if not name.startswith("_")]]
    arr = arr.view(dtype_no_pad).ravel()
    return pl.DataFrame(arr)


def _mask_arr(
    arr: np.ndarray,
    dtype: np.dtype,
    mask: int,
) -> pl.DataFrame:
    """Get polars dataframe from array, filtered on an identifier value."""
    return _view_arr(arr, dtype).filter(pl.col("identifier") == mask).drop("identifier")


def _nda_head_main_begin(mm: mmap.mmap, *, pos_offset: int = 32, pos64: bool = False) -> int:
    """Read the data start offset from the header.

    File versions 1-11, 25, 27, store nBegin, nLen at a fixed offset.
    File versions 12-29 except 25,27 use a different fixed offset.
    File versions 129-130 use 64-bit pairs instead.
    """
    size = 8 if pos64 else 4
    return int.from_bytes(mm[pos_offset : pos_offset + size], "little")


def _merge_aux(
    df: pl.DataFrame,
    aux_df: pl.DataFrame,
) -> pl.DataFrame:
    """Merge aux left into data, renaming columns if aux channel in data."""
    if not aux_df.is_empty():
        if "aux" in aux_df.columns:
            aux_df = aux_df.unique(subset=["index", "aux"])
            aux_df = aux_df.pivot(index="index", on="aux", separator="")
            # Rename - add number to aux prefix e.g. aux1_voltage_volt
            aux_df.columns = [f"aux{col[-1]}_{col[4:-1]}" if col != "index" else "index" for col in aux_df.columns]
        else:
            aux_df = aux_df.unique(subset=["index"])
        return df.join(aux_df, on="index", how="left")
    return df


def _read_nda(mm: mmap.mmap) -> pl.DataFrame:
    """Figure out nda version and pass to correct reader."""
    nda_version = int(mm[14])
    reader = _NDA_READERS.get(nda_version)
    if reader is None:
        msg = f"nda version {nda_version} is not yet supported!"
        raise NotImplementedError(msg) from None
    if nda_version not in _CONFIRMED_NDA_VERSIONS:
        warnings.warn(
            f"nda_version {nda_version} has not been verified against real Neware data - results may be "
            "incorrect. If you can, please share a sample file at "
            "https://github.com/empaeconversion/fastnda/issues so we can confirm this format.",
            UnverifiedFormatWarning,
            stacklevel=2,
        )
    logger.debug("Reading nda version %d", nda_version)
    return reader(mm)


def _read_nda_1(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 1."""
    header_idx = _nda_head_main_begin(mm)
    arr = _get_arr_from_nda(mm, header=header_idx, record_len=38)
    dtype = np.dtype(
        [
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("step_time_s", "<u4"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad1", "V8"),  # nIR, iTemp - unpopulated in all files checked
            ("capacity_mAh", "<i8"),
        ]
    )
    return (
        _view_arr(arr, dtype)
        .filter(pl.col("index") != 0)
        .with_columns(
            [
                pl.col("cycle_count") + 1,
                pl.col("step_time_s").cast(pl.Float32),
                pl.col("voltage_V").cast(pl.Float32) / 10000,
                pl.col("current_mA").cast(pl.Float32) / 1000,
                (pl.col("capacity_mAh").cast(pl.Float64) * pl.col("current_mA").sign()) / 3600000,
                _count_changes(pl.col("step_index")).alias("step_count"),
            ]
        )
    )


def _read_nda_2(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 2 (deprecated by Neware - unreachable in BTSDA)."""
    header_idx = _nda_head_main_begin(mm)
    arr = _get_arr_from_nda(mm, header=header_idx, record_len=57)
    dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("step_time_s", "<u4"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad1", "V8"),  # nIR, iTemp
            ("capacity_mAh", "<i8"),
            ("_pad2", "V1"),  # bEng flag
            ("energy_mWh", "<i8"),
            ("_pad3", "V1"),  # bLocalTime flag
            ("unix_time_s", "<u8"),
        ]
    )
    return (
        _view_arr(arr, dtype)
        .filter(pl.col("identifier").is_in([0, 85]))
        .drop("identifier")
        .with_columns(
            [
                pl.col("cycle_count") + 1,
                pl.col("step_time_s").cast(pl.Float32),
                pl.col("voltage_V").cast(pl.Float32) / 10000,
                pl.col("current_mA").cast(pl.Float32) / 1000,
                (pl.col("capacity_mAh").cast(pl.Float64) * pl.col("current_mA").sign()) / 3600000,
                (pl.col("energy_mWh").cast(pl.Float64) * pl.col("current_mA").sign()) / 3600000,
                _count_changes(pl.col("step_index")).alias("step_count"),
            ]
        )
    )


def _read_nda_3(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 3 (file version 3, 4)."""
    header_idx = _nda_head_main_begin(mm)
    arr = _get_arr_from_nda(mm, header=header_idx, record_len=43)
    dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("step_time_s", "<u4"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad1", "V8"),  # nIR, iTemp
            ("capacity_mAh", "<i8"),
            ("_pad2", "V4"),  # dwCRC32
        ]
    )
    return (
        _view_arr(arr, dtype)
        .filter(pl.col("identifier").is_in([0, 85]))
        .drop("identifier")
        .with_columns(
            [
                pl.col("cycle_count") + 1,
                pl.col("step_time_s").cast(pl.Float32),
                pl.col("voltage_V").cast(pl.Float32) / 10000,
                pl.col("current_mA").cast(pl.Float32) / 1000,
                (pl.col("capacity_mAh").cast(pl.Float64) * pl.col("current_mA").sign()) / 3600000,
                _count_changes(pl.col("step_index")).alias("step_count"),
            ]
        )
    )


def _read_nda_5(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 5 (file versions 5, 6, 7, 8)."""
    # Identify the beginning of the data section - first byte 255 and index = 1
    arr = _get_arr_from_nda(mm, header=b"\xff\x01\x00\x00\x00", record_len=59)
    dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("step_time_s", "<u4"),
            ("voltage_V", "<i4"),  # /10000
            ("current_mA", "<i4"),  # /1000
            ("_pad2", "V8"),
            ("capacity_mAh", "<i8"),  # /3600000
            ("energy_mWh", "<i8"),  # /3600000
            ("unix_time_s", "<u8"),
            ("_pad3", "V4"),  # Possibly a checksum
        ]
    )
    return _mask_arr(arr, dtype, 0).with_columns(
        [
            pl.col("step_time_s").cast(pl.Float32),
            pl.col("voltage_V").cast(pl.Float32) / 10000,
            pl.col("current_mA").cast(pl.Float32) / 1000,
            (pl.col("capacity_mAh").cast(pl.Float64) * pl.col("current_mA").sign()) / 3600000,
            (pl.col("energy_mWh").cast(pl.Float64) * pl.col("current_mA").sign()) / 3600000,
            _count_changes(pl.col("step_index")).alias("step_count"),
        ]
    )


def _read_nda_9(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 9."""
    header_idx = _nda_head_main_begin(mm)
    arr = _get_arr_from_nda(mm, header=header_idx, record_len=60)
    dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("_pad0", "V1"),  # btAuxChlID
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("step_time_s", "<u4"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad1", "V8"),  # nIR, iTemp
            ("capacity_mAh", "<i8"),
            ("energy_mWh", "<i8"),
            ("unix_time_s", "<u8"),
            ("_pad2", "V4"),  # dwCRC32
        ]
    )
    return _mask_arr(arr, dtype, 85).with_columns(
        [
            pl.col("cycle_count") + 1,
            pl.col("step_time_s").cast(pl.Float32),
            pl.col("voltage_V").cast(pl.Float32) / 10000,
            pl.col("current_mA").cast(pl.Float32) / 1000,
            (pl.col("capacity_mAh").cast(pl.Float64) * pl.col("current_mA").sign()) / 3600000,
            (pl.col("energy_mWh").cast(pl.Float64) * pl.col("current_mA").sign()) / 3600000,
            _count_changes(pl.col("step_index")).alias("step_count"),
        ]
    )


def _read_nda_10(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 10."""
    header_idx = _nda_head_main_begin(mm)
    arr = _get_arr_from_nda(mm, header=header_idx, record_len=64)
    dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("_pad0", "V1"),  # btAuxChlID
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("step_time_s", "<u8"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad1", "V8"),  # nIR, iTemp
            ("capacity_mAh", "<i8"),
            ("energy_mWh", "<i8"),
            ("unix_time_s", "<u8"),
            ("_pad2", "V4"),  # dwCRC32
        ]
    )
    return _mask_arr(arr, dtype, 85).with_columns(
        [
            pl.col("cycle_count") + 1,
            pl.col("step_time_s").cast(pl.Float32) / 1000,
            pl.col("voltage_V").cast(pl.Float32) / 10000,
            pl.col("current_mA").cast(pl.Float32) / 1000,
            (pl.col("capacity_mAh").cast(pl.Float64) * pl.col("current_mA").sign()) / 3600000,
            (pl.col("energy_mWh").cast(pl.Float64) * pl.col("current_mA").sign()) / 3600000,
            _count_changes(pl.col("step_index")).alias("step_count"),
        ]
    )


def _read_nda_11(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda struct 11 (file versions 11, 12, 13, 15, 18)."""
    pos_offset = 32 if int(mm[14]) == 11 else 64
    header_idx = _nda_head_main_begin(mm, pos_offset=pos_offset)
    arr = _get_arr_from_nda(mm, header=header_idx, record_len=69)
    dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("_pad0", "V1"),  # btAuxChlID
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u2"),
            ("step_type", "<u1"),
            ("step_time_s", "<u8"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad1", "V8"),  # nIR, iTemp
            ("capacity_mAh", "<i8"),
            ("energy_mWh", "<i8"),
            ("unix_time_s", "<u8"),
            ("range", "<i4"),
            ("_pad2", "V4"),  # dwCRC32
        ]
    )
    return (
        _mask_arr(arr, dtype, 85)
        .with_columns(
            [
                pl.col("cycle_count") + 1,
                pl.col("step_time_s").cast(pl.Float32) / 1000,
                pl.col("voltage_V").cast(pl.Float32) / 10000,
                pl.col("range").replace_strict(MULTIPLIER_MAP, return_dtype=pl.Float64).alias("multiplier"),
                _count_changes(pl.col("step_index")).alias("step_count"),
            ]
        )
        .with_columns(
            [
                pl.col("current_mA") * pl.col("multiplier"),
                (
                    pl.col("capacity_mAh").cast(pl.Float64) * pl.col("multiplier") * pl.col("current_mA").sign() / 3600
                ).cast(pl.Float32),
                (
                    pl.col("energy_mWh").cast(pl.Float64) * pl.col("multiplier") * pl.col("current_mA").sign() / 3600
                ).cast(pl.Float32),
            ]
        )
        .drop(["multiplier", "range"])
    )


def _read_nda_14(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 14 (file versions 14, 16, 17, 20, 22, 23, 24)."""
    arr = _get_arr_from_nda(mm, b"\xaa\x00\x01\x00\x00\x00", 86)
    data_dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("_pad1", "V1"),
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u2"),
            ("step_type", "<u1"),
            ("step_count", "<u1"),
            ("step_time_s", "<u8"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad3", "V8"),
            ("charge_capacity_mAh", "<i8"),
            ("discharge_capacity_mAh", "<i8"),
            ("charge_energy_mWh", "<i8"),
            ("discharge_energy_mWh", "<i8"),
            ("unix_time_s", "<u8"),
            ("range", "<i4"),
            ("_pad5", "V4"),
        ]
    )
    mult_cols = ["charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh"]
    return (
        _mask_arr(arr, data_dtype, 85)
        .with_columns(
            [
                pl.col("cycle_count") + 1,
                pl.col("step_time_s").cast(pl.Float32) / 1000,
                pl.col("voltage_V").cast(pl.Float32) / 10000,
                pl.col("range").replace_strict(MULTIPLIER_MAP, return_dtype=pl.Float64).alias("multiplier"),
                _count_changes(pl.col("step_index")).alias("step_count"),
            ]
        )
        .with_columns(
            [
                pl.col("current_mA") * pl.col("multiplier"),
                (pl.col(mult_cols).cast(pl.Float64) * pl.col("multiplier").cast(pl.Float64) / 3600).cast(pl.Float32),
            ]
        )
        .drop(["multiplier", "range"])
    )


def _read_nda_19(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 19."""
    header_idx = _nda_head_main_begin(mm, pos_offset=64)
    arr = _get_arr_from_nda(mm, header=header_idx, record_len=68)
    dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("_pad0", "V3"),  # btSubDevID, btChannelID, btAuxChlID
            ("_pad0b", "V4"),  # dwTestID
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u2"),
            ("step_type", "<u1"),
            ("_pad1", "V1"),  # btWorkType
            ("_pad2", "V1"),  # btStepChgCount
            ("_pad3", "V3"),  # btReserved
            ("step_time_s", "<u4"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad4", "V8"),  # nIR, iTemp
            ("charge_capacity_mAh", "<i4"),
            ("discharge_capacity_mAh", "<i4"),
            ("charge_energy_mWh", "<i4"),
            ("discharge_energy_mWh", "<i4"),
            ("unix_time_s", "<u4"),
            ("_pad5", "V4"),  # dwCRC32
        ]
    )
    mult_cols = ["charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh"]
    return _mask_arr(arr, dtype, 85).with_columns(
        [
            pl.col("cycle_count") + 1,
            pl.col("step_time_s").cast(pl.Float32),
            pl.col("voltage_V").cast(pl.Float32) / 10000,
            pl.col("current_mA").cast(pl.Float32) / 1000,
            (pl.col(mult_cols).cast(pl.Float64) / 3600).cast(pl.Float32),
            _count_changes(pl.col("step_index")).alias("step_count"),
        ]
    )


def _read_nda_25(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 25 (file versions 25, 27)."""
    header_idx = _nda_head_main_begin(mm)
    arr = _get_arr_from_nda(mm, header=header_idx, record_len=70)
    dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("_pad0", "V1"),  # btAuxChlID
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u2"),
            ("step_type", "<u1"),
            ("step_time_s", "<u8"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad1", "V8"),  # nIR, iTemp
            ("capacity_mAh", "<i8"),
            ("energy_mWh", "<i8"),
            ("unix_time_s", "<u8"),
            ("range", "<i4"),
            ("_pad2", "V1"),  # btStepChgCount
            ("_pad3", "V4"),  # dwCRC32
        ]
    )
    return (
        _mask_arr(arr, dtype, 85)
        .with_columns(
            [
                pl.col("cycle_count") + 1,
                pl.col("step_time_s").cast(pl.Float32) / 1000,
                pl.col("voltage_V").cast(pl.Float32) / 10000,
                pl.col("range").replace_strict(MULTIPLIER_MAP, return_dtype=pl.Float64).alias("multiplier"),
                _count_changes(pl.col("step_index")).alias("step_count"),
            ]
        )
        .with_columns(
            [
                pl.col("current_mA") * pl.col("multiplier"),
                (
                    pl.col("capacity_mAh").cast(pl.Float64) * pl.col("multiplier") * pl.col("current_mA").sign() / 3600
                ).cast(pl.Float32),
                (
                    pl.col("energy_mWh").cast(pl.Float64) * pl.col("multiplier") * pl.col("current_mA").sign() / 3600
                ).cast(pl.Float32),
            ]
        )
        .drop(["multiplier", "range"])
    )


def _read_nda_29(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 29."""
    arr = _get_arr_from_nda(mm, b"\x55\x00\x01\x00\x00\x00", 86)
    data_dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("_pad1", "V1"),
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
    mult_cols = ["charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh"]
    data_df = (
        _mask_arr(arr, data_dtype, 85)
        .with_columns(
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
                (pl.col(mult_cols).cast(pl.Float64) * pl.col("multiplier").cast(pl.Float64) / 3600).cast(pl.Float32),
                (pl.col("timestamp").cast(pl.Float64) * 1e-6).alias("unix_time_s"),
            ]
        )
        .drop(["Y", "M", "D", "h", "m", "s", "multiplier", "range"])
    )

    aux_dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("aux", "<u1"),
            ("index", "<u4"),
            ("_pad2", "V16"),
            ("aux_voltage_V", "<i4"),
            ("_pad3", "V8"),
            ("aux_temperature_degC", "<i2"),
            ("_pad4", "V50"),
        ]
    )
    aux_df = _mask_arr(arr, aux_dtype, 101).with_columns(
        [
            pl.col("aux_temperature_degC").cast(pl.Float32) / 10,  # 0.1'C -> 'C
            pl.col("aux_voltage_V").cast(pl.Float32) / 10000,  # 0.1 mV -> V
        ]
    )
    return _merge_aux(data_df, aux_df)


def _read_nda_129(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 129 (deprecated by Neware)."""
    header_idx = _nda_head_main_begin(mm, pos_offset=82, pos64=True)
    arr = _get_arr_from_nda(mm, header=header_idx, record_len=88)
    dtype = np.dtype(
        [
            ("identifier", "<u1"),
            ("_pad0", "V5"),  # btDevType, btDevID, btUnitID, btChlID, btAuxChlIndex
            ("_pad0b", "V2"),  # wReserve
            ("_pad0c", "V4"),  # dwTestID
            ("index", "<u4"),  # dwTestDataSN
            ("_pad0d", "V4"),  # dwUnitNuid
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("_pad1", "V1"),  # btStepChgCount
            ("_pad2", "V1"),  # btReserve
            ("_pad3", "V4"),  # stWorkStatus
            ("step_time_s", "<u4"),  # Time64.dwS (seconds)
            ("step_time_ns", "<u4"),  # Time64.dwNS (nanoseconds)
            ("voltage_V", "<f4"),
            ("current_mA", "<f4"),
            ("_pad5", "V8"),  # fInterRes, fTempture
            ("charge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("unix_time_s", "<u8"),  # microseconds
            ("_pad6", "V12"),  # dwCurStepRange, dwLogCode, dwCRC32
        ]
    )
    mult_cols = ["charge_capacity_mAh", "discharge_capacity_mAh", "charge_energy_mWh", "discharge_energy_mWh"]
    return (
        _view_arr(arr, dtype)
        .filter(pl.col("identifier").is_in([0, 85]))
        .drop("identifier")
        .with_columns(
            [
                pl.col(mult_cols) / 3600,
                (pl.col("unix_time_s").cast(pl.Float64) / 1e6).alias("unix_time_s"),
                (pl.col("step_time_s").cast(pl.Float64) + pl.col("step_time_ns") / 1e9)
                .cast(pl.Float32)
                .alias("step_time_s"),
                _count_changes(pl.col("step_index")).alias("step_count"),
            ]
        )
        .drop("step_time_ns")
    )


def _read_nda_130(mm: mmap.mmap) -> pl.DataFrame:
    """Figure out whether BTS9.0 or BTS9.1 and pass to correct function."""
    subver = int(mm[1024])
    if subver == 85:
        return _read_nda_130_91(mm)
    if subver == 18:
        return _read_nda_130_90(mm)
    msg = f"nda 130 subversion {subver} not supported"
    raise NotImplementedError(msg)


def _read_nda_130_91(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 130 BTS9.1."""
    # Data starts at 1024, search forward for next identifier for record length
    identifier_bytes = mm[1024:1026]
    identifier_int = int.from_bytes(identifier_bytes, byteorder="little", signed=False)
    record_len = mm.find(mm[1024:1026], 1026) - 1024

    arr = _get_arr_from_nda(mm, 1024, record_len)

    # In BTS9.1, data and aux are in the same rows
    dtype_list = [
        ("identifier", "<u2"),
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
        ("_pad3", "V4"),  # Data here, looks like <f4 doesn't match anything in ref
        ("unix_time_s", "<u4"),
        ("uts_ns", "<u4"),
    ]
    if record_len >= 56:
        dtype_list += [("aux_temperature_degC", "<f4")]
    if record_len > 56:
        dtype_list.append(("_pad4", f"V{record_len - 56}"))
    data_dtype = np.dtype(dtype_list)

    data_df = _mask_arr(arr, data_dtype, identifier_int).with_columns(
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
    return data_df.drop(["uts_ns", "energy_mWs", "capacity_mAs", "time_ns", "max_total_time_s"])


def _read_nda_130_90(mm: mmap.mmap) -> pl.DataFrame:
    """Read nda version 130 BTS9.0."""
    # Data start seems to be (18, 80, 0, 7, 85, 129, 1, 6)
    # Aux identifiers are (18, 80, 0, 7, 88, 129, 1, 6) and (18, 80, 0, 7, 89, 129, 1, 6)
    arr = _get_arr_from_nda(mm, header=b"\x12\x50\x00\x07\x55\x81\x01\x06", record_len=88)
    data_dtype = np.dtype(
        [
            ("_pad1", "V4"),
            ("identifier", "<u1"),
            ("_pad2", "V4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("_pad3", "V5"),
            ("index", "<u4"),
            ("_pad4", "V8"),
            ("step_time_s", "<u8"),
            ("voltage_V", "<f4"),
            ("current_mA", "<f4"),
            ("_pad5", "V16"),
            ("capacity_mAh", "<f4"),
            ("energy_mWh", "<f4"),
            ("unix_time_s", "<u8"),
            ("_pad6", "V12"),
        ]
    )
    return _mask_arr(arr, data_dtype, 85).with_columns(
        [
            pl.col("unix_time_s").cast(pl.Float64) / 1e6,  # us -> s
            (pl.col("step_time_s") / 1e6).cast(pl.Float32),  # us -> s
            pl.col(["capacity_mAh", "energy_mWh"]) / 3600,
            _count_changes(pl.col("step_index")).alias("step_count"),
        ]
    )


# NDA FileVer code -> struct type reader
_NDA_READERS: dict[int, Callable[[mmap.mmap], pl.DataFrame]] = {
    1: _read_nda_1,
    2: _read_nda_2,  # Deprecated by Neware
    3: _read_nda_3,
    4: _read_nda_3,
    5: _read_nda_5,
    6: _read_nda_5,
    7: _read_nda_5,
    8: _read_nda_5,
    9: _read_nda_9,
    10: _read_nda_10,
    11: _read_nda_11,
    12: _read_nda_11,
    13: _read_nda_11,
    14: _read_nda_14,
    15: _read_nda_11,
    16: _read_nda_14,
    17: _read_nda_14,
    18: _read_nda_11,
    19: _read_nda_19,
    20: _read_nda_14,
    # 21: Missing in Neware
    22: _read_nda_14,
    23: _read_nda_14,
    24: _read_nda_14,
    25: _read_nda_25,
    26: _read_nda_29,
    27: _read_nda_25,
    28: _read_nda_29,
    29: _read_nda_29,
    129: _read_nda_129,  # Deprecated by Neware
    130: _read_nda_130,  # Variable length
}

# Reader functions confirmed against real data
_CONFIRMED_READER_NAMES = frozenset({"_read_nda_5", "_read_nda_14", "_read_nda_29", "_read_nda_130"})
_CONFIRMED_NDA_VERSIONS = frozenset(
    version for version, reader in _NDA_READERS.items() if reader.__name__ in _CONFIRMED_READER_NAMES
)
