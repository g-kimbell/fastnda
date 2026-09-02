# Copyright © 2026, Empa.
"""Read metadata from Neware .nda files."""

import datetime
import logging
import mmap
import re
import warnings
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


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
    "text128": (128, _decode_text),
    "u32": (4, _decode_u32),
    "datetime_us": (8, _decode_datetime_us),
    "hex21": (21, _decode_hex),
}


def _read_fields(
    record: bytes,
    fields: dict[str, tuple[int, str]],
    base: int = 0,
    limit: int | None = None,
) -> dict[str, str | int]:
    """Decode a set of name -> (byte offset, dtype) fields, with offsets taken from base.

    Fields reaching past limit, or past the end of the record, are skipped rather than
    read from whatever follows them.
    """
    end = len(record) if limit is None else min(limit, len(record))
    metadata: dict[str, str | int] = {}
    for name, (offset, dtype) in fields.items():
        length, decode = _FIELD_DECODERS[dtype]
        start = base + offset
        if start < 0 or start + length > end:
            continue
        value = decode(record[start : start + length])
        if value is not None:
            metadata[name] = value
    return metadata


# BTS8 (nda versions 1-29)

# Active mass in ug is stored at a fixed offset in the 2048-byte NDA header
# It is separate to the test info records, which have relative positions
# Tuple is (minimum version, byte offset)
_ACTIVE_MASS_OFFSETS: list[tuple[int, int]] = [
    (129, 330),
    (9, 152),
    (8, 144),
    (1, 80),
]


def _nda_active_mass_mg(mm: mmap.mmap, nda_version: int) -> float:
    """Read active material mass (mg) from the NDA header, for nda_version 1-29 or 129/130."""
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
    version_loc = mm.find(b"BTSServer", 0, 65536)
    if version_loc != -1:
        # Two 50-byte fields, separated by 50 unused bytes
        return {
            "server_version": mm[version_loc : version_loc + 50].strip(b"\x00").decode(),
            "client_version": mm[version_loc + 100 : version_loc + 150].strip(b"\x00").decode(),
        }
    xwj = mm.find(b"BTS_XWJ", 0, 1024)
    if xwj == -1:
        logger.info("BTS version not found!")
        return {}
    end = mm.find(b"\x00", xwj, 1024)
    return {"server_version": mm[xwj:end].decode().strip()} if end != -1 else {}


# BTS9 (nda versions 129, 130)
# Reads from 'pack test info' record
#
# The record is a versioned struct: u16 swjVer, u16 record length, then either a field
# presence mask (swjVer < 8) or a fixed schema (swjVer >= 8).
# Seems to always end with test_id, timestamps and usually num_datapoints
# Older files then have a chain of Pascal strings

# swjVer at or above this uses the newer record layout
_SWJ_VER_NEW_LAYOUT = 8

# Tail fields, offsets relative to the start of the tail.
# The count sits after the timestamps in the old layout and before them in the new one.
_TAIL_OLD: dict[str, tuple[int, str]] = {
    "test_id": (0, "u32"),
    "start_time": (4, "datetime_us"),
    "stop_time": (12, "datetime_us"),
    "num_datapoints": (20, "u32"),  # dropped again if the mask says it is absent
}
_TAIL_NEW: dict[str, tuple[int, str]] = {
    "test_id": (0, "u32"),  # Not very confident
    "num_datapoints": (4, "u32"),
    "start_time": (8, "datetime_us"),
    "stop_time": (16, "datetime_us"),
    "UNKNOWN_15": (24, "u32"),  # zero in every sample seen
    "UNKNOWN_16": (28, "u32"),  # decodes to a nonsense 2010-era date, not a real timestamp
    "UNKNOWN_17": (32, "u32"),  # constant 1 in every sample seen
    "UNKNOWN_18": (36, "hex21"),  # rest of the record, resembles a pattern also seen in piLogEx
}
_OLD_TIMES_AT = _TAIL_OLD["start_time"][0]
_NEW_TIMES_AT = _TAIL_NEW["start_time"][0]

# test_id plus the two timestamps, then the count or a single filler byte
_OLD_TAIL_LEN = 20
_COUNT_LEN = 4
_NO_COUNT_LEN = 1

# swjVer < 8 records start with the two fields that describe the rest of the record
_HEADER_OLD: dict[str, tuple[int, str]] = {
    "UNKNOWN_19": (4, "u32"),  # the field presence mask
    "UNKNOWN_5": (8, "u32"),
}

# Each set mask bit adds one equal-sized text block, packed in bit order from _BLOCKS_AT.
# Names past creator are a best guess: only bit 2 has non-empty values in enough files.
_BLOCK_LEN = 128
_BLOCKS_AT = 12
_BLOCK_BITS: dict[int, str] = {
    2: "creator",
    3: "sn",
    4: "remarks",
    5: "UNKNOWN_20",
}
# Set when the tail carries num_datapoints, clear on the 2013-era builds that omit it
_COUNT_BIT = 30

# swjVer >= 8 has no mask, so its fields sit at fixed offsets and the tail is per schema version
_FIELDS_NEW: dict[str, tuple[int, str]] = {
    "start_step_id": (4, "u32"),
    "creator": (8, "text32"),
    "sn": (40, "text32"),
    "UNKNOWN_1": (72, "text32"),  # desc? empty in test data
    "UNKNOWN_2": (104, "text64"),  # step_file_name? empty in test data
    "UNKNOWN_3": (168, "text64"),  # step_name? empty in test data
    "UNKNOWN_4": (232, "text32"),  # battery_model? empty in test data
    "remarks": (264, "text64"),
    "UNKNOWN_14": (328, "text32"),
    "server_ip": (360, "text64"),
}
_NEW_TAIL_AT: dict[int, int] = {19: 424, 42: 552}

# Microsecond epoch bounds used to recognise the start_time/stop_time pair in a record
_TIMESTAMP_MIN_US = int(datetime.datetime(2005, 1, 1, tzinfo=datetime.timezone.utc).timestamp() * 1e6)
_TIMESTAMP_MAX_US = int(datetime.datetime(2035, 1, 1, tzinfo=datetime.timezone.utc).timestamp() * 1e6)
_MAX_TEST_DURATION_US = 5 * 365 * 24 * 3600 * 1_000_000

# Version string like "9.1.5.7.20250527.R5", 4+ dot-groups
_BTS_VERSION_RE = re.compile(rb"\d+(?:\.[\dA-Za-z]+){4,}")


def _is_timestamp_pair(record: bytes, pos: int) -> bool:
    """Whether the two u64 microsecond timestamps at pos are a plausible (start, stop) pair."""
    start = int.from_bytes(record[pos : pos + 8], "little")
    stop = int.from_bytes(record[pos + 8 : pos + 16], "little")
    return (
        _TIMESTAMP_MIN_US < start < _TIMESTAMP_MAX_US
        and _TIMESTAMP_MIN_US < stop < _TIMESTAMP_MAX_US
        and 0 <= stop - start < _MAX_TEST_DURATION_US
    )


def _find_timestamp_pair(record: bytes) -> int | None:
    """Offset of the only plausible start_time/stop_time pair in a record, or None.

    Fallback for a record the layout rules do not describe. Requiring exactly one match
    keeps an unrecognised layout from yielding plausible-looking numbers.
    """
    hits = [pos for pos in range(len(record) - 15) if _is_timestamp_pair(record, pos)]
    if len(hits) != 1:
        logger.info("Expected 1 timestamp pair in pack test info record, found %d.", len(hits))
        return None
    return hits[0]


def _find_version_pstring(record: bytes) -> int | None:
    """Offset of the length byte of the bts_version string, which starts the trailing chain."""
    for match in _BTS_VERSION_RE.finditer(record):
        if match.start() and record[match.start() - 1] == len(match.group()):
            return match.start() - 1
    return None


def _decode_pstring(data: bytes, pos: int) -> tuple[str, int]:
    """Decode one Pascal-style string (1-byte length prefix, no terminator); return (text, next_pos)."""
    end = pos + 1 + data[pos]
    if end > len(data):
        raise IndexError(pos)
    return data[pos + 1 : end].decode("gb2312", errors="ignore"), end


def _read_pack_test_info_chain(record: bytes, pos: int, *, counted: bool) -> dict[str, str]:
    """Read the Pascal-string chain that ends a swjVer < 8 record.

    The two label strings are separated by a u32 on builds that also carry num_datapoints;
    builds without the count run the labels together instead.
    """
    try:
        bts_version, pos = _decode_pstring(record, pos)
        guid, pos = _decode_pstring(record, pos)
        guid_repeat, pos = _decode_pstring(record, pos)
        device_ip, pos = _decode_pstring(record, pos)
        label, pos = _decode_pstring(record, pos)  # "[org] - dedicated use" in examples
        pos += _COUNT_LEN if counted else 0
        label_repeat, pos = _decode_pstring(record, pos)  # equal to label in every sample
        pos += _COUNT_LEN
        server_ip, _pos = _decode_pstring(record, pos)
    except IndexError:
        logger.info("Pack test info string chain ran past the end of the record.")
        return {}
    return {
        "bts_version": bts_version,
        "guid": guid,
        "guid2": guid_repeat,
        "device_ip": device_ip,
        "server_ip": server_ip,
        "UNKNOWN_10": label,
        "UNKNOWN_13": label_repeat,
    }


def _old_blocks(mask: int) -> dict[str, int]:
    """Offset of each text block the presence mask says is present, keyed by field name."""
    blocks: dict[str, int] = {}
    offset = _BLOCKS_AT
    for bit, name in sorted(_BLOCK_BITS.items()):
        if mask >> bit & 1:
            blocks[name] = offset
            offset += _BLOCK_LEN
    return blocks


def _search_old_tail(record: bytes) -> tuple[int, bool] | None:
    """Locate the tail of a swjVer < 8 record the mask does not describe.

    Returns the tail offset and whether num_datapoints is present, taking the gap between
    the timestamps and the string chain as the deciding evidence.
    """
    pair_pos = _find_timestamp_pair(record)
    if pair_pos is None:
        return None
    tail = pair_pos - _OLD_TIMES_AT
    chain_pos = _find_version_pstring(record)
    for counted, count_len in ((True, _COUNT_LEN), (False, _NO_COUNT_LEN)):
        if chain_pos == tail + _OLD_TAIL_LEN + count_len:
            return tail, counted
    return None


def _read_pack_test_info_old(record: bytes) -> dict[str, str | int]:
    """Read a swjVer < 8 pack test info record, laid out by its field presence mask."""
    mask = int.from_bytes(record[4:8], "little")
    blocks = _old_blocks(mask)
    tail = _BLOCKS_AT + _BLOCK_LEN * len(blocks)
    counted = bool(mask >> _COUNT_BIT & 1)

    if not _is_timestamp_pair(record, tail + _OLD_TIMES_AT):
        logger.info("Pack test info mask 0x%08X does not describe this record, searching instead.", mask)
        found = _search_old_tail(record)
        if found is None:
            return {}
        blocks = {}
        tail, counted = found

    metadata = _read_fields(record, _HEADER_OLD)
    metadata.update(_read_fields(record, {name: (off, "text128") for name, off in blocks.items()}))
    metadata.update(_read_fields(record, _TAIL_OLD, base=tail))
    if not counted:
        metadata.pop("num_datapoints", None)
    chain_pos = tail + _OLD_TAIL_LEN + (_COUNT_LEN if counted else _NO_COUNT_LEN)
    metadata.update(_read_pack_test_info_chain(record, chain_pos, counted=counted))
    return metadata


def _read_pack_test_info_new(record: bytes) -> dict[str, str | int]:
    """Read a swjVer >= 8 pack test info record, whose tail offset is per schema version."""
    swj_ver = int.from_bytes(record[0:2], "little")
    tail = _NEW_TAIL_AT.get(swj_ver)
    if tail is None or not _is_timestamp_pair(record, tail + _NEW_TIMES_AT):
        logger.info("Unknown pack test info schema swjVer %d, searching for the tail.", swj_ver)
        pair_pos = _find_timestamp_pair(record)
        if pair_pos is None:
            return {}
        tail = pair_pos - _NEW_TIMES_AT
    metadata = _read_fields(record, _FIELDS_NEW, limit=tail)
    metadata.update(_read_fields(record, _TAIL_NEW, base=tail))
    return metadata


def _read_bts9_test_info(mm: mmap.mmap) -> dict[str, str | int]:
    """Read records from 'pack test info' from an nda_version 129/130 file.

    The header stores a u64 {begin, length} pointer to this block at fixed offsets 34/42.
    """
    test_begin = int.from_bytes(mm[34:42], "little")
    test_len = int.from_bytes(mm[42:50], "little")
    if not test_begin or not test_len:
        return {}
    record = mm[test_begin : test_begin + test_len]
    swj_ver = int.from_bytes(record[0:2], "little")
    if swj_ver < _SWJ_VER_NEW_LAYOUT:
        return _read_pack_test_info_old(record)
    return _read_pack_test_info_new(record)


# nda_version 129/130 'log ex' block
# piLogEx, doesn't seem to be present in the older BTS9.0.3 file.
# IP address is probably device or middle machine, server_ip is usually 127.0.0.1
_IPV4_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")
_PILOGEX_FIELD_LEN = 32


def _read_pilogex_fields(record: bytes) -> dict[str, str]:
    """Read the device IP and host name from a piLogEx block.

    Record size and the IP's position within it both move between BTS9 generations, so the
    IP is found by shape and the host name taken from the next fixed-width field.
    """
    for offset in range(max(0, len(record) - _PILOGEX_FIELD_LEN + 1)):
        # Only consider the start of a null-terminated field, so a partial IP cannot match
        if not record[offset] or (offset and record[offset - 1]):
            continue
        device_ip = _decode_text(record[offset : offset + _PILOGEX_FIELD_LEN])
        if not _IPV4_RE.fullmatch(device_ip):
            continue
        hostname_at = offset + _PILOGEX_FIELD_LEN
        return {
            "device_ip": device_ip,
            "hostname": _decode_text(record[hostname_at : hostname_at + _PILOGEX_FIELD_LEN]),
        }
    logger.info("No device IP found in piLogEx block.")
    return {}


def _read_bts9_log_ex(mm: mmap.mmap) -> dict[str, str]:
    """Read the 'log ex' block from an nda_version 129/130 file.

    The head info 9022 header stores a u64 {begin, length} pointer to this block
    at fixed offsets 242/250.
    """
    log_ex_begin = int.from_bytes(mm[242:250], "little")
    log_ex_len = int.from_bytes(mm[250:258], "little")
    if not log_ex_begin or not log_ex_len or log_ex_begin + log_ex_len > len(mm):
        return {}
    return _read_pilogex_fields(mm[log_ex_begin : log_ex_begin + log_ex_len])


_BTS9_WARNING = (
    "read_metadata for NDA 129/130 (BTS9) has not been thoroughly tested due to lack of test data. "
    "There may be wrong fields or 'UNKNOWN_x' keys present. "
    "If you can, please share a sample file at "
    "https://github.com/empaeconversion/fastnda/issues so we can improve reading this format."
)


def _read_bts9_metadata(mm: mmap.mmap) -> dict[str, str | float | int]:
    """Read metadata specific to an nda_version 129/130 (BTS9) file."""
    metadata: dict[str, str | float | int] = {}
    match = _BTS_VERSION_RE.search(mm[:2048])
    if match:
        metadata["bts_version"] = match.group().decode()
    else:
        logger.info("BTS version not found in header.")

    metadata["active_mass_mg"] = _nda_active_mass_mg(mm, 130)
    metadata.update(_read_bts9_test_info(mm))
    metadata.update(_read_bts9_log_ex(mm))
    return metadata


def _is_empty(value: str | float) -> bool:
    """Whether a metadata value is blank text or zero."""
    return not value.strip("\x00").strip() if isinstance(value, str) else not value


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

    if mm[0:6] != b"NEWARE":
        msg = f"{file} does not appear to be a Neware file."
        raise ValueError(msg)

    nda_version = int(mm[14])
    metadata: dict[str, int | str | float] = {"nda_version": nda_version}
    metadata.update(_read_nda_version_info(mm))

    # NDA 1-29 fields: header stores {nBegin, nLen} pointers to a device-info block and a test-info block
    if 1 <= nda_version <= 29:
        metadata["active_mass_mg"] = _nda_active_mass_mg(mm, nda_version)
        metadata.update(_read_nda_test_info(mm, nda_version))

    # NDA 129/130 specific fields
    elif nda_version in {129, 130}:
        warnings.warn(_BTS9_WARNING, stacklevel=3)
        metadata.update(_read_bts9_metadata(mm))

    return {k: v for k, v in metadata.items() if not (k.startswith("UNKNOWN_") and _is_empty(v))}
