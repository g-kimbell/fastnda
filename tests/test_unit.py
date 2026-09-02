# Copyright © 2026, Empa.
"""Unit tests for functions outside of utils."""

import datetime
import mmap

import polars as pl
import pytest

import fastnda
from fastnda.nda import _merge_aux, _read_nda_130_91
from fastnda.nda_meta import (
    _read_pack_test_info_new,
    _read_pack_test_info_old,
    _read_pilogex_fields,
)


def test_lazy_imports() -> None:
    """Lazy imports resolve known names and rejects unknown ones."""
    assert fastnda.step_type_map is fastnda.dicts.step_type_map
    assert fastnda.btsda_csv_to_parquet is fastnda.btsda.btsda_csv_to_parquet
    with pytest.raises(AttributeError, match="no attribute 'foo'"):
        fastnda.foo  # noqa: B018


def test_nda_aux_merge() -> None:
    """Test nda aux df merge."""
    df = pl.DataFrame(
        {
            "index": [1, 2, 3, 4, 5],
            "voltage_V": [3.3, 3.4, 3.5, 3.6, 3.7],
            "current_mA": [1, 1, 1, 1, 1],
        }
    )
    aux_df = pl.DataFrame(
        {
            "aux": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "index": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "aux_voltage_V": [1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
            "aux_temperature_degC": [2, 2, 2, 2, 2, 4, 4, 4, 4, 4],
        }
    )
    merged_df = _merge_aux(df, aux_df)
    assert len(merged_df) == 5
    assert "aux1_voltage_V" in merged_df.columns
    assert "aux1_temperature_degC" in merged_df.columns
    assert "aux2_voltage_V" in merged_df.columns
    assert "aux2_temperature_degC" in merged_df.columns
    assert merged_df["aux1_voltage_V"].to_list() == [1, 1, 1, 1, 1]
    assert merged_df["aux2_voltage_V"].to_list() == [3, 3, 3, 3, 3]
    assert merged_df["aux1_temperature_degC"].to_list() == [2, 2, 2, 2, 2]
    assert merged_df["aux2_temperature_degC"].to_list() == [4, 4, 4, 4, 4]
    assert "aux" not in merged_df.columns
    assert merged_df["index"].to_list() == [1, 2, 3, 4, 5]


def test_nda_aux_merge_no_aux_col() -> None:
    """Test nda aux df merge."""
    df = pl.DataFrame(
        {
            "index": [1, 2, 3, 4, 5],
            "voltage_V": [3.3, 3.4, 3.5, 3.6, 3.7],
            "current_mA": [1, 1, 1, 1, 1],
        }
    )
    aux_df = pl.DataFrame(
        {
            "index": [1, 2, 3, 4, 5],
            "aux_voltage_V": [1, 1, 1, 1, 1],
            "aux_temperature_degC": [2, 2, 2, 2, 2],
        }
    )
    merged_df = _merge_aux(df, aux_df)
    assert len(merged_df) == 5
    assert "aux_voltage_V" in merged_df.columns
    assert "aux_temperature_degC" in merged_df.columns
    assert merged_df["aux_voltage_V"].to_list() == [1, 1, 1, 1, 1]
    assert merged_df["aux_temperature_degC"].to_list() == [2, 2, 2, 2, 2]
    assert "aux" not in merged_df.columns
    assert merged_df["index"].to_list() == [1, 2, 3, 4, 5]


def test_read_nda_130_91_odd_size() -> None:
    """Test reading nda 130-91 with different record lengths (52,56,60)."""
    _header = bytearray.fromhex(
        (
            "4e455741524532303234303430338200130010b70500000000007b010000000000008bb8050000000000e1010000000000006cba05"
        ).ljust(2048, "0")
    )
    # {begin, length} pointer to the data block, length 0 reads to the end of the file
    _header[82:90] = (1024).to_bytes(8, "little")
    header = _header.hex()
    records = [
        "550601042b0000000100000000000000809698000000000090727840000000000000000000000000000000002d3e54668006d139",
        "550601042b000000020000000c00000000497f0f00000000d4717840000000000000000000000000000000003a3e54668085b50d",
        "550601042b000000030000000c00000080df1710000000003b76784000000000000000000000000000000000f53f54668033023b",
        "550601042b000000040000004800000080df1710000000004d71784000000000000000000000000000000000314054668033023b",
        "550601042b000000050000008400000080df1710000000008b717840000000000000000000000000000000006d4054668033023b",
    ]
    footer = "550601042b000000"

    def _read_from_hex(data_hex: str) -> pl.DataFrame:
        data = bytes.fromhex(header + data_hex + footer)
        mm = mmap.mmap(-1, len(data))
        mm.write(data)
        mm.seek(0)
        return _read_nda_130_91(mm)

    df = _read_from_hex("".join(records))
    assert len(df) == 5
    assert "temperature_degC" not in df.columns

    df = _read_from_hex("".join(r + "5040b841" for r in records))
    assert len(df) == 5
    assert "aux_temperature_degC" in df.columns

    df = _read_from_hex("".join(r + "5040b841aaaaaaaa" for r in records))
    assert len(df) == 5
    assert "aux_temperature_degC" in df.columns


def _pstr(text: str) -> bytes:
    """Encode one Pascal-style string, as the tail of a swjVer < 8 record stores them."""
    raw = text.encode("gb2312")
    return bytes([len(raw)]) + raw


def _micros(iso: str) -> bytes:
    """Encode an ISO timestamp as the little-endian microsecond epoch the records use."""
    stamp = datetime.datetime.fromisoformat(iso).replace(tzinfo=datetime.timezone.utc)
    return int(stamp.timestamp() * 1_000_000).to_bytes(8, "little")


def _old_record(*, blocks: int, counted: bool, swj_ver: int = 2, mask: int | None = None) -> bytes:
    """Build a swjVer < 8 pack test info record with a given number of 128-byte text blocks."""
    if mask is None:
        mask = sum(1 << bit for bit in range(2, 2 + blocks)) | (1 << 30 if counted else 0)
    body = bytearray(bytes(4) + mask.to_bytes(4, "little") + b"\x0c\x00\x00\x00" + bytes(128 * blocks))
    if blocks:
        body[12:17] = b"admin"
    body += (61).to_bytes(4, "little")
    body += _micros("2016-04-19T12:21:01.022") + _micros("2016-04-19T13:21:03.143")
    body += (15567).to_bytes(4, "little") if counted else b"\x00"
    body += _pstr("9.0.3.16616.20160408.R5") + _pstr("38D3") + _pstr("38D3") + _pstr("192.168.3.98")
    body += _pstr("label")
    body += bytes(4) if counted else b""
    body += _pstr("label") + bytes(4) + _pstr("192.168.3.110") + bytes(3)
    return swj_ver.to_bytes(2, "little") + len(body).to_bytes(2, "little") + bytes(body[4:])


def _new_record(*, extra_block: bool, swj_ver: int = 19) -> bytes:
    """Build a swjVer >= 8 pack test info record, optionally with the extra 128-byte block."""
    body = bytearray(bytes(424))
    body[4:8] = (1).to_bytes(4, "little")
    body[8:13] = b"admin"
    body[360:369] = b"127.0.0.1"
    if extra_block:
        body += bytes(128)
    body += (26).to_bytes(4, "little") + (17587).to_bytes(4, "little")
    body += _micros("2025-12-22T08:44:24.914") + _micros("2025-12-24T09:33:13.562")
    body += bytes(12) + bytes(21)
    return swj_ver.to_bytes(2, "little") + len(body).to_bytes(2, "little") + bytes(body[4:])


@pytest.mark.parametrize("blocks", [0, 1, 2, 3])
def test_pack_test_info_old_block_counts(blocks: int) -> None:
    """The tail is found whatever number of 128-byte text blocks precedes it."""
    metadata = _read_pack_test_info_old(_old_record(blocks=blocks, counted=True))
    assert metadata["test_id"] == 61
    assert metadata["num_datapoints"] == 15567
    assert metadata["start_time"] == "2016-04-19T12:21:01.022+00:00"
    assert metadata["stop_time"] == "2016-04-19T13:21:03.143+00:00"
    assert metadata["device_ip"] == "192.168.3.98"
    assert metadata["server_ip"] == "192.168.3.110"
    # The lowest set mask bit owns the first block, so creator is absent when its bit is clear
    assert metadata.get("creator", "") == ("admin" if blocks else "")


def test_pack_test_info_old_block_belongs_to_bit_not_slot() -> None:
    """A block belongs to its mask bit, so a cleared low bit shifts the names along."""
    # Bits 3 and 4 set: the block at offset 12 is sn, not creator
    record = _old_record(blocks=2, counted=True, mask=(1 << 3) | (1 << 4) | (1 << 30))
    metadata = _read_pack_test_info_old(record)
    assert "creator" not in metadata
    assert metadata["sn"] == "admin"
    assert metadata["num_datapoints"] == 15567


def test_pack_test_info_old_mask_disagrees_with_record() -> None:
    """A mask that does not describe the record falls back to locating the tail by content."""
    record = _old_record(blocks=2, counted=True, mask=0)
    metadata = _read_pack_test_info_old(record)
    assert metadata["test_id"] == 61
    assert metadata["num_datapoints"] == 15567
    assert metadata["server_ip"] == "192.168.3.110"


def test_pack_test_info_old_without_count() -> None:
    """Builds that omit num_datapoints drop the key rather than reading the next field."""
    metadata = _read_pack_test_info_old(_old_record(blocks=2, counted=False))
    assert "num_datapoints" not in metadata
    assert metadata["test_id"] == 61
    assert metadata["bts_version"] == "9.0.3.16616.20160408.R5"
    assert metadata["server_ip"] == "192.168.3.110"


@pytest.mark.parametrize("extra_block", [False, True])
def test_pack_test_info_new_block_counts(*, extra_block: bool) -> None:
    """The newer layout is read at both observed tail offsets."""
    metadata = _read_pack_test_info_new(_new_record(extra_block=extra_block))
    assert metadata["test_id"] == 26
    assert metadata["num_datapoints"] == 17587
    assert metadata["start_time"] == "2025-12-22T08:44:24.914+00:00"
    assert metadata["creator"] == "admin"
    assert metadata["server_ip"] == "127.0.0.1"


def test_pilogex_field_offset_varies() -> None:
    """The device IP and host name are found wherever they sit in the piLogEx record."""
    for offset in (220, 223, 234):
        block = bytearray(bytes(offset) + b"192.168.1.33" + bytes(20) + b"Dell-33" + bytes(25))
        assert _read_pilogex_fields(bytes(block)) == {
            "device_ip": "192.168.1.33",
            "hostname": "Dell-33",
        }


def test_pilogex_no_ip() -> None:
    """A piLogEx block with no IP-shaped field yields nothing rather than a partial string."""
    assert _read_pilogex_fields(bytes(64) + b"not-an-ip" + bytes(64)) == {}
