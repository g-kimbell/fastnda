# Copyright © 2026, Empa.
"""Unit tests for fastnda.nda's low-level `_read_nda_x` struct decoders.

Uses minimal synthetic byte buffers matching each struct's known layout and
calls the reader function directly. Tests logic but does not confirm
correctness against a real vendor file - readers still XFAIL in
test_read.py::TestNdaVersionCoverage if they've never been exercised by real
data.

Several scaling factors (voltage, current, capacity/energy, time, ...) for
newly-added structs are currently best guesses, not confirmed with real data.
"""

import mmap
import struct
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import ClassVar

import polars as pl
import pytest

from fastnda import nda

# numpy dtype typestr -> struct module format char
_TYPE_STRUCT = {
    "<u1": "B",
    "<u2": "H",
    "<u4": "I",
    "<u8": "Q",
    "<i1": "b",
    "<i2": "h",
    "<i4": "i",
    "<i8": "q",
    "<f4": "f",
    "<f8": "d",
}


def _pack_record(layout: list[tuple[str, str]], values: Mapping[str, float | int]) -> bytes:
    """Pack one record's bytes from a (name, numpy-typestr) layout and a value dict.

    Args:
        layout: (name, numpy_typestr) pairs of colum name and numpy datatype.
            Padding entries (typestr like "V8") are zero-filled.
        values: Field name -> value. Any field in `layout` but missing from
            `values` defaults to zero.

    Returns:
        The packed record bytes.

    """
    chunks = []
    for name, typestr in layout:
        if typestr.startswith("V"):
            chunks.append(b"\x00" * int(typestr[1:]))
        else:
            chunks.append(struct.pack("<" + _TYPE_STRUCT[typestr], values.get(name, 0)))
    return b"".join(chunks)


def _build_rows(
    layout: list[tuple[str, str]],
    defaults: Mapping[str, float | int],
    columns: Mapping[str, Sequence[float | int]],
) -> bytes:
    """Pack many records from a shared defaults dict and column-oriented values.

    Args:
        layout: (name, numpy_typestr) pairs, see `_pack_record`.
        defaults: Field values held constant across every row.
        columns: Field name -> list of values.

    Returns:
        Concatenated bytes for every row, in order.

    """
    lengths = {len(v) for v in columns.values()}
    assert len(lengths) == 1, f"columns must all have the same length, got {[len(v) for v in columns.values()]}"
    n_rows = lengths.pop()
    chunks = []
    for i in range(n_rows):
        values = dict(defaults)
        for name, vals in columns.items():
            values[name] = vals[i]
        chunks.append(_pack_record(layout, values))
    return b"".join(chunks)


def _charge_discharge_columns(
    n_cycles: int,
    *,
    charge_current: float,
    discharge_current: float,
) -> dict[str, list[int | float]]:
    """Build charge/discharge columns.

    Helper function for testing larger synthetic datasets.

    Args:
        n_cycles: Number of charge/discharge cycles to generate (2 rows each).
        charge_current: current_mA value for every charge (odd) row.
        discharge_current: current_mA value for every discharge (even) row.

    Returns:
        Columns for index, step_index, step_type, cycle_count, current_mA.
        Other fields (voltage, capacity, ...) go in `defaults` passed to
        `_build_rows` instead.

    """
    n_rows = 2 * n_cycles
    return {
        "index": list(range(1, n_rows + 1)),
        "step_index": [i for cycle in range(n_cycles) for i in (2 * cycle + 1, 2 * cycle + 2)],
        "step_type": [1, 2] * n_cycles,
        "cycle_count": [cycle for cycle in range(n_cycles) for _ in range(2)],
        "current_mA": [charge_current, discharge_current] * n_cycles,
    }


def _make_mmap(data: bytes) -> mmap.mmap:
    """Build an anonymous memory-mapped buffer containing exactly `data`.

    Args:
        data: Bytes to write into the buffer.

    Returns:
        The memory-mapped buffer.

    """
    mm = mmap.mmap(-1, len(data))
    mm[:] = data
    return mm


def _header_offset_preamble(
    pos_offset: int,
    main_begin: int,
    *,
    main_len: int = 0,
    pos64: bool = False,
    version_byte: int | None = None,
) -> bytes:
    """Build leading header bytes for a header-offset-based reader.

    Zero-filled up through pos_offset + 2 * (8 if pos64 else 4), holding the data
    section's {begin, length} pointer at pos_offset.

    Args:
        pos_offset: Byte offset of the position-info field in the header.
        main_begin: Start offset of the data section.
        main_len: Byte length of the data section, or 0 to read to the end of the file.
        pos64: Encode as a 64-bit pair instead of 32-bit.
        version_byte: File version, written to offset 14.

    Returns:
        The header preamble bytes.

    """
    size = 8 if pos64 else 4
    buf = bytearray(pos_offset + 2 * size)
    buf[pos_offset : pos_offset + size] = main_begin.to_bytes(size, "little")
    buf[pos_offset + size : pos_offset + 2 * size] = main_len.to_bytes(size, "little")
    if version_byte is not None:
        buf[14] = version_byte
    return bytes(buf)


def _nda_1_29_header(
    main_begin: int,
    *,
    main_len: int = 0,
    current_range: int = 0,
    version_byte: int | None = None,
) -> bytes:
    """Build the header of an nda_version 1-29 files.

    Holds a {begin, length} pointer to the device info block at offsets 16/20
    and one to the main data block at 64/68. The device info block has the
    channel current range 26 bytes in.

    Args:
        main_begin: Start offset of the data section.
        main_len: Byte length of the data section, or 0 to read to the end of the file.
        current_range: Channel current range, scales current, capacity and energy.
        version_byte: File version, written to offset 14.

    Returns:
        The header bytes, including the device info block.

    """
    device_block_len = 42
    device_block_begin = 72
    buf = bytearray(device_block_begin + device_block_len)
    buf[0:6] = b"NEWARE"
    if version_byte is not None:
        buf[14] = version_byte
    buf[16:20] = device_block_begin.to_bytes(4, "little")
    buf[20:24] = device_block_len.to_bytes(4, "little")
    buf[64:68] = main_begin.to_bytes(4, "little")
    buf[68:72] = main_len.to_bytes(4, "little")
    range_at = device_block_begin + 26
    buf[range_at : range_at + 4] = current_range.to_bytes(4, "little", signed=True)
    return bytes(buf)


def _assert_col(df: pl.DataFrame, col: str, expected: list, *, abs_tol: float = 1e-4) -> None:
    """Assert a column's values match expected, in row order, within tolerance.

    Args:
        df: DataFrame to check.
        col: Column name to check.
        expected: Expected values, in index order.
        abs_tol: Absolute tolerance used when an expected value is a float.

    """
    actual = df.sort("index")[col].to_list()
    assert len(actual) == len(expected), f"{col}: expected {len(expected)} rows, got {len(actual)}: {actual}"
    for a, e in zip(actual, expected, strict=True):
        if isinstance(e, float):
            assert a == pytest.approx(e, abs=abs_tol), f"{col}: {actual} != {expected}"
        else:
            assert a == e, f"{col}: {actual} != {expected}"


class TestReadNda1:
    """NDA file version 1."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("index", "<u4"),
        ("cycle_count", "<u4"),
        ("step_index", "<u1"),
        ("step_type", "<u1"),
        ("step_time_s", "<u4"),
        ("voltage_V", "<i4"),
        ("current_mA", "<i4"),
        ("_pad1", "V8"),
        ("capacity_mAh", "<i8"),
    ]
    DEFAULTS: ClassVar[dict[str, int]] = {}
    COLUMNS: ClassVar[dict[str, list[int]]] = {
        "index": [1, 2],
        "cycle_count": [0, 1],
        "step_index": [1, 2],
        "step_type": [1, 4],
        "step_time_s": [10, 20],
        "voltage_V": [36000, 35000],
        "current_mA": [200000, -150000],
        "capacity_mAh": [1800000, 900000],
    }

    def test_decodes_expected_values(self) -> None:
        """Pack synthetic records and check every decoded column against hand-computed values."""
        data = _build_rows(self.LAYOUT, self.DEFAULTS, columns=self.COLUMNS)
        header = _nda_1_29_header(main_begin=114, current_range=6000)
        mm = _make_mmap(header + data)

        df = nda._read_nda_1(mm)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "step_index", [1, 2])
        # Raw step_type 4 remapped to 2 (CC_DChg) - nda1 uses a legacy step type enum
        _assert_col(df, "step_type", [1, 2])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        # current_range 6000 gives a 0.1 mA multiplier
        _assert_col(df, "current_mA", [20000.0, -15000.0])
        _assert_col(df, "capacity_mAh", [50.0, -25.0])
        _assert_col(df, "step_count", [1, 2])
        _assert_col(df, "cycle_count", [1, 2])

    def test_current_range_scales_current_and_capacity(self) -> None:
        """A different header current range rescales current and capacity by the same factor."""
        data = _build_rows(self.LAYOUT, self.DEFAULTS, columns=self.COLUMNS)
        header = _nda_1_29_header(main_begin=114, current_range=1)
        mm = _make_mmap(header + data)

        df = nda._read_nda_1(mm)

        # current_range 1 gives a 1e-4 mA multiplier
        _assert_col(df, "current_mA", [20.0, -15.0])
        _assert_col(df, "capacity_mAh", [0.05, -0.025])

    def test_stops_at_data_section_length(self) -> None:
        """Trailing bytes past the data section length are not decoded as records."""
        data = _build_rows(self.LAYOUT, self.DEFAULTS, columns=self.COLUMNS)
        trailing = bytes([0xFF]) * 38
        header = _nda_1_29_header(main_begin=114, main_len=len(data), current_range=6000)
        mm = _make_mmap(header + data + trailing)

        df = nda._read_nda_1(mm)

        _assert_col(df, "index", [1, 2])

    def test_either_loop_counter_starts_a_cycle(self) -> None:
        """A change in either 16-bit half of the loop counter field increments the cycle."""
        columns = {
            **self.COLUMNS,
            "index": [1, 2, 3, 4],
            # Low half changes, then high half, then neither
            "cycle_count": [0, 1, 1 + (1 << 16), 1 + (1 << 16)],
            "step_index": [1, 2, 3, 3],
            "step_type": [1, 4, 1, 1],
            "step_time_s": [10, 20, 30, 40],
            "voltage_V": [36000, 35000, 36000, 36000],
            "current_mA": [200000, -150000, 200000, 200000],
            "capacity_mAh": [1800000, 900000, 1800000, 1800000],
        }
        data = _build_rows(self.LAYOUT, self.DEFAULTS, columns=columns)
        header = _nda_1_29_header(main_begin=114, current_range=6000)
        mm = _make_mmap(header + data)

        df = nda._read_nda_1(mm)

        _assert_col(df, "cycle_count", [1, 2, 3, 3])


class TestReadNda2:
    """NDA file version 2."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("identifier", "<u1"),
        ("index", "<u4"),
        ("cycle_count", "<u4"),
        ("step_index", "<u1"),
        ("step_type", "<u1"),
        ("step_time_s", "<u4"),
        ("voltage_V", "<i4"),
        ("current_mA", "<i4"),
        ("_pad1", "V8"),
        ("capacity_mAh", "<i8"),
        ("_pad2", "V1"),
        ("energy_mWh", "<i8"),
        ("_pad3", "V1"),
        ("unix_time_s", "<u8"),
    ]
    DEFAULTS: ClassVar[dict[str, int]] = {"cycle_count": 0}

    def test_decodes_expected_values(self) -> None:
        """Pack synthetic records and check every decoded column against hand-computed values."""
        data = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
            columns={
                "identifier": [85, 0],  # exercise the "0 or 85" mask
                "index": [1, 2],
                "step_index": [1, 2],
                "step_type": [1, 2],
                "step_time_s": [10, 20],
                "voltage_V": [36000, 35000],
                "current_mA": [200000, -150000],
                "capacity_mAh": [1800000, 900000],
                "energy_mWh": [3600000, 1800000],
                "unix_time_s": [1700000000, 1700000010],
            },
        )
        header = _nda_1_29_header(main_begin=114, current_range=10)
        mm = _make_mmap(header + data)

        df = nda._read_nda_2(mm)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "cycle_count", [1, 1])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])
        _assert_col(df, "capacity_mAh", [0.5, -0.25])
        _assert_col(df, "energy_mWh", [1.0, -0.5])
        _assert_col(df, "unix_time_s", [1700000000, 1700000010])
        _assert_col(df, "step_count", [1, 2])


class TestReadNda3:
    """NDA file version 3, 4."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("identifier", "<u1"),
        ("index", "<u4"),
        ("cycle_count", "<u4"),
        ("step_index", "<u1"),
        ("step_type", "<u1"),
        ("step_time_s", "<u4"),
        ("voltage_V", "<i4"),
        ("current_mA", "<i4"),
        ("_pad1", "V8"),
        ("capacity_mAh", "<i8"),
        ("_pad2", "V4"),
    ]
    DEFAULTS: ClassVar[dict[str, int]] = {"cycle_count": 0}

    def test_decodes_expected_values(self) -> None:
        """Pack synthetic records and check every decoded column against hand-computed values."""
        data = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
            columns={
                "identifier": [85, 0],
                "index": [1, 2],
                "step_index": [1, 2],
                "step_type": [1, 2],
                "step_time_s": [10, 20],
                "voltage_V": [36000, 35000],
                "current_mA": [200000, -150000],
                "capacity_mAh": [1800000, 900000],
            },
        )
        header = _nda_1_29_header(main_begin=114, current_range=10)
        mm = _make_mmap(header + data)

        df = nda._read_nda_3(mm)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "cycle_count", [1, 1])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])
        _assert_col(df, "capacity_mAh", [0.5, -0.25])
        _assert_col(df, "step_count", [1, 2])


class TestReadNda5:
    """NDA file version 5-8."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("identifier", "<u1"),
        ("index", "<u4"),
        ("cycle_count", "<u4"),
        ("step_index", "<u1"),
        ("step_type", "<u1"),
        ("step_time_s", "<u4"),
        ("voltage_V", "<i4"),
        ("current_mA", "<i4"),
        ("_pad2", "V8"),
        ("capacity_mAh", "<i8"),
        ("energy_mWh", "<i8"),
        ("unix_time_s", "<u8"),
        ("_pad3", "V4"),
    ]
    # cycle_count is 1-based in raw data
    DEFAULTS: ClassVar[dict[str, int]] = {"identifier": 0, "cycle_count": 1}
    SENTINEL: ClassVar[bytes] = b"\xff\x01\x00\x00\x00" + b"\x00" * (59 - 5)

    def test_decodes_expected_values(self) -> None:
        """Pack synthetic records and check every decoded column against hand-computed values."""
        # Sentinel record: identifier=255, index=1, rest arbitrary - located by
        # the magic-byte search, then filtered out since mask=0 excludes it.
        data = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
            columns={
                "index": [1, 2],
                "step_index": [1, 2],
                "step_type": [1, 2],
                "step_time_s": [10, 20],
                "voltage_V": [36000, 35000],
                "current_mA": [200000, -150000],
                "capacity_mAh": [1800000, 900000],
                "energy_mWh": [3600000, 1800000],
                "unix_time_s": [1700000000, 1700000010],
            },
        )
        mm = _make_mmap(_nda_1_29_header(main_begin=114, current_range=10) + self.SENTINEL + data)

        df = nda._read_nda_5(mm)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "cycle_count", [1, 1])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])
        _assert_col(df, "capacity_mAh", [0.5, -0.25])
        _assert_col(df, "energy_mWh", [1.0, -0.5])
        _assert_col(df, "unix_time_s", [1700000000, 1700000010])
        _assert_col(df, "step_count", [1, 2])

    def test_step_count_over_many_cycles(self) -> None:
        """Stress-test step_count's change-detection logic over a much larger row count."""
        n_cycles = 50
        columns = _charge_discharge_columns(n_cycles, charge_current=200000, discharge_current=-150000)
        # Cycle count is 1-based in nda5
        columns = {**columns, "cycle_count": [c + 1 for c in columns["cycle_count"]]}
        defaults = {
            **self.DEFAULTS,
            "voltage_V": 36000,
            "capacity_mAh": 1800000,
            "energy_mWh": 3600000,
            "step_time_s": 10,
        }
        data = _build_rows(self.LAYOUT, defaults, columns=columns)
        mm = _make_mmap(_nda_1_29_header(main_begin=114, current_range=10) + self.SENTINEL + data)

        df = nda._read_nda_5(mm)

        n_rows = 2 * n_cycles
        assert len(df) == n_rows
        _assert_col(df, "index", list(range(1, n_rows + 1)))
        _assert_col(df, "step_count", list(range(1, n_rows + 1)))
        _assert_col(df, "cycle_count", [c // 2 + 1 for c in range(n_rows)])
        _assert_col(df, "current_mA", [200.0, -150.0] * n_cycles)


class TestReadNda9:
    """NDA file version 9."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("identifier", "<u1"),
        ("_pad0", "V1"),
        ("index", "<u4"),
        ("cycle_count", "<u4"),
        ("step_index", "<u1"),
        ("step_type", "<u1"),
        ("step_time_s", "<u4"),
        ("voltage_V", "<i4"),
        ("current_mA", "<i4"),
        ("_pad1", "V8"),
        ("capacity_mAh", "<i8"),
        ("energy_mWh", "<i8"),
        ("unix_time_s", "<u8"),
        ("_pad2", "V4"),
    ]
    DEFAULTS: ClassVar[dict[str, int]] = {"identifier": 85, "cycle_count": 0}

    def test_decodes_expected_values(self) -> None:
        """Pack synthetic records and check every decoded column against hand-computed values."""
        data = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
            columns={
                "index": [1, 2],
                "step_index": [1, 2],
                "step_type": [1, 2],
                "step_time_s": [10, 20],
                "voltage_V": [36000, 35000],
                "current_mA": [200000, -150000],
                "capacity_mAh": [1800000, 900000],
                "energy_mWh": [3600000, 1800000],
                "unix_time_s": [1700000000, 1700000010],
            },
        )
        header = _nda_1_29_header(main_begin=114, current_range=10)
        mm = _make_mmap(header + data)

        df = nda._read_nda_9(mm)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])
        _assert_col(df, "capacity_mAh", [0.5, -0.25])
        _assert_col(df, "energy_mWh", [1.0, -0.5])
        _assert_col(df, "unix_time_s", [1700000000, 1700000010])
        _assert_col(df, "step_count", [1, 2])


class TestReadNda10:
    """NDA file version 10."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("identifier", "<u1"),
        ("_pad0", "V1"),
        ("index", "<u4"),
        ("cycle_count", "<u4"),
        ("step_index", "<u1"),
        ("step_type", "<u1"),
        ("step_time_s", "<u8"),
        ("voltage_V", "<i4"),
        ("current_mA", "<i4"),
        ("_pad1", "V8"),
        ("capacity_mAh", "<i8"),
        ("energy_mWh", "<i8"),
        ("unix_time_s", "<u8"),
        ("_pad2", "V4"),
    ]
    DEFAULTS: ClassVar[dict[str, int]] = {"identifier": 85, "cycle_count": 0}

    def test_decodes_expected_values(self) -> None:
        """Pack synthetic records and check every decoded column against hand-computed values."""
        data = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
            columns={
                "index": [1, 2],
                "step_index": [1, 2],
                "step_type": [1, 2],
                "step_time_s": [10000, 20000],
                "voltage_V": [36000, 35000],
                "current_mA": [200000, -150000],
                "capacity_mAh": [1800000, 900000],
                "energy_mWh": [3600000, 1800000],
                "unix_time_s": [1700000000, 1700000010],
            },
        )
        header = _nda_1_29_header(main_begin=114, current_range=10)
        mm = _make_mmap(header + data)

        df = nda._read_nda_10(mm)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])
        _assert_col(df, "capacity_mAh", [0.5, -0.25])
        _assert_col(df, "energy_mWh", [1.0, -0.5])
        _assert_col(df, "unix_time_s", [1700000000, 1700000010])
        _assert_col(df, "step_count", [1, 2])


class TestReadNda11:
    """NDA file versions 11, 12, 13, 15, 18."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("identifier", "<u1"),
        ("_pad0", "V1"),
        ("index", "<u4"),
        ("cycle_count", "<u4"),
        ("step_index", "<u2"),
        ("step_type", "<u1"),
        ("step_time_s", "<u8"),
        ("voltage_V", "<i4"),
        ("current_mA", "<i4"),
        ("_pad1", "V8"),
        ("capacity_mAh", "<i8"),
        ("energy_mWh", "<i8"),
        ("unix_time_s", "<u8"),
        ("range", "<i4"),
        ("_pad2", "V4"),
    ]
    # range=100 -> multiplier 1e-2
    DEFAULTS: ClassVar[dict[str, int]] = {"identifier": 85, "cycle_count": 0, "range": 100}

    def test_decodes_expected_values(self) -> None:
        """Pack synthetic records and check every decoded column against hand-computed values."""
        data = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
            columns={
                "index": [1, 2],
                "step_index": [1, 2],
                "step_type": [1, 2],
                "step_time_s": [10000, 20000],
                "voltage_V": [36000, 35000],
                "current_mA": [20000, -15000],
                "capacity_mAh": [180000, 90000],
                "energy_mWh": [360000, 180000],
                "unix_time_s": [1700000000, 1700000010],
            },
        )
        header = _header_offset_preamble(pos_offset=64, main_begin=72, version_byte=11)
        mm = _make_mmap(header + data)

        df = nda._read_nda_11(mm)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])
        _assert_col(df, "capacity_mAh", [0.5, -0.25])
        _assert_col(df, "energy_mWh", [1.0, -0.5])
        _assert_col(df, "unix_time_s", [1700000000, 1700000010])
        _assert_col(df, "step_count", [1, 2])

    def test_record_range_beats_header_range(self) -> None:
        """A non-zero range in the record wins over the fixed range in the header."""
        data = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,  # range=100 -> multiplier 1e-2
            columns={"index": [1], "step_index": [1], "step_type": [1], "current_mA": [20000]},
        )
        # header range 10 -> multiplier 1e-3, which the record range should override
        mm = _make_mmap(_nda_1_29_header(main_begin=114, current_range=10) + data)

        df = nda._read_nda_11(mm)

        _assert_col(df, "current_mA", [200.0])

    def test_zero_record_range_falls_back_to_header_range(self) -> None:
        """A zero range in the record means unset, so the fixed header range applies."""
        data = _build_rows(
            self.LAYOUT,
            {**self.DEFAULTS, "range": 0},
            columns={"index": [1], "step_index": [1], "step_type": [1], "current_mA": [20000]},
        )
        # header range 10 -> multiplier 1e-3
        mm = _make_mmap(_nda_1_29_header(main_begin=114, current_range=10) + data)

        df = nda._read_nda_11(mm)

        _assert_col(df, "current_mA", [20.0])

    def test_file_version_does_not_change_header_offset(self) -> None:
        """Every file version sharing this struct reads the same header offset."""
        data = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
            columns={
                "index": [1],
                "step_index": [1],
                "step_type": [1],
                "voltage_V": [36000],
                "current_mA": [20000],
                "capacity_mAh": [180000],
            },
        )
        header = _header_offset_preamble(pos_offset=64, main_begin=72, version_byte=12)
        mm = _make_mmap(header + data)

        df = nda._read_nda_11(mm)

        _assert_col(df, "index", [1])
        _assert_col(df, "voltage_V", [3.6])

    def test_step_count_over_many_cycles(self) -> None:
        """Stress-test step_count's change-detection logic over a much larger row count."""
        n_cycles = 50
        columns = _charge_discharge_columns(n_cycles, charge_current=20000, discharge_current=-15000)
        defaults = {
            **self.DEFAULTS,
            "voltage_V": 36000,
            "capacity_mAh": 180000,
            "energy_mWh": 360000,
            "step_time_s": 10000,
        }
        header = _header_offset_preamble(pos_offset=64, main_begin=72, version_byte=11)
        data = _build_rows(self.LAYOUT, defaults, columns=columns)
        mm = _make_mmap(header + data)

        df = nda._read_nda_11(mm)

        n_rows = 2 * n_cycles
        assert len(df) == n_rows
        _assert_col(df, "index", list(range(1, n_rows + 1)))
        _assert_col(df, "step_count", list(range(1, n_rows + 1)))
        _assert_col(df, "current_mA", [200.0, -150.0] * n_cycles)


class TestReadNda14:
    """NDA file versions 14, 16, 17, 20, 22, 23, 24."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
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
    DEFAULTS: ClassVar[dict[str, int]] = {"identifier": 85, "cycle_count": 0, "range": 100}
    SENTINEL: ClassVar[bytes] = b"\xaa\x00\x01\x00\x00\x00" + b"\x00" * (86 - 6)

    def test_decodes_expected_values(self) -> None:
        """Pack synthetic records and check every decoded column against hand-computed values."""
        # Sentinel: identifier=170, index=1, rest arbitrary - filtered out (mask=85).
        data = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
            columns={
                "index": [1, 2],
                "step_index": [1, 2],
                "step_type": [1, 2],
                "step_time_s": [10000, 20000],
                "voltage_V": [36000, 35000],
                "current_mA": [20000, -15000],
                "charge_capacity_mAh": [180000, 0],
                "discharge_capacity_mAh": [0, 90000],
                "charge_energy_mWh": [360000, 0],
                "discharge_energy_mWh": [0, 180000],
                "unix_time_s": [1700000000, 1700000010],
            },
        )
        mm = _make_mmap(self.SENTINEL + data)

        df = nda._read_nda_14(mm)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])
        _assert_col(df, "charge_capacity_mAh", [0.5, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 0.25])
        _assert_col(df, "charge_energy_mWh", [1.0, 0.0])
        _assert_col(df, "discharge_energy_mWh", [0.0, 0.5])
        _assert_col(df, "unix_time_s", [1700000000, 1700000010])
        _assert_col(df, "step_count", [1, 2])


class TestReadNda19:
    """NDA file version 19."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("identifier", "<u1"),
        ("_pad0", "V3"),
        ("_pad0b", "V4"),
        ("index", "<u4"),
        ("cycle_count", "<u4"),
        ("step_index", "<u2"),
        ("step_type", "<u1"),
        ("_pad1", "V1"),
        ("_pad2", "V1"),
        ("_pad3", "V3"),
        ("step_time_s", "<u4"),
        ("voltage_V", "<i4"),
        ("current_mA", "<i4"),
        ("_pad4", "V8"),
        ("charge_capacity_mAh", "<i4"),
        ("discharge_capacity_mAh", "<i4"),
        ("charge_energy_mWh", "<i4"),
        ("discharge_energy_mWh", "<i4"),
        ("unix_time_s", "<u4"),
        ("_pad5", "V4"),
    ]
    DEFAULTS: ClassVar[dict[str, int]] = {"identifier": 85, "cycle_count": 0}

    def test_decodes_expected_values(self) -> None:
        """Pack synthetic records and check every decoded column against hand-computed values."""
        data = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
            columns={
                "index": [1, 2],
                "step_index": [1, 2],
                "step_type": [1, 2],
                "step_time_s": [10, 20],
                "voltage_V": [36000, 35000],
                "current_mA": [200000, -150000],
                "charge_capacity_mAh": [1800000, 0],
                "discharge_capacity_mAh": [0, 900000],
                "charge_energy_mWh": [3600000, 0],
                "discharge_energy_mWh": [0, 1800000],
                "unix_time_s": [1700000000, 1700000010],
            },
        )
        header = _nda_1_29_header(main_begin=114, current_range=10)
        mm = _make_mmap(header + data)

        df = nda._read_nda_19(mm)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])
        _assert_col(df, "charge_capacity_mAh", [0.5, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 0.25])
        _assert_col(df, "charge_energy_mWh", [1.0, 0.0])
        _assert_col(df, "discharge_energy_mWh", [0.0, 0.5])
        _assert_col(df, "unix_time_s", [1700000000, 1700000010])
        _assert_col(df, "step_count", [1, 2])


class TestReadNda25:
    """NDA file versions 25, 27."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("identifier", "<u1"),
        ("_pad0", "V1"),
        ("index", "<u4"),
        ("cycle_count", "<u4"),
        ("step_index", "<u2"),
        ("step_type", "<u1"),
        ("step_time_s", "<u8"),
        ("voltage_V", "<i4"),
        ("current_mA", "<i4"),
        ("_pad1", "V8"),
        ("capacity_mAh", "<i8"),
        ("energy_mWh", "<i8"),
        ("unix_time_s", "<u8"),
        ("range", "<i4"),
        ("_pad2", "V1"),
        ("_pad3", "V4"),
    ]
    DEFAULTS: ClassVar[dict[str, int]] = {"identifier": 85, "cycle_count": 0, "range": 100}

    def test_decodes_expected_values(self) -> None:
        """Pack synthetic records and check every decoded column against hand-computed values."""
        data = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
            columns={
                "index": [1, 2],
                "step_index": [1, 2],
                "step_type": [1, 2],
                "step_time_s": [10000, 20000],
                "voltage_V": [36000, 35000],
                "current_mA": [20000, -15000],
                "capacity_mAh": [180000, 90000],
                "energy_mWh": [360000, 180000],
                "unix_time_s": [1700000000, 1700000010],
            },
        )
        header = _header_offset_preamble(pos_offset=64, main_begin=72)
        mm = _make_mmap(header + data)

        df = nda._read_nda_25(mm)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])
        _assert_col(df, "capacity_mAh", [0.5, -0.25])
        _assert_col(df, "energy_mWh", [1.0, -0.5])
        _assert_col(df, "unix_time_s", [1700000000, 1700000010])
        _assert_col(df, "step_count", [1, 2])


class TestReadNda29:
    """NDA file versions 26, 28, 29."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
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
    DEFAULTS: ClassVar[dict[str, int]] = {
        "identifier": 85,  # doubles as the magic-byte header search target
        "cycle_count": 0,
        "range": 100,
        "Y": 2024,
        "M": 1,
        "D": 15,
        "h": 10,
        "m": 30,
    }

    def test_decodes_expected_values(self) -> None:
        """Pack synthetic records and check every decoded column against hand-computed values."""
        data = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
            columns={
                "index": [1, 2],
                "step_index": [1, 2],
                "step_type": [1, 2],
                "step_count": [1, 2],
                "step_time_s": [10000, 20000],
                "voltage_V": [36000, 35000],
                "current_mA": [20000, -15000],
                "charge_capacity_mAh": [180000, 0],
                "discharge_capacity_mAh": [0, 90000],
                "charge_energy_mWh": [360000, 0],
                "discharge_energy_mWh": [0, 180000],
                "s": [0, 10],
            },
        )
        mm = _make_mmap(data)

        df = nda._read_nda_29(mm)

        expected_unix_a = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc).timestamp()
        expected_unix_b = datetime(2024, 1, 15, 10, 30, 10, tzinfo=timezone.utc).timestamp()

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])
        _assert_col(df, "charge_capacity_mAh", [0.5, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 0.25])
        _assert_col(df, "charge_energy_mWh", [1.0, 0.0])
        _assert_col(df, "discharge_energy_mWh", [0.0, 0.5])
        _assert_col(df, "unix_time_s", [expected_unix_a, expected_unix_b], abs_tol=1.0)
        _assert_col(df, "step_count", [1, 2])


class TestReadNda129:
    """NDA file version 129."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("identifier", "<u1"),
        ("_pad0", "V5"),
        ("_pad0b", "V2"),
        ("_pad0c", "V4"),
        ("index", "<u4"),
        ("_pad0d", "V4"),
        ("step_index", "<u1"),
        ("step_type", "<u1"),
        ("_pad1", "V1"),
        ("_pad2", "V1"),
        ("_pad3", "V4"),
        ("step_time_s", "<u4"),  # Time64.dwS (seconds)
        ("step_time_ns", "<u4"),  # Time64.dwNS (nanoseconds)
        ("voltage_V", "<f4"),
        ("current_mA", "<f4"),
        ("_pad5", "V8"),
        ("charge_capacity_mAh", "<f4"),
        ("charge_energy_mWh", "<f4"),
        ("discharge_capacity_mAh", "<f4"),
        ("discharge_energy_mWh", "<f4"),
        ("unix_time_s", "<u8"),
        ("_pad6", "V12"),
    ]
    DEFAULTS: ClassVar[dict[str, float]] = {}

    def test_decodes_expected_values(self) -> None:
        """Pack synthetic records and check every decoded column against hand-computed values.

        voltage/current/capacity/energy are already floats here (V, mA, mA*s,
        mW*s), unlike the integer-encoded older structs.
        """
        data = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
            columns={
                "identifier": [85, 0],  # exercise the "0 or 85" mask
                "index": [1, 2],
                "step_index": [1, 2],
                "step_type": [1, 2],
                "step_time_s": [10, 20],
                "step_time_ns": [500_000_000, 250_000_000],  # exercises Time64 dwS+dwNS combination
                "voltage_V": [3.6, 3.5],
                "current_mA": [200.0, -150.0],
                "charge_capacity_mAh": [1800.0, 0.0],
                "charge_energy_mWh": [3600.0, 0.0],
                "discharge_capacity_mAh": [0.0, 900.0],
                "discharge_energy_mWh": [0.0, 1800.0],
                "unix_time_s": [1_700_000_000_000_000, 1_700_000_010_000_000],  # microseconds
            },
        )
        header = _header_offset_preamble(pos_offset=82, main_begin=98, pos64=True)
        mm = _make_mmap(header + data)

        df = nda._read_nda_129(mm)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "step_time_s", [10.5, 20.25])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])
        _assert_col(df, "charge_capacity_mAh", [0.5, 0.0])
        _assert_col(df, "charge_energy_mWh", [1.0, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 0.25])
        _assert_col(df, "discharge_energy_mWh", [0.0, 0.5])
        _assert_col(df, "unix_time_s", [1700000000.0, 1700000010.0])
        _assert_col(df, "step_count", [1, 2])


class TestReadNda13090:
    """NDA file version 130, BTS9.0 sub-format."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
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
    DEFAULTS: ClassVar[dict[str, int]] = {"identifier": 85}
    MAGIC: ClassVar[bytes] = b"\x12\x50\x00\x07\x55\x81\x01\x06"

    def test_decodes_expected_values(self) -> None:
        """Pack synthetic records and check every decoded column against hand-computed values.

        Unlike the integer structs, sign lives directly in the raw float here
        (no separate current-sign multiplication downstream).
        """
        data = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
            columns={
                "index": [1, 2],
                "step_index": [1, 2],
                "step_type": [1, 2],
                "step_time_s": [10_000_000, 20_000_000],  # microseconds
                "voltage_V": [3.6, 3.5],
                "current_mA": [200.0, -150.0],
                "capacity_mAh": [1800.0, -900.0],
                "energy_mWh": [3600.0, -1800.0],
                "unix_time_s": [1_700_000_000_000_000, 1_700_000_010_000_000],
            },
        )
        # Splice in the fixed magic bytes the reader searches for - the real
        # pad1/identifier/pad2 bytes it expects at the very start of the data
        # section, not reproducible via generic zero-filled padding.
        data = self.MAGIC + data[len(self.MAGIC) :]
        mm = _make_mmap(data)

        df = nda._read_nda_130_90(mm)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])
        _assert_col(df, "capacity_mAh", [0.5, -0.25])
        _assert_col(df, "energy_mWh", [1.0, -0.5])
        _assert_col(df, "unix_time_s", [1700000000.0, 1700000010.0])
        _assert_col(df, "step_count", [1, 2])


class TestReadNda13091:
    """NDA file version 130, BTS9.1 sub-format NDA."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
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
        ("_pad3", "V4"),
        ("unix_time_s", "<u4"),
        ("uts_ns", "<u4"),
    ]
    RECORD_LEN = 52  # base layout size - stays under the 56-byte aux-temp threshold
    IDENTIFIER = 7  # arbitrary 2-byte marker; other field values below avoid reproducing it
    DEFAULTS: ClassVar[dict[str, int]] = {"identifier": IDENTIFIER, "time_ns": 0, "cycle_count": 0, "uts_ns": 0}

    def test_decodes_expected_values(self) -> None:
        """Pack synthetic records and check every decoded column against hand-computed values."""
        data = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
            columns={
                "step_index": [1, 2],
                "step_type": [1, 2],
                "index": [5, 6],
                "total_time_s": [10, 20],
                "current_mA": [200.0, -150.0],
                "voltage_V": [3.6, 3.5],
                "capacity_mAs": [1800.0, -900.0],
                "energy_mWs": [3600.0, -1800.0],
                "unix_time_s": [1_700_000_000, 1_700_000_010],
            },
        )
        rec_a, rec_b = data[: self.RECORD_LEN], data[self.RECORD_LEN :]
        assert len(rec_a) == self.RECORD_LEN
        assert len(rec_b) == self.RECORD_LEN
        buf = bytearray(1024) + bytearray(rec_a) + bytearray(rec_b)
        mm = _make_mmap(bytes(buf))

        # Pre-check: the reader infers record_len by re-finding the 2-byte
        # identifier starting at offset 1026. Confirm our synthetic bytes
        # don't produce a spurious earlier match before trusting the result.
        expected_second_record_pos = 1024 + self.RECORD_LEN
        assert mm.find(mm[1024:1026], 1026) == expected_second_record_pos

        df = nda._read_nda_130_91(mm)

        _assert_col(df, "index", [5, 6])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])
        _assert_col(df, "charge_capacity_mAh", [0.5, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 0.25])
        _assert_col(df, "charge_energy_mWh", [1.0, 0.0])
        _assert_col(df, "discharge_energy_mWh", [0.0, 0.5])
        _assert_col(df, "unix_time_s", [1700000000.0, 1700000010.0])
        _assert_col(df, "total_time_s", [10.0, 20.0])
        _assert_col(df, "step_count", [1, 2])


class TestUnverifiedFormatWarning:
    """UnverifiedFormatWarning fires for unconfirmed nda_versions, not for confirmed ones.

    Builds full "NEWARE"-prefixed buffers and reads them through nda._read_nda, where the
    warning fires.
    """

    def test_warns_for_unverified_version(self) -> None:
        """nda_version 1 (no real data) emits UnverifiedFormatWarning."""
        header = _nda_1_29_header(main_begin=114, current_range=6000, version_byte=1)
        record = _build_rows(
            TestReadNda1.LAYOUT,
            TestReadNda1.DEFAULTS,
            columns={
                "index": [1],
                "cycle_count": [0],
                "step_index": [1],
                "step_type": [1],
                "step_time_s": [10],
                "voltage_V": [36000],
                "current_mA": [200000],
                "capacity_mAh": [1800000],
            },
        )
        mm = _make_mmap(header + record)

        with pytest.warns(nda.UnverifiedFormatWarning, match="nda_version 1 "):
            df = nda._read_nda(mm)

        assert len(df) == 1

    def test_no_warning_for_confirmed_version(self) -> None:
        """nda_version 8 (has real data) emits no UnverifiedFormatWarning."""
        header = bytearray(15)
        header[0:6] = b"NEWARE"
        header[14] = 8  # nda_version
        record = _build_rows(
            TestReadNda5.LAYOUT,
            TestReadNda5.DEFAULTS,
            columns={
                "index": [1],
                "step_index": [1],
                "step_type": [1],
                "step_time_s": [10],
                "voltage_V": [36000],
                "current_mA": [200000],
                "capacity_mAh": [1800000],
                "energy_mWh": [3600000],
                "unix_time_s": [1700000000],
            },
        )
        mm = _make_mmap(bytes(header) + TestReadNda5.SENTINEL + record)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = nda._read_nda(mm)

        assert not any(issubclass(w.category, nda.UnverifiedFormatWarning) for w in caught)
        assert len(df) == 1
