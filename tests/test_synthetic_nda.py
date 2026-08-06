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
    pos64: bool = False,
    version_byte: int | None = None,
) -> bytes:
    """Build leading header bytes for a header-offset-based reader.

    Zero-filled up through pos_offset + (8 if pos64 else 4), with `main_begin`
    encoded as a little-endian uint32/uint64 at pos_offset.

    Args:
        pos_offset: Byte offset of the position-info field in the header.
        main_begin: Value to encode there - the data section's start offset.
        pos64: Encode as a 64-bit pair instead of 32-bit.
        version_byte: If given, placed at offset 14, since _read_nda_11 reads
            mm[14] directly to pick which header layout is in play.

    Returns:
        The header preamble bytes.

    """
    size = 8 if pos64 else 4
    buf = bytearray(pos_offset + size)
    buf[pos_offset : pos_offset + size] = main_begin.to_bytes(size, "little")
    if version_byte is not None:
        buf[14] = version_byte
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
    """NdaData1 (file version 1): no identifier byte, header offset 32."""

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
    DEFAULTS: ClassVar[dict[str, int]] = {"cycle_count": 0}

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
            },
        )
        header = _header_offset_preamble(pos_offset=32, main_begin=36)
        mm = _make_mmap(header + data)

        df = nda._read_nda_1(mm)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "cycle_count", [1, 1])
        _assert_col(df, "step_index", [1, 2])
        _assert_col(df, "step_type", [1, 2])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])
        _assert_col(df, "capacity_mAh", [0.5, -0.25])
        _assert_col(df, "step_count", [1, 2])


class TestReadNda2:
    """NdaData2 (file version 2, deprecated by Neware): identifier in {0, 85}, header offset 32."""

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
        header = _header_offset_preamble(pos_offset=32, main_begin=36)
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
    """NdaData3 (file versions 3, 4): identifier in {0, 85}, header offset 32."""

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
        header = _header_offset_preamble(pos_offset=32, main_begin=36)
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
    """NdaData5 (file versions 5-8): magic-byte header search, mask 0."""

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
        mm = _make_mmap(self.SENTINEL + data)

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
        mm = _make_mmap(self.SENTINEL + data)

        df = nda._read_nda_5(mm)

        n_rows = 2 * n_cycles
        assert len(df) == n_rows
        _assert_col(df, "index", list(range(1, n_rows + 1)))
        _assert_col(df, "step_count", list(range(1, n_rows + 1)))
        _assert_col(df, "cycle_count", [c // 2 + 1 for c in range(n_rows)])
        _assert_col(df, "current_mA", [200.0, -150.0] * n_cycles)


class TestReadNda9:
    """NdaData9 (file version 9): identifier must be exactly 85, header offset 32."""

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
        header = _header_offset_preamble(pos_offset=32, main_begin=36)
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
    """NdaData10 (file version 10): step_time_s is u8 in ms (/1000), header offset 32."""

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
        header = _header_offset_preamble(pos_offset=32, main_begin=36)
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
    """NdaData11 (file versions 11, 12, 13, 15, 18): range-based multiplier, signed net capacity/energy.

    Version 11 itself uses header offset 32; versions 12/13/15/18 use offset
    64 - _read_nda_11 picks between them by reading mm[14] directly, so the
    main test exercises the version==11 path explicitly.
    """

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
        header = _header_offset_preamble(pos_offset=32, main_begin=36, version_byte=11)
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

    def test_version_12_uses_offset_64_header(self) -> None:
        """Versions sharing this struct other than 11 use the unified header at offset 64."""
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
        header = _header_offset_preamble(pos_offset=64, main_begin=68, version_byte=12)
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
        header = _header_offset_preamble(pos_offset=32, main_begin=36, version_byte=11)
        data = _build_rows(self.LAYOUT, defaults, columns=columns)
        mm = _make_mmap(header + data)

        df = nda._read_nda_11(mm)

        n_rows = 2 * n_cycles
        assert len(df) == n_rows
        _assert_col(df, "index", list(range(1, n_rows + 1)))
        _assert_col(df, "step_count", list(range(1, n_rows + 1)))
        _assert_col(df, "current_mA", [200.0, -150.0] * n_cycles)


class TestReadNda14:
    """NdaData14 (file versions 14, 16, 17, 20, 22, 23, 24): magic-byte header, split charge/discharge."""

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
    """NdaData19 (file version 19): no range field, split charge/discharge in mA*s (/3600)."""

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
                "charge_capacity_mAh": [1800, 0],
                "discharge_capacity_mAh": [0, 900],
                "charge_energy_mWh": [3600, 0],
                "discharge_energy_mWh": [0, 1800],
                "unix_time_s": [1700000000, 1700000010],
            },
        )
        header = _header_offset_preamble(pos_offset=64, main_begin=68)
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
    """NdaData25 (file versions 25, 27): range multiplier, signed net capacity/energy, header offset 32."""

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
        header = _header_offset_preamble(pos_offset=32, main_begin=36)
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
    """NdaData29 (file versions 26, 28, 29): magic-byte header (identifier=85 itself), Y/M/D/h/m/s timestamp."""

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
    """DFDATA_9021 (file version 129, deprecated by Neware): 64-bit header offset 82, direct floats."""

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
        header = _header_offset_preamble(pos_offset=82, main_begin=90, pos64=True)
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
    """NDA 130, BTS9.0 sub-format: magic-byte header search, raw floats already signed."""

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
    """NDA 130, BTS9.1 sub-format: fixed offset 1024, self-describing record length."""

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
        header = bytearray(_header_offset_preamble(pos_offset=32, main_begin=36))
        header[0:6] = b"NEWARE"
        header[14] = 1  # nda_version
        record = _build_rows(
            TestReadNda1.LAYOUT,
            TestReadNda1.DEFAULTS,
            columns={
                "index": [1],
                "step_index": [1],
                "step_type": [1],
                "step_time_s": [10],
                "voltage_V": [36000],
                "current_mA": [200000],
                "capacity_mAh": [1800000],
            },
        )
        mm = _make_mmap(bytes(header) + record)

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
