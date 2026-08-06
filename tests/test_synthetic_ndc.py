# Copyright © 2026, Empa.
"""Unit tests for ndax `_read_ndc_{type}_x` parsers.

Uses minimal synthetic byte buffers matching each struct's known layout and
calls the reader function directly. Tests logic but does not confirm
correctness against a real vendor file - readers still XFAIL in
test_read.py::TestNdcVersionCoverage if they've never been exercised by real
data.

Several scaling factors (voltage, current, capacity/energy, time, ...) for
newly-added structs are currently best guesses, not confirmed with real data.
"""

import importlib
import struct
import warnings
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl
import pytest

from fastnda._ndc import ndc_aux, ndc_main, ndc_runinfo, ndc_step, read_ndc
from fastnda.ndax import read_ndax
from fastnda.utils import UnverifiedFormatWarning

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
        layout: (name, numpy_typestr) pairs of column name and numpy datatype.
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
    columns: dict[str, Sequence[float | int]],
    defaults: dict[str, int] | None = None,
) -> bytes:
    """Pack many records from a shared defaults dict and column-oriented values.

    Args:
        layout: (name, numpy_typestr) pairs, see `_pack_record`.
        columns: Field name -> list of values.
        defaults: Field values held constant across every row.

    Returns:
        Concatenated bytes for every row, in order.

    """
    lengths = {len(v) for v in columns.values()}
    assert len(lengths) == 1, f"columns must all have the same length, got {[len(v) for v in columns.values()]}"
    n_rows = lengths.pop()
    chunks = []
    defaults = defaults or {}
    for i in range(n_rows):
        values = dict(defaults)
        for name, vals in columns.items():
            values[name] = vals[i]
        chunks.append(_pack_record(layout, values))
    return b"".join(chunks)


def _assert_col(df: pl.DataFrame, col: str, expected: list, *, abs_tol: float = 1e-4) -> None:
    """Assert a column's values match expected, in row order, within tolerance.

    Args:
        df: DataFrame to check.
        col: Column name to check.
        expected: Expected values, in index order.
        abs_tol: Absolute tolerance used when an expected value is a float.

    """
    actual = (df.sort("index") if "index" in df.columns else df)[col].to_list()
    assert len(actual) == len(expected), f"{col}: expected {len(expected)} rows, got {len(actual)}: {actual}"
    for a, e in zip(actual, expected, strict=True):
        if isinstance(e, float):
            assert a == pytest.approx(e, abs=abs_tol), f"{col}: {actual} != {expected}"
        else:
            assert a == e, f"{col}: {actual} != {expected}"


def _make_ndc_file(
    dtype: np.dtype,
    row_bytes: list[bytes],
    *,
    filetype: int,
    version: int,
    data_start_ind: int = 132,
    record_size: int = 4096,
    use_bitmask: bool = True,
) -> bytes:
    """Build a minimal NDC file: one header block (declares filetype/version) + one data block.

    Args:
        dtype: Numpy dtype describing the record structure (same one the reader uses).
        row_bytes: Packed record bytes, one entry per row, in order.
        filetype: NDC filetype byte, written at header offset 0.
        version: NDC version byte, written at header offset 2.
        data_start_ind: Byte offset of the first row within the data block.
        record_size: Total size of the header block and the data block.
        use_bitmask: Whether to mark rows valid via the offset-4 bitmask.

    Returns:
        Concatenated header block + data block bytes.

    """
    header = bytearray(record_size)
    header[0] = filetype
    header[2] = version
    block = bytearray(record_size)
    if use_bitmask:
        rows_per_record = (record_size - data_start_ind - 1) // dtype.itemsize
        mask = np.zeros(rows_per_record, dtype=np.uint8)
        mask[: len(row_bytes)] = 1
        packed = np.packbits(mask, bitorder="little").tobytes()
        block[4 : 4 + len(packed)] = packed
    offset = data_start_ind
    for rec in row_bytes:
        assert len(rec) == dtype.itemsize
        block[offset : offset + len(rec)] = rec
        offset += dtype.itemsize
    return bytes(header) + bytes(block)


def _split_rows(rows: bytes, dtype: np.dtype) -> list[bytes]:
    """Split concatenated record bytes into one entry per record."""
    assert len(rows) % dtype.itemsize == 0, f"{len(rows)} bytes is not a whole number of {dtype.itemsize}-byte records"
    return [rows[i : i + dtype.itemsize] for i in range(0, len(rows), dtype.itemsize)]


class TestNdcMain:
    """Main ndc files."""

    def test_main_1(self) -> None:
        """Main for ndax 1, 3."""
        layout = [
            ("_pad1", "V8"),
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("_pad2", "V2"),
            ("step_time_s", "<u8"),
            ("voltage_V", "<u4"),
            ("current_mA", "<u4"),
            ("_pad3", "V4"),
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
            ("range", "<i4"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(
            layout,
            columns={
                "index": [1, 2],
                "step_index": [1, 2],
                "step_type": [1, 2],
                "step_time_s": [10000, 20000],
                "voltage_V": [36000, 35000],
                "current_mA": [20000, 15000],
                "charge_capacity_mAh": [180000, 90000],
                "discharge_capacity_mAh": [0, 0],
                "charge_energy_mWh": [360000, 180000],
                "discharge_energy_mWh": [0, 0],
                "s": [0, 10],
            },
            defaults={"cycle_count": 0, "range": 100, "Y": 2024, "M": 1, "D": 1},
        )
        buf = _make_ndc_file(
            dtype,
            _split_rows(rows, dtype),
            filetype=1,
            version=1,
            data_start_ind=5,
            record_size=512,
            use_bitmask=False,
        )

        df = ndc_main.read_ndc_main_1(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "cycle_count", [1, 1])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, 150.0])
        _assert_col(df, "charge_capacity_mAh", [0.5, 0.25])
        _assert_col(df, "charge_energy_mWh", [1.0, 0.5])
        _assert_col(df, "step_count", [1, 2])

    def test_main_2(self) -> None:
        """Main for ndax 2, 4."""
        layout = [
            ("_pad1", "V8"),
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("_pad2", "V5"),
            ("step_time_s", "<u8"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad3", "V4"),
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
            ("range", "<i4"),
            ("_pad4", "V8"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(
            layout,
            columns={
                "index": [1, 2],
                "step_index": [1, 2],
                "step_type": [1, 2],
                "step_time_s": [10000, 20000],
                "voltage_V": [36000, 35000],
                "current_mA": [20000, -15000],
                "charge_capacity_mAh": [180000, 90000],
                "discharge_capacity_mAh": [0, 0],
                "charge_energy_mWh": [360000, 180000],
                "discharge_energy_mWh": [0, 0],
                "s": [0, 10],
            },
            defaults={"cycle_count": 0, "range": 100, "Y": 2024, "M": 1, "D": 1},
        )
        buf = _make_ndc_file(
            dtype,
            _split_rows(rows, dtype),
            filetype=1,
            version=2,
            data_start_ind=5,
            record_size=512,
            use_bitmask=False,
        )

        df = ndc_main.read_ndc_main_2(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "cycle_count", [1, 1])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])
        _assert_col(df, "charge_capacity_mAh", [0.5, 0.25])
        _assert_col(df, "charge_energy_mWh", [1.0, 0.5])
        _assert_col(df, "step_count", [1, 2])

    def test_main_5(self) -> None:
        """Main for ndax 5."""
        layout = [
            ("_pad1", "V1"),
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("_pad2", "V5"),
            ("step_time_s", "<u8"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("_pad3", "V4"),
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
            ("range", "<i4"),
            ("_pad4", "V8"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(
            layout,
            columns={
                "index": [1, 2],
                "step_index": [1, 2],
                "step_type": [1, 2],
                "step_time_s": [10000, 20000],
                "voltage_V": [36000, 35000],
                "current_mA": [20000, -15000],
                "charge_capacity_mAh": [180000, 90000],
                "discharge_capacity_mAh": [0, 0],
                "charge_energy_mWh": [360000, 180000],
                "discharge_energy_mWh": [0, 0],
                "s": [0, 10],
            },
            defaults={"cycle_count": 0, "range": 100, "Y": 2024, "M": 1, "D": 1},
        )
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=1, version=5)

        df = ndc_main.read_ndc_main_5(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "cycle_count", [1, 1])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])
        _assert_col(df, "charge_capacity_mAh", [0.5, 0.25])
        _assert_col(df, "charge_energy_mWh", [1.0, 0.5])
        _assert_col(df, "step_count", [1, 2])

    def test_main_6(self) -> None:
        """Main for ndax 6."""
        layout = [
            ("step_time_s", "<u4"),
            ("_pad1", "V1"),
            ("voltage_V", "<f4"),
            ("current_mA", "<f4"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V1"),
            ("unix_time_s", "<u4"),
            ("step_count", "<u4"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(
            layout,
            columns={
                "step_time_s": [10000, 20000],
                "voltage_V": [3.6, 3.5],
                "current_mA": [0.2, -0.15],
                "charge_capacity_mAh": [1.8, 0.0],
                "discharge_capacity_mAh": [0.0, 0.9],
                "charge_energy_mWh": [3.6, 0.0],
                "discharge_energy_mWh": [0.0, 1.8],
                "unix_time_s": [1700000000, 1700000010],
                "step_count": [1, 2],
            },
        )
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=1, version=6)

        df = ndc_main.read_ndc_main_6(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])
        _assert_col(df, "charge_capacity_mAh", [1800.0, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 900.0])
        _assert_col(df, "unix_time_s", [1700000000, 1700000010])
        _assert_col(df, "step_count", [1, 2])

    @staticmethod
    def _make_ndax7_header(codes: list[int]) -> bytes:
        header = bytearray(4096)
        header[0] = 1  # filetype
        header[2] = 7  # version
        n_data_type = codes + [0] * (40 - len(codes))
        header[13 : 13 + 4 * 40] = struct.pack("<40i", *n_data_type)
        return bytes(header)

    def test_main_7(self) -> None:
        """Main and aux for ndax 7."""
        field_codes = [6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 29, 53]
        layout = [
            ("index", "<u4"),
            ("cycle_count", "<u4"),
            ("step_index", "<u1"),
            ("step_type", "<u1"),
            ("step_time_s", "<u8"),
            ("voltage_V", "<i4"),
            ("current_mA", "<i4"),
            ("temperature_degC", "<i2"),
            ("charge_capacity_mAh", "<i8"),
            ("discharge_capacity_mAh", "<i8"),
            ("charge_energy_mWh", "<i8"),
            ("discharge_energy_mWh", "<i8"),
            ("atime_year", "<u2"),
            ("atime_month", "<u1"),
            ("atime_day", "<u1"),
            ("atime_hour", "<u1"),
            ("atime_minute", "<u1"),
            ("atime_second", "<u1"),
            ("total_time_s", "<u8"),
            ("atime_ms", "<u2"),
        ]
        header = self._make_ndax7_header(field_codes)
        dtype = ndc_main._ndc7_schema(header)
        assert dtype == np.dtype(layout)
        rows = _build_rows(
            layout,
            columns={
                "index": [1, 2],
                "cycle_count": [0, 1],
                "step_index": [1, 2],
                "step_type": [1, 2],
                "step_time_s": [10000, 20000],
                "voltage_V": [36000, 35000],
                "current_mA": [200, -150],
                "temperature_degC": [250, 260],
                "charge_capacity_mAh": [1800, 0],
                "discharge_capacity_mAh": [0, 900],
                "charge_energy_mWh": [3600, 0],
                "discharge_energy_mWh": [0, 1800],
                "atime_hour": [3, 3],
                "atime_minute": [4, 4],
                "atime_second": [5, 6],
                "total_time_s": [20000, 40000],
                "atime_ms": [500, 0],
            },
            defaults={"atime_year": 2024, "atime_month": 1, "atime_day": 2},
        )
        gen = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=1, version=7)
        buf = header + gen[len(header) :]

        df = ndc_main.read_ndc_main_7(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "cycle_count", [1, 2])
        _assert_col(df, "step_index", [1, 2])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200, -150])
        _assert_col(df, "temperature_degC", [25.0, 26.0])
        _assert_col(df, "charge_capacity_mAh", [0.5, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 0.25])
        _assert_col(df, "charge_energy_mWh", [1.0, 0.0])
        _assert_col(df, "discharge_energy_mWh", [0.0, 0.5])
        _assert_col(df, "total_time_s", [20.0, 40.0])
        _assert_col(df, "unix_time_s", [1704164645.5, 1704164646.0])
        _assert_col(df, "step_count", [1, 2])
        for col in ("atime_year", "atime_month", "atime_day", "atime_hour", "atime_minute", "atime_second", "atime_ms"):
            assert col not in df.columns

    def test_main_7_unrecognized_field(self) -> None:
        """Abort if there is a field code with no known size."""
        header = self._make_ndax7_header([6, 999])
        with pytest.raises(NotImplementedError, match="unrecognized field type code"):
            ndc_main._ndc7_schema(header)

    def test_main_11(self) -> None:
        """Main for ndax 9, 11, 13."""
        layout = [
            ("voltage_V", "<f4"),
            ("current_mA", "<f4"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(layout, columns={"voltage_V": [36000.0, 35000.0], "current_mA": [200.0, -150.0]})
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=1, version=11)

        df = ndc_main.read_ndc_main_11(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])

    def test_main_14(self) -> None:
        """Main for ndax 8, 12, 14, 17."""
        layout = [
            ("voltage_V", "<f4"),
            ("current_mA", "<f4"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(layout, columns={"voltage_V": [3.6, 3.5], "current_mA": [0.2, -0.15]})
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=1, version=14)

        df = ndc_main.read_ndc_main_14(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])

    @staticmethod
    def _make_ndc15_header(record_itemsize: int, custom_items: list[tuple[int, int, int]]) -> bytes:
        header = bytearray(4096)
        header[0] = 1  # filetype
        header[2] = 15  # version
        header[516:520] = struct.pack("<I", record_itemsize)
        offset = 520
        for axis_type, value_type, pos in custom_items:
            header[offset : offset + 9] = struct.pack("<iBi", axis_type, value_type, pos)
            offset += 9
        return bytes(header)

    @staticmethod
    def _make_ndc15_record(
        *,
        step_index: int,
        step_type_raw: int,
        index: int,
        total_dws: int,
        total_dwns: int,
        curr: float,
        volt: float,
        fixed_cap: float,
        fixed_eng: float,
        extra: bytes = b"",
    ) -> bytes:
        prefix = struct.pack(
            "<BBBBIIIIffff",
            0,  # btDataFlag
            0,  # btReserved
            step_index,
            step_type_raw,
            0,  # dwTestID
            index,
            total_dws,
            total_dwns,
            curr,
            volt,
            fixed_cap,
            fixed_eng,
        )
        return prefix + extra

    @staticmethod
    def _make_ndc15_block(records: list[bytes], record_itemsize: int) -> bytes:
        block = bytearray(4096)
        rows_per_record = (4096 - 132 - 1) // record_itemsize
        mask = np.zeros(rows_per_record, dtype=np.uint8)
        mask[: len(records)] = 1
        packed = np.packbits(mask, bitorder="little").tobytes()
        block[4 : 4 + len(packed)] = packed
        offset = 132
        for rec in records:
            assert len(rec) == record_itemsize
            block[offset : offset + len(rec)] = rec
            offset += record_itemsize
        return bytes(block)

    def test_main_15_fixed(self) -> None:
        """No dynamic fields declared: capacity/energy come from the fixed prefix, gated by step type."""
        record_itemsize = 36
        header = self._make_ndc15_header(record_itemsize, custom_items=[])
        records = [
            self._make_ndc15_record(
                step_index=1,
                step_type_raw=21,  # BTS9 stChgPowerVolt -> Common.StepType.cpcv_chg(27), charge
                index=10,
                total_dws=5,
                total_dwns=500_000_000,
                curr=123.45,
                volt=3.6,
                fixed_cap=1800.0,
                fixed_eng=3600.0,
            ),
            self._make_ndc15_record(
                step_index=2,
                step_type_raw=22,  # BTS9 stDChgPowerVolt -> Common.StepType.cpcv_dchg(26), discharge
                index=11,
                total_dws=10,
                total_dwns=0,
                curr=-67.89,
                volt=3.5,
                fixed_cap=900.0,
                fixed_eng=1800.0,
            ),
        ]
        buf = header + self._make_ndc15_block(records, record_itemsize)

        df = ndc_main.read_ndc_main_15(buf)

        _assert_col(df, "index", [10, 11])
        _assert_col(df, "step_index", [1, 2])
        _assert_col(df, "step_type", [27, 26])
        _assert_col(df, "total_time_s", [5.5, 10.0])
        _assert_col(df, "current_mA", [123.45, -67.89])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "capacity_mAh", [0.5, -0.25])
        _assert_col(df, "energy_mWh", [1.0, -0.5])
        _assert_col(df, "step_count", [1, 2])
        for col in ("step_time_s", "unix_time_s"):
            assert col not in df.columns

    def test_main_15_dynamic(self) -> None:
        """Test ndax 15 with optional extra columns.

        Step time and unix time should be added.
        ccap/dcap, ceng/deng should take priority over fixed cap/eng.
        pow should be ignored, and shouldn't affect other column reading.
        """
        record_itemsize = 72
        header = self._make_ndc15_header(
            record_itemsize,
            custom_items=[
                (65, 6, 36),  # StepTime, UInt32UInt32 (Time64)
                (15, 6, 44),  # absTime, UInt32UInt32 (Time64)
                (5, 8, 52),  # ccap, Float
                (6, 8, 56),  # dcap, Float
                (11, 8, 60),  # pow, Float, ignored, shouldn't affect others columns
                (8, 8, 64),  # ceng, Float
                (9, 8, 68),  # deng, Float
            ],
        )
        extra = struct.pack("<IIIIfffff", 2, 500_000_000, 1_700_000_000, 250_000_000, 1800.0, 0.0, 123.4, 3600.0, 0.0)
        record = self._make_ndc15_record(
            step_index=1,
            step_type_raw=1,  # BTS9 stChgCurr -> Common.StepType.cc_chg(1), charge
            index=100,
            total_dws=5,
            total_dwns=0,
            curr=100.0,
            volt=3.7,
            fixed_cap=0.0,
            fixed_eng=0.0,
            extra=extra,
        )
        buf = header + self._make_ndc15_block([record], record_itemsize)

        df = ndc_main.read_ndc_main_15(buf)

        _assert_col(df, "index", [100])
        _assert_col(df, "total_time_s", [5.0])
        _assert_col(df, "step_time_s", [2.5])
        _assert_col(df, "unix_time_s", [1_700_000_000.25])
        _assert_col(df, "capacity_mAh", [0.5])
        _assert_col(df, "energy_mWh", [1.0])

    def test_main_15_unmapped(self) -> None:
        """A step type that is in bts8 but not bts9 should map to undefined (0) here."""
        record_itemsize = 36
        header = self._make_ndc15_header(record_itemsize, custom_items=[])
        record = self._make_ndc15_record(
            step_index=1,
            step_type_raw=26,
            index=1,
            total_dws=0,
            total_dwns=0,
            curr=0.0,
            volt=0.0,
            fixed_cap=0.0,
            fixed_eng=0.0,
        )
        buf = header + self._make_ndc15_block([record], record_itemsize)

        df = ndc_main.read_ndc_main_15(buf)

        _assert_col(df, "step_type", [0])

    def test_main_16(self) -> None:
        """Main for ndax 16."""
        layout = [
            ("voltage_V", "<f4"),
            ("current_mA", "<f4"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(layout, columns={"voltage_V": [36000.0, 35000.0], "current_mA": [200.0, -150.0]})
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=1, version=16)

        df = ndc_main.read_ndc_main_16(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])


class TestNdcAux:
    """Aux ndc files."""

    # 0x65 sub-format, shared by the ndax 9/11/13 and ndax 16 readers
    VOLTAGE_LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("_mask", "<i1"),
        ("voltage_V", "<f4"),
        ("temperature_degC", "<i2"),
    ]

    def test_aux_2(self) -> None:
        """Aux for ndax 2, 4."""
        layout = [
            ("_pad2", "V8"),
            ("index", "<u4"),
            ("_pad3", "V19"),
            ("voltage_V", "<i4"),
            ("_pad4", "V6"),
            ("temperature_degC", "<i2"),
            ("temperature_setpoint_degC", "<i2"),
            ("_pad5", "V49"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(
            layout,
            columns={
                "index": [1, 2],
                "voltage_V": [36000, 35000],
                "temperature_degC": [250, 260],
                "temperature_setpoint_degC": [255, 255],
            },
        )
        buf = _make_ndc_file(
            dtype,
            _split_rows(rows, dtype),
            filetype=5,
            version=2,
            data_start_ind=5,
            record_size=512,
            use_bitmask=False,
        )

        df = ndc_aux.read_ndc_aux_2(buf)

        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "temperature_degC", [25.0, 26.0])
        _assert_col(df, "temperature_setpoint_degC", [25.5, 25.5])

    def test_aux_5(self) -> None:
        """Aux for ndax 5."""
        layout = [
            ("_pad2", "V1"),
            ("index", "<u4"),
            ("_pad3", "V19"),
            ("voltage_V", "<i4"),
            ("_pad4", "V6"),
            ("temperature_degC", "<i2"),
            ("temperature_setpoint_degC", "<i2"),
            ("_pad5", "V49"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(
            layout,
            columns={
                "index": [1, 2],
                "voltage_V": [36000, 35000],
                "temperature_degC": [250, 260],
                "temperature_setpoint_degC": [255, 255],
            },
        )
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=5, version=5)

        df = ndc_aux.read_ndc_aux_5(buf)

        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "temperature_degC", [25.0, 26.0])
        _assert_col(df, "temperature_setpoint_degC", [25.5, 25.5])

    def test_aux_6(self) -> None:
        """Aux for ndax 6, 8, 12, 14, 17."""
        layout = [("?", "<f4")]  # Column name is assigned later from TestInfo.xml
        dtype = np.dtype(layout)
        rows = _build_rows(layout, columns={"?": [25.0, 26.0]})
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=5, version=14)

        df = ndc_aux.read_ndc_aux_6(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "?", [25.0, 26.0])

    def test_aux_11_voltage(self) -> None:
        """Aux for ndax 9, 11, 13 - identifier byte 0x65 selects the voltage+temperature sub-format."""
        dtype = np.dtype(self.VOLTAGE_LAYOUT)
        rows = _build_rows(
            self.VOLTAGE_LAYOUT,
            columns={"voltage_V": [36000.0, 35000.0], "temperature_degC": [250, 260]},
            defaults={"_mask": 0x65},
        )
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=5, version=11)

        df = ndc_aux.read_ndc_aux_11(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "temperature_degC", [25.0, 26.0])

    def test_aux_11_temperature(self) -> None:
        """Aux for ndax 9, 11, 13 - identifier byte 0x74 selects the index+temperature sub-format."""
        layout = [
            ("_pad1", "V1"),
            ("index", "<u4"),
            ("Aux", "<i1"),
            ("_pad2", "V29"),
            ("temperature_degC", "<i2"),
            ("_pad3", "V51"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(layout, columns={"index": [1, 2], "temperature_degC": [250, 260]})
        # _pad1 (the identifier byte the reader inspects) is zero-filled by _pack_record - patch it in directly.
        rows = bytes([0x74]) + rows[1:]
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=5, version=11)

        df = ndc_aux.read_ndc_aux_11(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "temperature_degC", [25.0, 26.0])

    def test_aux_16(self) -> None:
        """Aux for ndax 16 - only the 0x65 voltage+temperature sub-format is implemented."""
        dtype = np.dtype(self.VOLTAGE_LAYOUT)
        rows = _build_rows(
            self.VOLTAGE_LAYOUT,
            columns={"voltage_V": [36000.0, 35000.0], "temperature_degC": [250, 260]},
            defaults={"_mask": 0x65},
        )
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=5, version=16)

        df = ndc_aux.read_ndc_aux_16(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "temperature_degC", [25.0, 26.0])


class TestNdcStep:
    """Step ndc files."""

    def test_step_6(self) -> None:
        """Step for ndax 6, 8, 9, 11, 12, 13, 14."""
        layout = [
            ("cycle_count", "<u4"),
            ("step_index", "<u4"),
            ("_pad1", "V16"),
            ("step_type", "<u1"),
            ("_pad2", "V12"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(layout, columns={"step_index": [1, 2], "step_type": [1, 2]}, defaults={"cycle_count": 0})
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=7, version=14)

        df = ndc_step.read_ndc_step_6(buf)

        _assert_col(df, "cycle_count", [1, 1])
        _assert_col(df, "step_index", [1, 2])
        _assert_col(df, "step_type", [1, 2])
        _assert_col(df, "step_count", [1, 2])

    def test_step_16(self) -> None:
        """Step for ndax 16, 17."""
        layout = [
            ("cycle_count", "<u4"),
            ("step_index", "<u4"),
            ("_pad1", "V16"),
            ("step_type", "<u1"),
            ("_pad2", "V8"),
            ("index", "<u4"),
            ("_pad3", "V63"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(
            layout,
            columns={"step_index": [1, 2], "step_type": [1, 2], "index": [1, 2]},
            defaults={"cycle_count": 0},
        )
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=7, version=16)

        df = ndc_step.read_ndc_step_16(buf)

        _assert_col(df, "cycle_count", [1, 1])
        _assert_col(df, "step_index", [1, 2])
        _assert_col(df, "step_type", [1, 2])
        _assert_col(df, "index", [1, 2])
        _assert_col(df, "step_count", [1, 2])


class TestNdcRunInfo:
    """Run info ndc files."""

    def test_runinfo_8(self) -> None:
        """Run info for ndax 8."""
        layout = [
            ("step_time_s", "<u4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V5"),
            ("dt", "<i4"),
            ("unix_time_s", "<u4"),
            ("step_count", "<u4"),
            ("index", "<u4"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(
            layout,
            columns={
                "step_time_s": [10000, 20000],
                "charge_capacity_mAh": [1.8, 0.0],
                "discharge_capacity_mAh": [0.0, 0.9],
                "charge_energy_mWh": [3.6, 0.0],
                "discharge_energy_mWh": [0.0, 1.8],
                "dt": [10000, 10000],
                "unix_time_s": [1700000000, 1700000010],
                "step_count": [1, 2],
                "index": [1, 2],
            },
        )
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=18, version=8)

        df = ndc_runinfo.read_ndc_runinfo_1(buf)

        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "charge_capacity_mAh", [1800.0, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 900.0])
        _assert_col(df, "dt", [10.0, 10.0])
        _assert_col(df, "unix_time_s", [1700000000, 1700000010])
        _assert_col(df, "step_count", [1, 2])
        _assert_col(df, "index", [1, 2])

    def test_runinfo_9(self) -> None:
        """Run info for ndax 9."""
        layout = [
            ("step_time_s", "<u4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V8"),
            ("dt", "<i4"),
            ("unix_time_s", "<u4"),
            ("step_count", "<u4"),
            ("index", "<u4"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(
            layout,
            columns={
                "step_time_s": [10000, 20000],
                "charge_capacity_mAh": [1800.0, 0.0],
                "discharge_capacity_mAh": [0.0, 900.0],
                "charge_energy_mWh": [3600.0, 0.0],
                "discharge_energy_mWh": [0.0, 1800.0],
                "dt": [10000, 10000],
                "unix_time_s": [1700000000, 1700000010],
                "step_count": [1, 2],
                "index": [1, 2],
            },
        )
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=18, version=9)

        df = ndc_runinfo.read_ndc_runinfo_2(buf)

        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "charge_capacity_mAh", [0.5, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 0.25])
        _assert_col(df, "unix_time_s", [1700000000, 1700000010])
        _assert_col(df, "step_count", [1, 2])
        _assert_col(df, "index", [1, 2])

    def test_runinfo_11(self) -> None:
        """Run info for ndax 11."""
        layout = [
            ("step_time_s", "<u4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V8"),
            ("dt", "<i4"),
            ("unix_time_s", "<u4"),
            ("step_count", "<u4"),
            ("index", "<u4"),
            ("uts_ms", "<u2"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(
            layout,
            columns={
                "step_time_s": [10000, 20000],
                "charge_capacity_mAh": [1800.0, 0.0],
                "discharge_capacity_mAh": [0.0, 900.0],
                "charge_energy_mWh": [3600.0, 0.0],
                "discharge_energy_mWh": [0.0, 1800.0],
                "dt": [10000, 10000],
                "unix_time_s": [1700000000, 1700000010],
                "step_count": [1, 2],
                "index": [1, 2],
                "uts_ms": [500, 250],
            },
        )
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=18, version=11)

        df = ndc_runinfo.read_ndc_runinfo_11(buf)

        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "charge_capacity_mAh", [0.5, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 0.25])
        _assert_col(df, "unix_time_s", [1700000000.5, 1700000010.25])
        _assert_col(df, "step_count", [1, 2])
        _assert_col(df, "index", [1, 2])

    def test_runinfo_12(self) -> None:
        """Run info for ndax 12."""
        layout = [
            ("step_time_s", "<u4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V8"),
            ("dt", "<i4"),
            ("unix_time_s", "<u4"),
            ("step_count", "<u4"),
            ("index", "<u4"),
            ("uts_ms", "<u2"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(
            layout,
            columns={
                "step_time_s": [10000, 20000],
                "charge_capacity_mAh": [1.8, 0.0],
                "discharge_capacity_mAh": [0.0, 0.9],
                "charge_energy_mWh": [3.6, 0.0],
                "discharge_energy_mWh": [0.0, 1.8],
                "dt": [10000, 10000],
                "unix_time_s": [1700000000, 1700000010],
                "step_count": [1, 2],
                "index": [1, 2],
                "uts_ms": [500, 250],
            },
        )
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=18, version=12)

        df = ndc_runinfo.read_ndc_runinfo_12(buf)

        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "charge_capacity_mAh", [1800.0, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 900.0])
        _assert_col(df, "unix_time_s", [1700000000.5, 1700000010.25])
        _assert_col(df, "step_count", [1, 2])
        _assert_col(df, "index", [1, 2])

    def test_runinfo_13(self) -> None:
        """Run info for ndax 13."""
        layout = [
            ("step_time_s", "<u4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V8"),
            ("dt", "<i4"),
            ("unix_time_s", "<u4"),
            ("step_count", "<u4"),
            ("index", "<u4"),
            ("uts_ms", "<u2"),
            ("_pad3", "V8"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(
            layout,
            columns={
                "step_time_s": [10000, 20000],
                "charge_capacity_mAh": [1800.0, 0.0],
                "discharge_capacity_mAh": [0.0, 900.0],
                "charge_energy_mWh": [3600.0, 0.0],
                "discharge_energy_mWh": [0.0, 1800.0],
                "dt": [10000, 10000],
                "unix_time_s": [1700000000, 1700000010],
                "step_count": [1, 2],
                "index": [1, 2],
                "uts_ms": [500, 250],
            },
        )
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=18, version=13)

        df = ndc_runinfo.read_ndc_runinfo_13(buf)

        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "charge_capacity_mAh", [0.5, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 0.25])
        _assert_col(df, "unix_time_s", [1700000000.5, 1700000010.25])
        _assert_col(df, "step_count", [1, 2])
        _assert_col(df, "index", [1, 2])

    def test_runinfo_14(self) -> None:
        """Run info for ndax 14."""
        layout = [
            ("step_time_s", "<u4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V8"),
            ("dt", "<i4"),
            ("unix_time_s", "<u4"),
            ("step_count", "<u4"),
            ("index", "<u4"),
            ("uts_ms", "<i2"),
            ("_pad3", "V8"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(
            layout,
            columns={
                "step_time_s": [10000, 20000],
                "charge_capacity_mAh": [1.8, 0.0],
                "discharge_capacity_mAh": [0.0, 0.9],
                "charge_energy_mWh": [3.6, 0.0],
                "discharge_energy_mWh": [0.0, 1.8],
                "dt": [10000, 10000],
                "unix_time_s": [1700000000, 1700000010],
                "step_count": [1, 2],
                "index": [1, 2],
                "uts_ms": [500, 250],
            },
        )
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=18, version=14)

        df = ndc_runinfo.read_ndc_runinfo_14(buf)

        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "charge_capacity_mAh", [1800.0, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 900.0])
        _assert_col(df, "unix_time_s", [1700000000.5, 1700000010.25])
        _assert_col(df, "step_count", [1, 2])
        _assert_col(df, "index", [1, 2])

    def test_runinfo_16(self) -> None:
        """Run info for ndax 16."""
        layout = [
            ("step_time_s", "<u4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V8"),
            ("dt", "<i4"),
            ("unix_time_s", "<u4"),
            ("step_count", "<u4"),
            ("index", "<u4"),
            ("uts_ms", "<u2"),
            ("_pad3", "V53"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(
            layout,
            columns={
                "step_time_s": [10000, 20000],
                "charge_capacity_mAh": [1800.0, 0.0],
                "discharge_capacity_mAh": [0.0, 900.0],
                "charge_energy_mWh": [3600.0, 0.0],
                "discharge_energy_mWh": [0.0, 1800.0],
                "dt": [10000, 10000],
                "unix_time_s": [1700000000, 1700000010],
                "step_count": [1, 2],
                "index": [1, 2],
                "uts_ms": [500, 250],
            },
        )
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=18, version=16)

        df = ndc_runinfo.read_ndc_runinfo_16(buf)

        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "charge_capacity_mAh", [0.5, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 0.25])
        _assert_col(df, "unix_time_s", [1700000000.5, 1700000010.25])
        _assert_col(df, "step_count", [1, 2])
        _assert_col(df, "index", [1, 2])

    def test_runinfo_17(self) -> None:
        """Run info for ndax 17."""
        layout = [
            ("step_time_s", "<u4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V8"),
            ("dt", "<i4"),
            ("unix_time_s", "<u4"),
            ("step_count", "<u4"),
            ("index", "<u4"),
            ("uts_ms", "<u2"),
            ("_pad3", "V53"),
        ]
        dtype = np.dtype(layout)
        rows = _build_rows(
            layout,
            columns={
                "step_time_s": [10000, 20000],
                "charge_capacity_mAh": [1.8, 0.0],
                "discharge_capacity_mAh": [0.0, 0.9],
                "charge_energy_mWh": [3.6, 0.0],
                "discharge_energy_mWh": [0.0, 1.8],
                "dt": [10000, 10000],
                "unix_time_s": [1700000000, 1700000010],
                "step_count": [1, 2],
                "index": [1, 2],
                "uts_ms": [500, 250],
            },
        )
        buf = _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=18, version=17)

        df = ndc_runinfo.read_ndc_runinfo_17(buf)

        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "charge_capacity_mAh", [1800.0, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 900.0])
        _assert_col(df, "unix_time_s", [1700000000.5, 1700000010.25])
        _assert_col(df, "step_count", [1, 2])
        _assert_col(df, "index", [1, 2])


class TestUnverifiedFormatWarning:
    """UnverifiedFormatWarning fires for unconfirmed (version, filetype) keys, not for confirmed ones."""

    MAIN_LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("voltage_V", "<f4"),
        ("current_mA", "<f4"),
    ]

    def _main_ndc(self, version: int, n_rows: int) -> bytes:
        dtype = np.dtype(self.MAIN_LAYOUT)
        rows = _build_rows(
            self.MAIN_LAYOUT,
            columns={"voltage_V": [3.6] * n_rows, "current_mA": [0.2] * n_rows},
        )
        return _make_ndc_file(dtype, _split_rows(rows, dtype), filetype=1, version=version)

    def test_warns_for_unverified_key(self) -> None:
        """Ndc version 12 filetype 1 (no real data) emits UnverifiedFormatWarning."""
        buf = self._main_ndc(version=12, n_rows=1)

        with pytest.warns(UnverifiedFormatWarning, match="ndc version 12 filetype 1 "):
            df = read_ndc(buf)

        assert len(df) == 1

    def test_no_warning_for_confirmed_key(self) -> None:
        """Ndc version 14 filetype 1 (has real data) emits no UnverifiedFormatWarning."""
        buf = self._main_ndc(version=14, n_rows=1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = read_ndc(buf)

        assert not any(issubclass(w.category, UnverifiedFormatWarning) for w in caught)
        assert len(df) == 1

    def test_read_ndax_consolidates_warnings_from_multiple_unverified_files(self, tmp_path: Path) -> None:
        """A single ndax with several unverified .ndc members should still only warn once.

        Builds a synthetic .ndax with an unverified main, verified step, unverified runInfo.
        Two internal warnings get merged into a single warning.
        """
        data_ndc = self._main_ndc(version=12, n_rows=2)

        runinfo_layout = [
            ("step_time_s", "<u4"),
            ("_pad1", "V1"),
            ("charge_capacity_mAh", "<f4"),
            ("discharge_capacity_mAh", "<f4"),
            ("charge_energy_mWh", "<f4"),
            ("discharge_energy_mWh", "<f4"),
            ("_pad2", "V8"),
            ("dt", "<i4"),
            ("unix_time_s", "<u4"),
            ("step_count", "<u4"),
            ("index", "<u4"),
            ("uts_ms", "<u2"),
            ("_pad3", "V8"),
        ]
        runinfo_dtype = np.dtype(runinfo_layout)
        runinfo_rows = _build_rows(
            runinfo_layout,
            columns={
                "step_time_s": [10000, 20000],
                "charge_capacity_mAh": [1800.0, 0.0],
                "discharge_capacity_mAh": [0.0, 900.0],
                "charge_energy_mWh": [3600.0, 0.0],
                "discharge_energy_mWh": [0.0, 1800.0],
                "dt": [10000, 10000],
                "unix_time_s": [1700000000, 1700001000],
                "step_count": [1, 1],
                "index": [1, 2],
                "uts_ms": [500, 250],
            },
        )
        runinfo_ndc = _make_ndc_file(runinfo_dtype, _split_rows(runinfo_rows, runinfo_dtype), filetype=18, version=13)

        step_layout = [
            ("cycle_count", "<u4"),
            ("step_index", "<u4"),
            ("_pad1", "V16"),
            ("step_type", "<u1"),
            ("_pad2", "V12"),
        ]
        step_dtype = np.dtype(step_layout)
        step_rows = _build_rows(
            step_layout,
            columns={"step_index": [1, 1], "step_type": [1, 1]},
            defaults={"cycle_count": 0},
        )
        step_ndc = _make_ndc_file(step_dtype, _split_rows(step_rows, step_dtype), filetype=7, version=14)

        ndax_path = tmp_path / "synthetic.ndax"
        with zipfile.ZipFile(ndax_path, "w") as zf:
            zf.writestr("data.ndc", data_ndc)
            zf.writestr("data_step.ndc", step_ndc)
            zf.writestr("data_runInfo.ndc", runinfo_ndc)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = read_ndax(ndax_path)

        unverified = [w for w in caught if issubclass(w.category, UnverifiedFormatWarning)]
        assert len(unverified) == 1, (
            f"expected exactly one warning, got {len(unverified)}: {[str(w.message) for w in unverified]}"
        )
        assert "ndc version 12 filetype 1 " in str(unverified[0].message)
        assert "ndc version 14 filetype 7 " not in str(unverified[0].message)
        assert "ndc version 13 filetype 18 " in str(unverified[0].message)
        assert len(df) == 2

    def test_read_ndax_does_not_swallow_unrelated_warnings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only UnverifiedFormatWarning gets combined, everything else passes through."""
        ndax_path = tmp_path / "synthetic.ndax"
        with zipfile.ZipFile(ndax_path, "w") as zf:
            zf.writestr("data.ndc", self._main_ndc(version=12, n_rows=1))

        # Resolve fastnda.ndax fresh via sys.modules rather than trusting this test file's own
        # `from fastnda.ndax import read_ndax` binding - this environment's editable-install finder
        # can re-import a module mid-session, leaving file-top bindings stale relative to what
        # monkeypatch (and importlib.import_module) resolve as "the current module".
        ndax_module = importlib.import_module("fastnda.ndax")
        original_read_ndc = ndax_module.read_ndc

        def read_ndc_with_extra_warning(buf: bytes) -> pl.DataFrame:
            """Add extra warning to test if it passes through."""
            warnings.warn("some unrelated warning", RuntimeWarning, stacklevel=2)
            return original_read_ndc(buf)

        monkeypatch.setattr(ndax_module, "read_ndc", read_ndc_with_extra_warning)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = ndax_module.read_ndax(ndax_path)

        categories = [w.category for w in caught]
        assert RuntimeWarning in categories, "unrelated warning was silently swallowed"
        # Compare against the class as fastnda.ndax itself resolves it, not this file's own
        # (possibly stale) `from fastnda.utils import UnverifiedFormatWarning` binding.
        assert ndax_module.UnverifiedFormatWarning in categories
        assert len(df) == 1


class TestReadNdaxAuxScaling:
    """read_ndax() applies AUX_CHL_SCALE_MAP when merging a generic ('?') aux channel."""

    def test_current_aux_channel_is_scaled_a_to_ma(self, tmp_path: Path) -> None:
        """A ChlType=104 (current) aux channel is renamed to current_mA and scaled A -> mA."""
        main_layout = [
            ("voltage_V", "<f4"),
            ("current_mA", "<f4"),
        ]
        main_dtype = np.dtype(main_layout)
        main_rows = _build_rows(main_layout, columns={"voltage_V": [3.6, 3.7], "current_mA": [0.2, 0.2]})
        data_ndc = _make_ndc_file(main_dtype, _split_rows(main_rows, main_dtype), filetype=1, version=14)

        # ndc version 14 filetype 5 gives one "?" column that looks up the AUX_CHL_MAP and scales
        aux_layout = [("?", "<f4")]
        aux_dtype = np.dtype(aux_layout)
        aux_rows = _build_rows(aux_layout, columns={"?": [0.05, -0.03]})
        aux_ndc = _make_ndc_file(aux_dtype, _split_rows(aux_rows, aux_dtype), filetype=5, version=14)

        # ChlType 104 = current, AUX_CHL_SCALE_MAP scales it A->mA.
        test_info_xml = (
            '<?xml version="1.0" encoding="gb2312"?>'
            "<root><config><TestInfo>"
            '<AuxChl_1 ChlType="104" AuxID="1"/>'
            "</TestInfo></config></root>"
        )

        ndax_path = tmp_path / "synthetic.ndax"
        with zipfile.ZipFile(ndax_path, "w") as zf:
            zf.writestr("data.ndc", data_ndc)
            zf.writestr("data_AUX_1_1_1.ndc", aux_ndc)
            zf.writestr("TestInfo.xml", test_info_xml.encode("gb2312"))

        df = read_ndax(ndax_path)

        assert "?" not in df.columns
        _assert_col(df, "aux1_current_mA", [50.0, -30.0])
