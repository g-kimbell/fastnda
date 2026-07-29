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


class TestReadNdcMain1:
    """DFDATA_V1 (ndc versions 1, 3): old-format container, record_size=512, no bitmask."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
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
    DEFAULTS: ClassVar[dict[str, int]] = {"cycle_count": 0, "range": 100, "Y": 2024, "M": 1, "D": 1}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values."""
        rows = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
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
        )
        row_bytes = [rows[i * 83 : (i + 1) * 83] for i in range(2)]
        dtype = np.dtype(self.LAYOUT)
        buf = _make_ndc_file(
            dtype, row_bytes, filetype=1, version=1, data_start_ind=5, record_size=512, use_bitmask=False
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


class TestReadNdcMain2:
    """DFDATA_V2 (ndc versions 2, 4): old-format container, record_size=512, no bitmask."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
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
    DEFAULTS: ClassVar[dict[str, int]] = {"cycle_count": 0, "range": 100, "Y": 2024, "M": 1, "D": 1}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values."""
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
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
        )
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(
            dtype, row_bytes, filetype=1, version=2, data_start_ind=5, record_size=512, use_bitmask=False
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


class TestReadNdcAux2:
    """DFDATA_V2 aux channel (ndc versions 2, 4): old-format container, voltage + 2 temperatures."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("_pad2", "V8"),
        ("index", "<u4"),
        ("_pad3", "V19"),
        ("voltage_V", "<i4"),
        ("_pad4", "V6"),
        ("temperature_degC", "<i2"),
        ("temperature_setpoint_degC", "<i2"),
        ("_pad5", "V49"),
    ]
    DEFAULTS: ClassVar[dict[str, int]] = {}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values."""
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
            columns={
                "index": [1, 2],
                "voltage_V": [36000, 35000],
                "temperature_degC": [250, 260],
                "temperature_setpoint_degC": [255, 255],
            },
        )
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(
            dtype, row_bytes, filetype=5, version=2, data_start_ind=5, record_size=512, use_bitmask=False
        )

        df = ndc_aux.read_ndc_aux_2(buf)

        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "temperature_degC", [25.0, 26.0])
        _assert_col(df, "temperature_setpoint_degC", [25.5, 25.5])


class TestReadNdcMain5:
    """DFDATA_V5 (ndc versions 5, 7): new-format container, bitmask-based."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
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
    DEFAULTS: ClassVar[dict[str, int]] = {"cycle_count": 0, "range": 100, "Y": 2024, "M": 1, "D": 1}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values."""
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
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
        )
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=1, version=5)

        df = ndc_main.read_ndc_main_5(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "cycle_count", [1, 1])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])
        _assert_col(df, "charge_capacity_mAh", [0.5, 0.25])
        _assert_col(df, "charge_energy_mWh", [1.0, 0.5])
        _assert_col(df, "step_count", [1, 2])


class TestReadNdcAux5:
    """DFDATA_V5 aux channel (ndc versions 5, 7): new-format container, voltage + 2 temperatures."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("_pad2", "V1"),
        ("index", "<u4"),
        ("_pad3", "V19"),
        ("voltage_V", "<i4"),
        ("_pad4", "V6"),
        ("temperature_degC", "<i2"),
        ("temperature_setpoint_degC", "<i2"),
        ("_pad5", "V49"),
    ]
    DEFAULTS: ClassVar[dict[str, int]] = {}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values."""
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
            columns={
                "index": [1, 2],
                "voltage_V": [36000, 35000],
                "temperature_degC": [250, 260],
                "temperature_setpoint_degC": [255, 255],
            },
        )
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=5, version=5)

        df = ndc_aux.read_ndc_aux_5(buf)

        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "temperature_degC", [25.0, 26.0])
        _assert_col(df, "temperature_setpoint_degC", [25.5, 25.5])


class TestReadNdcMain6:
    """DFDATA_V6 (ndc version 6): main record already carries cumulative cap/energy, no runinfo file."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
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
    DEFAULTS: ClassVar[dict[str, int]] = {}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values.

        Raw capacity/energy is Ah|Wh and needs *1000 for mAh|mWh.
        """
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
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
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=1, version=6)

        df = ndc_main.read_ndc_main_6(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])
        _assert_col(df, "charge_capacity_mAh", [1800.0, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 900.0])
        _assert_col(df, "unix_time_s", [1700000000, 1700000010])
        _assert_col(df, "step_count", [1, 2])


class TestReadNdcRunInfo8:
    """DFDATARunInfo_V8 (ndc version 8): new-format container, no wATimeMS field."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
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
    DEFAULTS: ClassVar[dict[str, int]] = {}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values."""
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
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
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=18, version=8)

        df = ndc_runinfo.read_ndc_runinfo_1(buf)

        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "charge_capacity_mAh", [1800.0, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 900.0])
        _assert_col(df, "dt", [10.0, 10.0])
        _assert_col(df, "unix_time_s", [1700000000, 1700000010])
        _assert_col(df, "step_count", [1, 2])
        _assert_col(df, "index", [1, 2])


class TestReadNdcRunInfo9:
    """DFDATARunInfo_V9 (ndc version 9): new-format container, dwWorkType is 4 bytes (unlike v8's 1)."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
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
    DEFAULTS: ClassVar[dict[str, int]] = {}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values.

        Raw capacity/energy is mAs|mWs and needs /3600 for mAh|mWh.
        """
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
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
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=18, version=9)

        df = ndc_runinfo.read_ndc_runinfo_2(buf)

        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "charge_capacity_mAh", [0.5, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 0.25])
        _assert_col(df, "unix_time_s", [1700000000, 1700000010])
        _assert_col(df, "step_count", [1, 2])
        _assert_col(df, "index", [1, 2])


class TestReadNdcRunInfo13:
    """DFDATARunInfo13 (ndc version 13): new-format container, adds fTotalCap/fTotalEng and wATimeMS."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
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
    DEFAULTS: ClassVar[dict[str, int]] = {}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values.

        Raw capacity/energy is mAs|mWs and needs /3600 for mAh|mWh.
        """
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
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
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=18, version=13)

        df = ndc_runinfo.read_ndc_runinfo_13(buf)

        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "charge_capacity_mAh", [0.5, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 0.25])
        _assert_col(df, "unix_time_s", [1700000000.5, 1700000010.25])
        _assert_col(df, "step_count", [1, 2])
        _assert_col(df, "index", [1, 2])


class TestReadNdcMain11:
    """DFDATA (ndc versions 8, 9, 11, 12, 13): bare voltage/current float pair, add_index=True."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("voltage_V", "<f4"),
        ("current_mA", "<f4"),
    ]
    DEFAULTS: ClassVar[dict[str, int]] = {}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values."""
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(
            self.LAYOUT, self.DEFAULTS, columns={"voltage_V": [36000.0, 35000.0], "current_mA": [200.0, -150.0]}
        )
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=1, version=11)

        df = ndc_main.read_ndc_main_11(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])


class TestReadNdcAux11:
    """Aux channel (ndc version 11): dispatches on the first data byte between two sub-formats."""

    VOLTAGE_LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("_mask", "<i1"),
        ("voltage_V", "<f4"),
        ("temperature_degC", "<i2"),
    ]
    TEMP_LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("_pad1", "V1"),
        ("index", "<u4"),
        ("Aux", "<i1"),
        ("_pad2", "V29"),
        ("temperature_degC", "<i2"),
        ("_pad3", "V51"),
    ]

    def test_voltage_subformat(self) -> None:
        """Identifier byte 0x65 selects the voltage+temperature sub-format."""
        dtype = np.dtype(self.VOLTAGE_LAYOUT)
        rows = _build_rows(
            self.VOLTAGE_LAYOUT,
            {"_mask": 0x65},
            columns={"voltage_V": [36000.0, 35000.0], "temperature_degC": [250, 260]},
        )
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=5, version=11)

        df = ndc_aux.read_ndc_aux_11(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "temperature_degC", [25.0, 26.0])

    def test_temperature_subformat(self) -> None:
        """Identifier byte 0x74 selects the index+temperature sub-format."""
        dtype = np.dtype(self.TEMP_LAYOUT)
        rows = _build_rows(
            self.TEMP_LAYOUT,
            {},
            columns={"index": [1, 2], "temperature_degC": [250, 260]},
        )
        # _pad1 (the identifier byte the reader inspects) is zero-filled by _pack_record - patch it in directly.
        rows = bytes([0x74]) + rows[1:]
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=5, version=11)

        df = ndc_aux.read_ndc_aux_11(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "temperature_degC", [25.0, 26.0])


class TestReadNdcRunInfo11:
    """DFDATARunInfo (ndc version 11): new-format container, with wATimeMS but no fTotalCap/Eng."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
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
    DEFAULTS: ClassVar[dict[str, int]] = {}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values."""
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
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
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=18, version=11)

        df = ndc_runinfo.read_ndc_runinfo_11(buf)

        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "charge_capacity_mAh", [0.5, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 0.25])
        _assert_col(df, "unix_time_s", [1700000000.5, 1700000010.25])
        _assert_col(df, "step_count", [1, 2])
        _assert_col(df, "index", [1, 2])


class TestReadNdcRunInfo12:
    """DFDATARunInfo (ndc version 12): similar to 11 with different scaling."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
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
    DEFAULTS: ClassVar[dict[str, int]] = {}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values.

        Raw capacity/energy is Ah|Wh and needs *1000 for mAh|mWh.
        """
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
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
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=18, version=12)

        df = ndc_runinfo.read_ndc_runinfo_12(buf)

        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "charge_capacity_mAh", [1800.0, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 900.0])
        _assert_col(df, "unix_time_s", [1700000000.5, 1700000010.25])
        _assert_col(df, "step_count", [1, 2])
        _assert_col(df, "index", [1, 2])


class TestReadNdcMain14:
    """DFDATA (ndc versions 8, 12, 14): bare voltage/current float pair, current scaled by 1000."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("voltage_V", "<f4"),
        ("current_mA", "<f4"),
    ]
    DEFAULTS: ClassVar[dict[str, int]] = {}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values."""
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(self.LAYOUT, self.DEFAULTS, columns={"voltage_V": [3.6, 3.5], "current_mA": [0.2, -0.15]})
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=1, version=14)

        df = ndc_main.read_ndc_main_14(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])


class TestReadNdcAux14:
    """Generic aux channel (ndc versions 6, 8, 12, 14, 17): single value column, renamed later from TestInfo.xml."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [("?", "<f4")]
    DEFAULTS: ClassVar[dict[str, int]] = {}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values."""
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(self.LAYOUT, self.DEFAULTS, columns={"?": [25.0, 26.0]})
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=5, version=14)

        df = ndc_aux.read_ndc_aux_6(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "?", [25.0, 26.0])


class TestReadNdcStep14:
    """StepDFDATA (ndc versions 6, 8, 9, 11, 12, 13, 14): sequential step_count from int_range, not step_index."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("cycle_count", "<u4"),
        ("step_index", "<u4"),
        ("_pad1", "V16"),
        ("step_type", "<u1"),
        ("_pad2", "V12"),
    ]
    DEFAULTS: ClassVar[dict[str, int]] = {"cycle_count": 0}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values."""
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(self.LAYOUT, self.DEFAULTS, columns={"step_index": [1, 2], "step_type": [1, 2]})
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=7, version=14)

        df = ndc_step.read_ndc_step_6(buf)

        _assert_col(df, "cycle_count", [1, 1])
        _assert_col(df, "step_index", [1, 2])
        _assert_col(df, "step_type", [1, 2])
        _assert_col(df, "step_count", [1, 2])


class TestReadNdcRunInfo14:
    """DFDATARunInfo13-like struct (ndc version 14): uts_ms is signed, capacity/energy scaled *1000."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
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
    DEFAULTS: ClassVar[dict[str, int]] = {}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values."""
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
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
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=18, version=14)

        df = ndc_runinfo.read_ndc_runinfo_14(buf)

        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "charge_capacity_mAh", [1800.0, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 900.0])
        _assert_col(df, "unix_time_s", [1700000000.5, 1700000010.25])
        _assert_col(df, "step_count", [1, 2])
        _assert_col(df, "index", [1, 2])


class TestReadNdcMain16:
    """DFDATA_V16-family main (ndc version 16): voltage/10000, current unscaled."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("voltage_V", "<f4"),
        ("current_mA", "<f4"),
    ]
    DEFAULTS: ClassVar[dict[str, int]] = {}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values."""
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(
            self.LAYOUT, self.DEFAULTS, columns={"voltage_V": [36000.0, 35000.0], "current_mA": [200.0, -150.0]}
        )
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=1, version=16)

        df = ndc_main.read_ndc_main_16(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "current_mA", [200.0, -150.0])


class TestReadNdcAux16:
    """Aux channel (ndc version 16): only the 0x65 voltage+temperature sub-format is implemented."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("_mask", "<i1"),
        ("voltage_V", "<f4"),
        ("temperature_degC", "<i2"),
    ]

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values."""
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(
            self.LAYOUT,
            {"_mask": 0x65},
            columns={"voltage_V": [36000.0, 35000.0], "temperature_degC": [250, 260]},
        )
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=5, version=16)

        df = ndc_aux.read_ndc_aux_16(buf)

        _assert_col(df, "index", [1, 2])
        _assert_col(df, "voltage_V", [3.6, 3.5])
        _assert_col(df, "temperature_degC", [25.0, 26.0])


class TestReadNdcStep16:
    """StepDFDATA16 (ndc versions 16, 17): adds an explicit index field, step_count via count-changes."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
        ("cycle_count", "<u4"),
        ("step_index", "<u4"),
        ("_pad1", "V16"),
        ("step_type", "<u1"),
        ("_pad2", "V8"),
        ("index", "<u4"),
        ("_pad3", "V63"),
    ]
    DEFAULTS: ClassVar[dict[str, int]] = {"cycle_count": 0}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values."""
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
            columns={"step_index": [1, 2], "step_type": [1, 2], "index": [1, 2]},
        )
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=7, version=16)

        df = ndc_step.read_ndc_step_16(buf)

        _assert_col(df, "cycle_count", [1, 1])
        _assert_col(df, "step_index", [1, 2])
        _assert_col(df, "step_type", [1, 2])
        _assert_col(df, "index", [1, 2])
        _assert_col(df, "step_count", [1, 2])


class TestReadNdcRunInfo16:
    """DFDATARunInfo16 (ndc version 16): same as v11's runinfo but with a larger trailing pad."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
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
    DEFAULTS: ClassVar[dict[str, int]] = {}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values."""
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
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
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=18, version=16)

        df = ndc_runinfo.read_ndc_runinfo_16(buf)

        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "charge_capacity_mAh", [0.5, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 0.25])
        _assert_col(df, "unix_time_s", [1700000000.5, 1700000010.25])
        _assert_col(df, "step_count", [1, 2])
        _assert_col(df, "index", [1, 2])


class TestReadNdcRunInfo17:
    """DFDATARunInfo16 (ndc version 17): capacity/energy scaled *1000, same struct as v16's runinfo."""

    LAYOUT: ClassVar[list[tuple[str, str]]] = [
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
    DEFAULTS: ClassVar[dict[str, int]] = {}

    def test_decodes_expected_values(self) -> None:
        """Create synthetic ndc, read back, check against expected values."""
        dtype = np.dtype(self.LAYOUT)
        rows = _build_rows(
            self.LAYOUT,
            self.DEFAULTS,
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
        row_bytes = [rows[i * dtype.itemsize : (i + 1) * dtype.itemsize] for i in range(2)]
        buf = _make_ndc_file(dtype, row_bytes, filetype=18, version=17)

        df = ndc_runinfo.read_ndc_runinfo_17(buf)

        _assert_col(df, "step_time_s", [10.0, 20.0])
        _assert_col(df, "charge_capacity_mAh", [1800.0, 0.0])
        _assert_col(df, "discharge_capacity_mAh", [0.0, 900.0])
        _assert_col(df, "unix_time_s", [1700000000.5, 1700000010.25])
        _assert_col(df, "step_count", [1, 2])
        _assert_col(df, "index", [1, 2])


class TestUnverifiedFormatWarning:
    """UnverifiedFormatWarning fires for unconfirmed (version, filetype) keys, not for confirmed ones.

    Builds full header+data-block buffers and reads them through ndax._read_ndc, where the
    warning fires.
    """

    def test_warns_for_unverified_key(self) -> None:
        """Ndc version 1 filetype 1 (no real data) emits UnverifiedFormatWarning."""
        dtype = np.dtype(TestReadNdcMain1.LAYOUT)
        rows = _build_rows(
            TestReadNdcMain1.LAYOUT,
            TestReadNdcMain1.DEFAULTS,
            columns={
                "index": [1],
                "step_index": [1],
                "step_type": [1],
                "step_time_s": [10000],
                "voltage_V": [36000],
                "current_mA": [20000],
                "charge_capacity_mAh": [180000],
                "discharge_capacity_mAh": [0],
                "charge_energy_mWh": [360000],
                "discharge_energy_mWh": [0],
                "s": [0],
            },
        )
        buf = _make_ndc_file(dtype, [rows], filetype=1, version=1, data_start_ind=5, record_size=512, use_bitmask=False)

        with pytest.warns(UnverifiedFormatWarning, match="ndc version 1 filetype 1 "):
            df = read_ndc(buf)

        assert len(df) == 1

    def test_no_warning_for_confirmed_key(self) -> None:
        """Ndc version 5 filetype 1 (has real data) emits no UnverifiedFormatWarning."""
        dtype = np.dtype(TestReadNdcMain5.LAYOUT)
        rows = _build_rows(
            TestReadNdcMain5.LAYOUT,
            TestReadNdcMain5.DEFAULTS,
            columns={
                "index": [1],
                "step_index": [1],
                "step_type": [1],
                "step_time_s": [10000],
                "voltage_V": [36000],
                "current_mA": [20000],
                "charge_capacity_mAh": [180000],
                "discharge_capacity_mAh": [0],
                "charge_energy_mWh": [360000],
                "discharge_energy_mWh": [0],
                "s": [0],
            },
        )
        buf = _make_ndc_file(dtype, [rows], filetype=1, version=5)

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
        main_dtype = np.dtype(TestReadNdcMain6.LAYOUT)
        main_rows = _build_rows(
            TestReadNdcMain6.LAYOUT,
            TestReadNdcMain6.DEFAULTS,
            columns={
                "step_time_s": [1000, 2000],
                "voltage_V": [3.6, 3.7],
                "current_mA": [0.5, 0.5],
                "charge_capacity_mAh": [1.0, 1.1],
                "discharge_capacity_mAh": [0.0, 0.0],
                "charge_energy_mWh": [3.6, 3.7],
                "discharge_energy_mWh": [0.0, 0.0],
                "unix_time_s": [1700000000, 1700001000],
                "step_count": [1, 1],
            },
        )
        main_row_bytes = [main_rows[i * main_dtype.itemsize : (i + 1) * main_dtype.itemsize] for i in range(2)]
        data_ndc = _make_ndc_file(main_dtype, main_row_bytes, filetype=1, version=6)

        runinfo_dtype = np.dtype(TestReadNdcRunInfo13.LAYOUT)
        runinfo_rows = _build_rows(
            TestReadNdcRunInfo13.LAYOUT,
            TestReadNdcRunInfo13.DEFAULTS,
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
        runinfo_row_bytes = [
            runinfo_rows[i * runinfo_dtype.itemsize : (i + 1) * runinfo_dtype.itemsize] for i in range(2)
        ]
        runinfo_ndc = _make_ndc_file(runinfo_dtype, runinfo_row_bytes, filetype=18, version=13)

        step_dtype = np.dtype(TestReadNdcStep14.LAYOUT)
        step_rows = _build_rows(
            TestReadNdcStep14.LAYOUT,
            TestReadNdcStep14.DEFAULTS,
            columns={"step_index": [1, 1], "step_type": [1, 1]},
        )
        step_row_bytes = [step_rows[i * step_dtype.itemsize : (i + 1) * step_dtype.itemsize] for i in range(2)]
        step_ndc = _make_ndc_file(step_dtype, step_row_bytes, filetype=7, version=14)

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
        assert "ndc version 6 filetype 1 " in str(unverified[0].message)
        assert "ndc version 14 filetype 7 " not in str(unverified[0].message)
        assert "ndc version 13 filetype 18 " in str(unverified[0].message)
        assert len(df) == 2

    def test_read_ndax_does_not_swallow_unrelated_warnings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only UnverifiedFormatWarning gets combined, everything else passes through."""
        main_dtype = np.dtype(TestReadNdcMain6.LAYOUT)
        main_rows = _build_rows(
            TestReadNdcMain6.LAYOUT,
            TestReadNdcMain6.DEFAULTS,
            columns={
                "step_time_s": [1000],
                "voltage_V": [3.6],
                "current_mA": [0.5],
                "charge_capacity_mAh": [1.0],
                "discharge_capacity_mAh": [0.0],
                "charge_energy_mWh": [3.6],
                "discharge_energy_mWh": [0.0],
                "unix_time_s": [1700000000],
                "step_count": [1],
            },
        )
        data_ndc = _make_ndc_file(main_dtype, [main_rows], filetype=1, version=6)
        ndax_path = tmp_path / "synthetic.ndax"
        with zipfile.ZipFile(ndax_path, "w") as zf:
            zf.writestr("data.ndc", data_ndc)

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
        main_dtype = np.dtype(TestReadNdcMain14.LAYOUT)
        main_rows = _build_rows(
            TestReadNdcMain14.LAYOUT,
            TestReadNdcMain14.DEFAULTS,
            columns={"voltage_V": [3.6, 3.7], "current_mA": [0.2, 0.2]},
        )
        main_row_bytes = [main_rows[i * main_dtype.itemsize : (i + 1) * main_dtype.itemsize] for i in range(2)]
        data_ndc = _make_ndc_file(main_dtype, main_row_bytes, filetype=1, version=6)

        # ndc version 14 filetype 5 gives one "?" column that looksup the AUX_CHL_MAP and scales
        aux_dtype = np.dtype(TestReadNdcAux14.LAYOUT)
        aux_rows = _build_rows(TestReadNdcAux14.LAYOUT, TestReadNdcAux14.DEFAULTS, columns={"?": [0.05, -0.03]})
        aux_row_bytes = [aux_rows[i * aux_dtype.itemsize : (i + 1) * aux_dtype.itemsize] for i in range(2)]
        aux_ndc = _make_ndc_file(aux_dtype, aux_row_bytes, filetype=5, version=14)

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
