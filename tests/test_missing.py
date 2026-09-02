# Copyright © 2026, Empa.
"""Ensure functions behave for missing/unknown files."""

import datetime
import mmap
from pathlib import Path

import pytest

from fastnda._ndc import read_ndc
from fastnda._ndc.ndc_aux import read_ndc_aux_11, read_ndc_aux_16
from fastnda.nda import _read_nda_29, read_nda
from fastnda.nda_meta import (
    _decode_datetime_us,
    _find_version_pstring,
    _read_bts9_metadata,
    _read_bts9_test_info,
    _read_fields,
    _read_nda_test_info,
    _read_pack_test_info_chain,
    _read_pack_test_info_new,
    _read_pack_test_info_old,
    read_nda_metadata,
)


class TestMissing:
    """Tests for bad/missing files."""

    def test_bad_ndc(self) -> None:
        """Unknown ndc type/file patterns."""
        with pytest.raises(NotImplementedError):
            read_ndc(b"999999999")
        with pytest.raises(NotImplementedError):
            read_ndc_aux_11(b"999999999")
        with pytest.raises(NotImplementedError):
            read_ndc_aux_16(b"999999999")

    def test_bad_nda(self, tmp_path: Path) -> None:
        """Unknown nda type."""
        file = tmp_path / "file.nda"
        with file.open("w") as f:
            f.write("NEWARE this is not a real nda file")
        with pytest.raises(NotImplementedError):
            read_nda(file)
        with file.open("w") as f:
            f.write("this doesnt even have neware at the start")
        with pytest.raises(ValueError):
            read_nda_metadata(file)
        with file.open("wb") as f:
            f.write(b"NEWARE" + b"\x00" * 8 + b"\x82" + 1024 * b"\x00")
        with pytest.raises(NotImplementedError, match=r"does not match BTS9.0 or BTS9.1"):
            read_nda(file)
        with file.open("rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        with pytest.raises(EOFError):
            _read_nda_29(mm)

    def test_missing_test_info(self) -> None:
        """Zeroed test info pointers give empty metadata."""
        mm = mmap.mmap(-1, 2048)
        assert _read_nda_test_info(mm, 29) == {}
        assert _read_bts9_test_info(mm) == {}

    def test_decode_datetime_invalid(self) -> None:
        """Zero and out-of-range timestamps decode to None."""
        assert _decode_datetime_us(bytes(8)) is None
        assert _decode_datetime_us(b"\xff" * 8) is None

    def test_pack_test_info_no_timestamp_anchor(self) -> None:
        """A record with no start/stop timestamp pair gives empty metadata."""
        assert _read_pack_test_info_old(bytes(512)) == {}
        assert _read_pack_test_info_new(bytes(512)) == {}

    def test_read_fields_out_of_range(self) -> None:
        """Fields reaching past the record, past the limit, or before its start are skipped."""
        assert _read_fields(bytes(3), {"test_id": (0, "u32")}) == {}
        assert _read_fields(bytes(8), {"test_id": (0, "u32")}, limit=2) == {}
        assert _read_fields(bytes(8), {"test_id": (0, "u32")}, base=-4) == {}

    def test_pack_test_info_old_no_version_string(self) -> None:
        """A record with timestamps but no version string at either chain offset gives empty metadata."""
        record = bytearray(512)
        start = int(datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc).timestamp() * 1e6)
        record[200:208] = start.to_bytes(8, "little")
        record[208:216] = (start + 3600 * 1_000_000).to_bytes(8, "little")
        assert _find_version_pstring(record) is None
        assert _read_pack_test_info_old(bytes(record)) == {}

        # a version string too far from the timestamps to be this record's chain
        version = b"8.0.0.1.2"
        record[300] = len(version)
        record[301 : 301 + len(version)] = version
        assert _find_version_pstring(record) == 300
        assert _read_pack_test_info_old(bytes(record)) == {}

    def test_pack_test_info_chain_truncated(self) -> None:
        """A string chain running past the end of the record gives empty metadata."""
        assert _read_pack_test_info_chain(b"\x40abc", 0, counted=True) == {}

    def test_bts9_metadata_no_version_string(self) -> None:
        """Header without a version string skips the BTS version field."""
        metadata = _read_bts9_metadata(mmap.mmap(-1, 2048))
        assert "bts_version" not in metadata
