# Copyright © 2026, Empa.
"""Ensure functions behave for missing/unknown files."""

import mmap
from pathlib import Path

import pytest

from fastnda._ndc import read_ndc
from fastnda._ndc.ndc_aux import read_ndc_aux_11, read_ndc_aux_16
from fastnda.nda import (
    _decode_datetime_us,
    _read_nda_29,
    _read_nda_130_metadata,
    _read_nda_130_test_info,
    _read_nda_test_info,
    _read_pack_test_info_old_extension,
    read_nda,
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
        assert _read_nda_130_test_info(mm) == {}

    def test_decode_datetime_invalid(self) -> None:
        """Zero and out-of-range timestamps decode to None."""
        assert _decode_datetime_us(bytes(8)) is None
        assert _decode_datetime_us(b"\xff" * 8) is None

    def test_pack_test_info_old_extension_truncated(self) -> None:
        """Record too short for the pstring section gives empty metadata."""
        assert _read_pack_test_info_old_extension(b"") == {}

    def test_nda_130_metadata_no_version_string(self) -> None:
        """Header without a version string skips the BTS version field."""
        metadata = _read_nda_130_metadata(mmap.mmap(-1, 2048))
        assert "bts_version" not in metadata
