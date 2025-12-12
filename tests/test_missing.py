"""Ensure functions behave for missing/unknown files."""

import mmap
from pathlib import Path

import pytest

from fastnda.nda import _read_nda_8, _read_nda_29, read_nda, read_nda_metadata
from fastnda.ndax import _read_ndc_16_filetype_5, read_ndc


class TestMissing:
    """Tests for bad/missing files."""

    def test_bad_ndc(self) -> None:
        """Unknown ndc type."""
        with pytest.raises(NotImplementedError):
            read_ndc(b"999999999")

    def test_bad_16_5(self) -> None:
        """Unknown ndc16 file5 pattern."""
        with pytest.raises(NotImplementedError):
            _read_ndc_16_filetype_5(b"999999999")

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
        with file.open("rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        with pytest.raises(EOFError):
            _read_nda_29(mm)
        with pytest.raises(EOFError):
            _read_nda_8(mm)
