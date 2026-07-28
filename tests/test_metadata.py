# Copyright © 2026, Empa.
"""Tests for read metadata functions."""

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pytest

from fastnda import read_metadata


def _read_metadata(test_file: Path) -> dict:
    if "nometa" in test_file.stem:
        pytest.skip(f"Explicitly no metadata in {test_file.stem}")

    if test_file.suffix == ".zip":
        with TemporaryDirectory() as tmp_dir, ZipFile(test_file, "r") as zip_test:
            # unzip file to a temp location and read
            zip_test.extractall(tmp_dir)
            test_file = Path(tmp_dir) / test_file.stem
            return read_metadata(test_file)
    return read_metadata(test_file)


class TestMetaData:
    """Test class for reading metadata."""

    def test_read_metadata(self, file_pair: tuple[Path, Path]) -> None:
        """Basic checks for metadata reading."""
        test_file = file_pair[0]
        metadata = _read_metadata(test_file)
        assert isinstance(metadata, dict)
        if test_file.suffix == ".ndax":
            assert "VersionInfo" in metadata
            assert "Step" in metadata
            assert "TestInfo" in metadata
        else:
            assert "nda_version" in metadata

    def test_read_bad_file(self, tmp_path: Path) -> None:
        """Test reading invalid file metadata."""
        wrong_file = tmp_path / "wrong_file.txt"
        with wrong_file.open("w") as f:
            f.write("not a neware file")
        with pytest.raises(ValueError):
            read_metadata(wrong_file)

    def test_nda8(self) -> None:
        """Test reading metadata from NDA8."""
        current_folder = Path(__file__).parent
        test_file = current_folder / "test_data" / "nda_v8.nda.zip"
        metadata = _read_metadata(test_file)
        assert metadata["nda_version"] == 8
        assert metadata["start_time"] == "2015.09.16 12:08:41"
        assert metadata["creator"] == ""
        assert metadata["sn"] == ""
        assert metadata["remarks"] == ""
        assert metadata["active_mass_mg"] == 0.0
        assert "barcode" not in metadata  # not present before the v17

    def test_nda22(self) -> None:
        """Test reading metadata from NDA22."""
        current_folder = Path(__file__).parent
        test_file = current_folder / "test_data" / "nda_v22.nda.zip"
        metadata = _read_metadata(test_file)
        assert metadata["nda_version"] == 22
        assert metadata["start_time"] == "2015-10-14 11:28:31"
        assert metadata["creator"] == ""
        assert metadata["sn"] == "2015-10-14 11-28-31"
        assert metadata["barcode"] == ""
        assert metadata["active_mass_mg"] == 1.0

    def test_nda23(self) -> None:
        """Test reading metadata from NDA23."""
        current_folder = Path(__file__).parent
        test_file = current_folder / "test_data" / "nda_v23.nda.zip"
        metadata = _read_metadata(test_file)
        assert metadata["nda_version"] == 23
        assert metadata["creator"] == ""
        assert metadata["sn"] == "2017-05-10 10-20-37"
        assert metadata["remarks"] == ""

    def test_nda26(self) -> None:
        """Test reading metadata from NDA26."""
        current_folder = Path(__file__).parent
        test_file = current_folder / "test_data" / "nda_v26.nda.zip"
        metadata = _read_metadata(test_file)
        assert metadata["nda_version"] == 26
        assert metadata["creator"] == "LIN"
        assert metadata["sn"] == "2016-09-18 10-38-50"
        assert metadata["remarks"] == "2-1"
