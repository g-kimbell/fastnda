"""Tests for read metadata functions."""

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pytest

from fastnda import read_metadata


class TestMetaData:
    """Test class for reading metadata."""

    def test_read_metadata(self, file_pair: tuple[Path, Path]) -> None:
        """Basic checks for metadata reading."""
        test_file = file_pair[0]
        is_ndax = test_file.suffix == ".ndax"

        if test_file.suffix == ".zip":
            with TemporaryDirectory() as tmp_dir, ZipFile(test_file, "r") as zip_test:
                # unzip file to a temp location and read
                zip_test.extractall(tmp_dir)
                test_file = Path(tmp_dir) / test_file.stem
                metadata = read_metadata(test_file)
        else:
            metadata = read_metadata(test_file)

        assert isinstance(metadata, dict)

        if is_ndax:
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
