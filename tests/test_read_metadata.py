"""Test read_metadata functionality."""

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pytest

import fastnda


class TestRead:
    """Compared parsed data to reference from BTSDA."""

    def test_read_metadata(self, file_pair: tuple[Path, Path]) -> None:
        """Test reading metadata."""
        test_file, _ref_file = file_pair
        if test_file.suffix == ".zip":  # Is zipped nda
            with TemporaryDirectory() as tmp_dir, ZipFile(test_file, "r") as zip_test:
                # unzip file to a temp location and read
                zip_test.extractall(tmp_dir)
                test_file = Path(tmp_dir) / test_file.stem
                metadata = fastnda.read_metadata(test_file)
        else:
            metadata = fastnda.read_metadata(test_file)
        assert isinstance(metadata, dict)

    def test_read_metadata_wrong_filetype(self) -> None:
        """Test using the wrong file."""
        test_file = Path(r"wrong_file.csv")
        with pytest.raises(ValueError):
            fastnda.read_metadata(test_file)
