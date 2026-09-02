# Copyright © 2026, Empa.
"""Tests for read metadata functions."""

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pytest

from fastnda import read_metadata

# NDA 129/130 currently always warn about metadata reading
pytestmark = pytest.mark.filterwarnings("ignore:read_metadata for NDA 129/130")


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

    def test_nda130_bts903(self) -> None:
        """Test reading metadata from an old BTS9.0.3 nda_version 130 file."""
        current_folder = Path(__file__).parent
        test_file = current_folder / "test_data" / "2-1-6_61_07005012.nda.zip"
        metadata = _read_metadata(test_file)
        assert metadata["nda_version"] == 130
        assert metadata["bts_version"] == "9.0.3.16616.20160408.R5"
        assert metadata["active_mass_mg"] == 0.0
        assert metadata["creator"] == "admin"
        assert metadata["sn"] == ""
        assert metadata["remarks"] == ""
        assert metadata["start_time"] == "2016-04-19T12:21:01.022+00:00"
        assert metadata["stop_time"] == "2016-04-19T13:21:03.143+00:00"
        assert metadata["UNKNOWN_19"] == 2147082270
        assert metadata["test_id"] == 61
        assert metadata["num_datapoints"] == 15567
        assert metadata["guid"] == "38D34B02BDED1948A70DB98814D67B22"
        assert metadata["guid2"] == "38D34B02BDED1948A70DB98814D67B22"
        assert metadata["device_ip"] == "192.168.3.98"
        assert metadata["server_ip"] == "192.168.3.110"
        assert metadata["UNKNOWN_5"] == 1
        assert metadata["UNKNOWN_10"] == "武工专用"
        assert "UNKNOWN_11" not in metadata
        assert "UNKNOWN_12" not in metadata
        assert metadata["UNKNOWN_13"] == "武工专用"

    def test_nda130_sintef(self) -> None:
        """Test reading metadata from a BTS9.1.5 nda_version 130 file."""
        current_folder = Path(__file__).parent
        test_file = current_folder / "test_data" / "SINTEF__G20M7_BTS91.nda.zip"
        metadata = _read_metadata(test_file)
        assert metadata["nda_version"] == 130
        assert metadata["bts_version"] == "9.1.5.7.20250527.R5"
        assert metadata["creator"] == "admin"
        assert metadata["sn"] == "2025-12"
        assert metadata["remarks"] == "C30 Charge"
        assert metadata["start_step_id"] == 1
        assert metadata["start_time"] == "2025-12-22T08:44:24.914+00:00"
        assert metadata["stop_time"] == "2025-12-24T09:33:13.562+00:00"
        assert metadata["UNKNOWN_14"] == "Google Pixel 10"
        assert metadata["server_ip"] == "127.0.0.1"
        assert metadata["test_id"] == 26
        assert metadata["num_datapoints"] == 17587
        assert metadata["UNKNOWN_16"] == 1267237200
        assert metadata["UNKNOWN_17"] == 1
        assert metadata["UNKNOWN_18"] == "1b 30 02 10 10 02 30 1b 81 02 01 02 01 01 00 00 00 00 00 00 00"
        assert metadata["device_ip"] == "192.168.1.250"
        assert metadata["hostname"] == "SINTEFPC10925"

    def test_nda130_testfile(self) -> None:
        """Test reading metadata from another BTS9.1.5 nda_version 130 file."""
        current_folder = Path(__file__).parent
        test_file = current_folder / "test_data" / "TestFile.nda.zip"
        metadata = _read_metadata(test_file)
        assert metadata["nda_version"] == 130
        assert metadata["bts_version"] == "9.1.5.7.20240403.R5"
        assert metadata["creator"] == "admin"
        assert metadata["sn"] == "P_DCH-CH"
        assert metadata["remarks"] == "cell16"
        assert metadata["start_time"] == "2024-05-27T08:02:48.782+00:00"
        assert metadata["stop_time"] == "2024-05-28T08:56:24.076+00:00"
        assert metadata["start_step_id"] == 1
        assert metadata["UNKNOWN_14"] == "VAPCELL_F60_6000"
        assert metadata["server_ip"] == ""
        assert metadata["test_id"] == 43
        assert metadata["num_datapoints"] == 6670
        assert metadata["UNKNOWN_16"] == 1269091200
        assert metadata["UNKNOWN_17"] == 1
        assert metadata["UNKNOWN_18"] == "10 93 97 12 12 97 93 10 81 08 07 08 07 01 00 00 00 00 00 00 00"
        assert metadata["device_ip"] == "192.168.1.250"
        assert metadata["hostname"] == "LENOVO-L0X1245"
