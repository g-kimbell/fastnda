"""Test reading files that do not have reference datasets.

Does not ensure correctness, just that fastnda does not crash and values are somewhat sensible.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import polars as pl
import pytest

import fastnda


@pytest.fixture
def parsed_data(file_pair: tuple[Path, Path | None]) -> pl.DataFrame:
    """Read in the data for each file ONCE."""
    test_file, ref_file = file_pair
    if ref_file is not None:
        pytest.skip("Only want files without reference.")
    if test_file.suffix == ".zip":  # Is nda or ndax zipped
        with TemporaryDirectory() as tmp_dir, ZipFile(test_file, "r") as zip_test:
            # unzip file to a temp location and read
            zip_test.extractall(tmp_dir)
            test_file = Path(tmp_dir) / test_file.stem
            df = fastnda.read(test_file, cycle_mode="raw")
    else:
        df = fastnda.read(test_file, cycle_mode="raw")
    return df


class TestReadNoRef:
    """Basic checks for files without BTSDA reference."""

    def test_file_columns(self, parsed_data: pl.DataFrame) -> None:
        """Check that the expected columns are in the DataFrames."""
        df = parsed_data
        df_columns = {
            "index",
            "voltage_V",
            "current_mA",
            "unix_time_s",
            "step_time_s",
            "cycle_count",
            "step_count",
            "step_index",
            "step_type",
            "capacity_mAh",
            "energy_mWh",
        }
        assert all(col in df.columns for col in df_columns), (
            f"Missing columns in DataFrame: {df_columns - set(df.columns)}"
        )
        # Should not be any nulls
        assert any((df.null_count() == 0).row(0)), "DataFrame contains nulls"

    def test_values(self, parsed_data: pl.DataFrame) -> None:
        """Check index increments by 1."""
        df = parsed_data
        assert all(df["index"].diff()[1:] == 1)
        assert max(df["voltage_V"]) < 5
        assert min(df["voltage_V"]) > -5
        assert max(df["current_mA"]) < 5000
        assert min(df["current_mA"]) > -5000
        assert max(df["unix_time_s"]) < 2.5e9
        assert min(df["unix_time_s"]) > 0.88e9
        assert max(df["step_time_s"]) < 31536000  # one year
        assert min(df["step_time_s"]) >= 0
