"""Test btsda module."""

from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
from polars.testing import assert_frame_equal

from fastnda.btsda import _time_str_to_float, btsda_csv_to_parquet


class TestBTSDA:
    """Test module for creating test data from BTSDA."""

    def test_time_str_to_float(self) -> None:
        """Test converting time strings to floats."""
        assert _time_str_to_float("001:23:45.1") == 1 * 3600 + 23 * 60 + 45.1
        assert _time_str_to_float("0:00:00.0000") == 0
        assert _time_str_to_float("0:02:30.0000") == 2 * 60 + 30
        assert _time_str_to_float("12:34:56.7890") == 12 * 3600 + 34 * 60 + 56.7890

    def test_btsda_csv_to_parquet(self) -> None:
        """Test converting BTSDA CSV to Parquet."""
        current_dir = Path(__file__).parent
        csv_path = current_dir / "test_data" / "interp-test.csv"
        ref_file = current_dir / "test_data" / "interp-test.parquet"
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "interp-test.parquet"
            btsda_csv_to_parquet(csv_path, tmp_path)
            df_test = pl.read_parquet(tmp_path)
            df_ref = pl.read_parquet(ref_file)
            assert_frame_equal(df_test, df_ref)
