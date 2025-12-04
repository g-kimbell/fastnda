"""Tests for fastnda CLI with optional dependencies."""

from pathlib import Path

import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

import fastnda
from fastnda.cli import app


class TestCliWithOptionalDeps:
    """Test CLI with optional dependencies."""

    runner = CliRunner()
    current_folder = current_dir = Path(__file__).parent
    test_file = current_folder / "test_data" / "21_10_7_85.ndax"
    ref_df = fastnda.read(test_file)

    def test_convert_hdf5(self, tmp_path: Path) -> None:
        """Converting HDF5 with pandas."""
        output = tmp_path / self.test_file.with_suffix(".h5").name
        result = self.runner.invoke(
            app,
            [
                "convert",
                str(self.test_file),
                str(output),
                "--filetype=hdf5",
            ],
        )
        assert result.exit_code == 0
        assert output.exists()
        df = pd.read_hdf(output, key="data")
        assert_frame_equal(
            pl.DataFrame(pl.from_pandas(df)),
            self.ref_df.with_columns(pl.col("step_type").cast(pl.Categorical)),
        )

    def test_convert_parquet_pandas(self, tmp_path: Path) -> None:
        """Converting pandas-safe parquet."""
        output = tmp_path / self.test_file.with_suffix(".parquet").name
        result = self.runner.invoke(
            app,
            ["convert", str(self.test_file), str(output), "--filetype=parquet", "--pandas"],
        )
        assert result.exit_code == 0
        assert output.exists()
        df = pl.read_parquet(output)
        assert_frame_equal(
            df,
            self.ref_df.with_columns(pl.col("step_type").cast(pl.Categorical)),
        )

    def test_convert_arrow_pandas(self, tmp_path: Path) -> None:
        """Converting pandas-safe arrow."""
        output = tmp_path / self.test_file.with_suffix(".arrow").name
        result = self.runner.invoke(
            app,
            ["convert", str(self.test_file), str(output), "--filetype=arrow", "--pandas"],
        )
        assert result.exit_code == 0
        assert output.exists()
        df = pl.read_ipc(output)
        assert_frame_equal(
            df,
            self.ref_df.with_columns(pl.col("step_type").cast(pl.Categorical)),
        )

    def test_convert_csv(self, tmp_path: Path) -> None:
        """Converting csv."""
        output = tmp_path / self.test_file.with_suffix(".csv").name
        result = self.runner.invoke(
            app,
            [
                "convert",
                str(self.test_file),
                str(output),
                "--filetype=csv",
            ],
        )
        assert result.exit_code == 0
        assert output.exists()
        df = pl.read_csv(output)
        assert_frame_equal(
            df,
            self.ref_df.with_columns(pl.col("step_type").cast(pl.Utf8)),
            check_dtypes=False,
        )

    def test_convert_parquet(self, tmp_path: Path) -> None:
        """Converting polars-style parquet."""
        output = tmp_path / self.test_file.with_suffix(".parquet").name
        result = self.runner.invoke(
            app,
            [
                "convert",
                str(self.test_file),
                str(output),
                "--filetype=parquet",
            ],
        )
        assert result.exit_code == 0
        assert output.exists()
        df = pl.read_parquet(output)
        assert_frame_equal(df, self.ref_df)
