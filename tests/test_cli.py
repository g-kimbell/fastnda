"""Tests for fastnda CLI with optional dependencies."""

import logging
import shutil
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
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
                "--filetype=h5",
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

    def test_convert_arrow(self, tmp_path: Path) -> None:
        """Converting polars-style arrow."""
        output = tmp_path / self.test_file.with_suffix(".arrow").name
        result = self.runner.invoke(
            app,
            [
                "convert",
                str(self.test_file),
                str(output),
                "--filetype=arrow",
            ],
        )
        assert result.exit_code == 0
        assert output.exists()
        df = pl.read_ipc(output)
        assert_frame_equal(df, self.ref_df)

    def test_auto_output(self, tmp_path: Path) -> None:
        """Converting polars-style parquet."""
        # copy file to tmp path
        copied_file = tmp_path / self.test_file.name
        shutil.copy(self.test_file, copied_file)
        output = tmp_path / self.test_file.with_suffix(".parquet").name
        result = self.runner.invoke(
            app,
            [
                "convert",
                str(copied_file),
                "--filetype=parquet",
            ],
        )
        assert result.exit_code == 0
        assert output.exists()
        df = pl.read_parquet(output)
        assert_frame_equal(df, self.ref_df)

    def test_empty_batch_convert(self, tmp_path: Path) -> None:
        """Converting polars-style parquet."""
        result = self.runner.invoke(
            app,
            [
                "batch-convert",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 1
        assert "No .nda or .ndax files found." in str(result.exception)

    def test_batch_convert(self, tmp_path: Path) -> None:
        """Converting polars-style parquet."""
        # copy file to tmp path
        copied_file_1 = tmp_path / (self.test_file.stem + "_1.ndax")
        copied_file_2 = tmp_path / (self.test_file.stem + "_2.ndax")
        shutil.copy(self.test_file, copied_file_1)
        shutil.copy(self.test_file, copied_file_2)
        output_1 = copied_file_1.with_suffix(".parquet")
        output_2 = copied_file_2.with_suffix(".parquet")
        result = self.runner.invoke(
            app,
            [
                "batch-convert",
                str(tmp_path),
                "--filetype=parquet",
            ],
        )
        assert result.exit_code == 0
        assert output_1.exists()
        assert output_2.exists()
        df = pl.read_parquet(output_1)
        assert_frame_equal(df, self.ref_df)

    def test_recursive_batch_convert(self, tmp_path: Path) -> None:
        """Converting polars-style parquet."""
        (tmp_path / "subfolder").mkdir()
        copied_file_1 = tmp_path / "subfolder" / (self.test_file.stem + "_1.ndax")
        shutil.copy(self.test_file, copied_file_1)
        output_1 = copied_file_1.with_suffix(".parquet")
        result = self.runner.invoke(
            app,
            [
                "batch-convert",
                str(tmp_path),
                "--filetype=parquet",
            ],
        )
        assert result.exit_code == 1
        assert not output_1.exists()
        assert "--recursive" in str(result.exception)

        result = self.runner.invoke(
            app,
            ["batch-convert", str(tmp_path), "--filetype=parquet", "--recursive"],
        )
        assert result.exit_code == 0
        assert output_1.exists()

    def test_hdf5_batch_convert(self, tmp_path: Path) -> None:
        """Batch convert hdf5 files."""
        # copy file to tmp path
        copied_file_1 = tmp_path / (self.test_file.stem + "_1.ndax")
        copied_file_2 = tmp_path / (self.test_file.stem + "_2.ndax")
        shutil.copy(self.test_file, copied_file_1)
        shutil.copy(self.test_file, copied_file_2)
        output_1 = copied_file_1.with_suffix(".h5")
        output_2 = copied_file_2.with_suffix(".h5")
        result = self.runner.invoke(
            app,
            [
                "batch-convert",
                str(tmp_path),
                "--filetype=h5",
            ],
        )
        assert result.exit_code == 0
        assert output_1.exists()
        assert output_2.exists()
        df = pd.read_hdf(output_1, key="data")
        assert_frame_equal(
            pl.DataFrame(pl.from_pandas(df)),
            self.ref_df.with_columns(pl.col("step_type").cast(pl.Categorical)),
        )

    def test_batch_convert_bad_inputs(self, tmp_path: Path) -> None:
        """Test batch convert with bad inputs."""
        result = self.runner.invoke(
            app,
            [
                "batch-convert",
                str(self.test_file),
            ],
        )
        assert result.exit_code == 1
        assert "not a folder" in str(result.exception)

        result = self.runner.invoke(
            app,
            [
                "batch-convert",
                str(tmp_path / "subfolder" / "doesntexist"),
            ],
        )
        assert result.exit_code == 1
        assert "does not exist" in str(result.exception)

    def test_batch_convert_bad_files(self, tmp_path: Path, caplog: pytest.FixtureRequest) -> None:
        """Test that batch convert works even if there is a bad file."""
        copied_file_1 = tmp_path / (self.test_file.stem + "_1.nda")
        copied_file_2 = tmp_path / (self.test_file.stem + "_2.ndax")
        # just make some file
        with copied_file_1.open("w") as f:
            f.write("this is not a real ndax file")
        shutil.copy(self.test_file, copied_file_2)
        output_1 = copied_file_1.with_suffix(".parquet")
        output_2 = copied_file_2.with_suffix(".parquet")
        result = self.runner.invoke(
            app,
            [
                "batch-convert",
                str(tmp_path),
                "--filetype=parquet",
            ],
        )
        assert result.exit_code == 0
        assert not output_1.exists()
        assert output_2.exists()

    def test_verbosity(self, tmp_path: Path, caplog: pytest.FixtureRequest) -> None:  # noqa: ARG002
        """Test batch convert with bad inputs."""
        output = tmp_path / self.test_file.with_suffix(".parquet").name

        self.runner.invoke(app, ["-vv", "convert", str(self.test_file), str(output)])
        assert logging.getLogger().level == logging.DEBUG

        self.runner.invoke(app, ["-v", "convert", str(self.test_file), str(output)])
        assert logging.getLogger().level == logging.INFO

        self.runner.invoke(app, ["convert", str(self.test_file), str(output)])
        assert logging.getLogger().level == logging.WARNING

        self.runner.invoke(app, ["-q", "convert", str(self.test_file), str(output)])
        assert logging.getLogger().level == logging.CRITICAL

        self.runner.invoke(app, ["-qq", "convert", str(self.test_file), str(output)])
        assert logging.getLogger().level == logging.ERROR

        self.runner.invoke(app, ["-vvvvv", "-qqqqq", "convert", str(self.test_file), str(output)])
        assert logging.getLogger().level == logging.WARNING
