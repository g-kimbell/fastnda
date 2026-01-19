"""Tests for fastnda CLI with optional dependencies."""

import json
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
    ref_df_raw_categories = fastnda.read(test_file, raw_categories=True)

    def test_convert_hdf5(self, tmp_path: Path) -> None:
        """Converting HDF5 with pandas."""
        output = tmp_path / self.test_file.with_suffix(".h5").name
        result = self.runner.invoke(
            app,
            [
                "convert",
                str(self.test_file),
                str(output),
                "--format=h5",
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
            ["convert", str(self.test_file), str(output), "--format=parquet", "--pandas"],
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
            ["convert", str(self.test_file), str(output), "--format=arrow", "--pandas"],
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
                "--format=csv",
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

    def test_convert_csv_raw_categories(self, tmp_path: Path) -> None:
        """Converting csv with raw categories."""
        output = tmp_path / self.test_file.with_suffix(".csv").name
        result = self.runner.invoke(
            app,
            [
                "convert",
                str(self.test_file),
                str(output),
                "--format=csv",
                "--raw-categories",
            ],
        )
        assert result.exit_code == 0
        assert output.exists()
        df = pl.read_csv(output)
        assert_frame_equal(df, self.ref_df_raw_categories, check_dtypes=False)

    def test_convert_parquet(self, tmp_path: Path) -> None:
        """Converting polars-style parquet."""
        output = tmp_path / self.test_file.with_suffix(".parquet").name
        result = self.runner.invoke(
            app,
            [
                "convert",
                str(self.test_file),
                str(output),
                "--format=parquet",
            ],
        )
        assert result.exit_code == 0
        assert output.exists()
        df = pl.read_parquet(output)
        assert_frame_equal(df, self.ref_df)

        # With pyarrow 23 this should be compatible with pandas
        df2 = pd.read_parquet(output)
        df2 = pl.DataFrame(df2)
        assert_frame_equal(df, df2)

    def test_convert_parquet_raw_categories(self, tmp_path: Path) -> None:
        """Converting polars-style parquet with raw categories."""
        output = tmp_path / self.test_file.with_suffix(".parquet").name
        result = self.runner.invoke(
            app,
            [
                "convert",
                str(self.test_file),
                str(output),
                "--format=parquet",
                "--raw-categories",
            ],
        )
        assert result.exit_code == 0
        assert output.exists()
        df = pl.read_parquet(output)
        assert_frame_equal(df, self.ref_df_raw_categories)

    def test_convert_arrow(self, tmp_path: Path) -> None:
        """Converting polars-style arrow."""
        output = tmp_path / self.test_file.with_suffix(".arrow").name
        result = self.runner.invoke(
            app,
            [
                "convert",
                str(self.test_file),
                str(output),
                "--format=arrow",
            ],
        )
        assert result.exit_code == 0
        assert output.exists()
        df = pl.read_ipc(output)
        assert_frame_equal(df, self.ref_df)

        # With pyarrow 23 this should be compatible with pandas
        df2 = pd.read_feather(output)
        df2 = pl.DataFrame(df2)
        assert_frame_equal(df, df2)

    def test_auto_output(self, tmp_path: Path) -> None:
        """Converting polars-style parquet without explicit output file."""
        copied_file = tmp_path / self.test_file.name
        shutil.copy(self.test_file, copied_file)
        output = tmp_path / self.test_file.with_suffix(".parquet").name
        result = self.runner.invoke(
            app,
            [
                "convert",
                str(copied_file),
                "--format=parquet",
            ],
        )
        assert result.exit_code == 0
        assert output.exists()
        df = pl.read_parquet(output)
        assert_frame_equal(df, self.ref_df)

    def test_auto_format(self, tmp_path: Path) -> None:
        """File format should come from out_file if --format is not used."""
        output = tmp_path / self.test_file.with_suffix(".parquet").name
        result = self.runner.invoke(
            app,
            [
                "convert",
                str(self.test_file),
                str(output),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()
        df = pl.read_parquet(output)
        assert_frame_equal(df, self.ref_df)

    def test_format_override(self, tmp_path: Path) -> None:
        """--format option overrides inferring from Path."""
        output = tmp_path / self.test_file.with_suffix(".parquet").name
        result = self.runner.invoke(
            app,
            [
                "convert",
                str(self.test_file),
                str(output),
                "--format=csv",
            ],
        )
        assert result.exit_code == 0
        # Has parquet extension, but is csv, not parquet format
        assert output.exists()
        with pytest.raises(pl.exceptions.ComputeError):
            df = pl.read_parquet(output)
        df = pl.read_csv(output)
        assert_frame_equal(
            df,
            self.ref_df.with_columns(pl.col("step_type").cast(pl.Utf8)),
            check_dtypes=False,
        )

    def test_default_format(self, tmp_path: Path) -> None:
        """Format is csv if no output or --format given."""
        copied_file = tmp_path / (self.test_file.stem + ".ndax")
        shutil.copy(self.test_file, copied_file)
        output = copied_file.with_suffix(".csv")
        result = self.runner.invoke(app, ["convert", str(copied_file)])
        assert result.exit_code == 0
        assert output.exists()
        df = pl.read_csv(output)
        assert_frame_equal(
            df,
            self.ref_df.with_columns(pl.col("step_type").cast(pl.Utf8)),
            check_dtypes=False,
        )

    def test_unknown_format(self, tmp_path: Path) -> None:
        """Format is csv if it cannot be inferred from path."""
        copied_file = tmp_path / (self.test_file.stem + ".ndax")
        shutil.copy(self.test_file, copied_file)
        output = copied_file.with_suffix(".bloop")
        result = self.runner.invoke(app, ["convert", str(copied_file), str(output)])
        assert result.exit_code == 0
        assert output.exists()
        df = pl.read_csv(output)
        assert_frame_equal(
            df,
            self.ref_df.with_columns(pl.col("step_type").cast(pl.Utf8)),
            check_dtypes=False,
        )

    def test_empty_batch_convert(self, tmp_path: Path) -> None:
        """Batch converting with an empty folder raises error."""
        result = self.runner.invoke(
            app,
            [
                "batch-convert",
                str(tmp_path),
                "--format=parquet",
            ],
        )
        assert result.exit_code == 1
        assert "No .nda or .ndax files found." in str(result.exception)

    def test_batch_convert(self, tmp_path: Path) -> None:
        """Basic batch converting parquet files."""
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
                "--format=parquet",
            ],
        )
        assert result.exit_code == 0
        assert output_1.exists()
        assert output_2.exists()
        df = pl.read_parquet(output_1)
        assert_frame_equal(df, self.ref_df)

    def test_recursive_batch_convert(self, tmp_path: Path) -> None:
        """Recursive batch converting requires -r or --recursive."""
        (tmp_path / "subfolder").mkdir()
        copied_file_1 = tmp_path / "subfolder" / (self.test_file.stem + "_1.ndax")
        shutil.copy(self.test_file, copied_file_1)
        output_1 = copied_file_1.with_suffix(".parquet")
        result = self.runner.invoke(
            app,
            [
                "batch-convert",
                str(tmp_path),
                "--format=parquet",
            ],
        )
        assert result.exit_code == 1
        assert not output_1.exists()
        assert "--recursive" in str(result.exception)

        result = self.runner.invoke(
            app,
            ["batch-convert", str(tmp_path), "--format=parquet", "--recursive"],
        )
        assert result.exit_code == 0
        assert output_1.exists()

    def test_hdf5_batch_convert(self, tmp_path: Path) -> None:
        """Batch convert hdf5 files."""
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
                "--format=h5",
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
        """Batch converting with bad inputs."""
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

    def test_batch_convert_bad_files(self, tmp_path: Path, caplog: pytest.FixtureRequest) -> None:  # noqa: ARG002
        """Batch convert continues even if there is a bad file."""
        copied_file_1 = tmp_path / (self.test_file.stem + "_1.nda")
        copied_file_2 = tmp_path / (self.test_file.stem + "_2.ndax")
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
                "--format=parquet",
            ],
        )
        assert result.exit_code == 0
        assert not output_1.exists()
        assert output_2.exists()

    def test_verbosity(self, tmp_path: Path, caplog: pytest.FixtureRequest) -> None:  # noqa: ARG002
        """User can change to different verbosity levels."""
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

    def test_print_metadata(self) -> None:
        """Test that printing metadata to terminal works."""
        result = self.runner.invoke(app, ["print-metadata", str(self.test_file)])
        metadata = json.loads(result.stdout)
        ref_metadata = fastnda.read_metadata(self.test_file)
        assert metadata == ref_metadata

    def test_convert_metadata(self, tmp_path: Path) -> None:
        """Test that converting metadata to json works."""
        output = tmp_path / self.test_file.with_suffix(".json").name
        self.runner.invoke(app, ["convert-metadata", str(self.test_file), str(output)])
        assert output.exists()
        with output.open("r") as f:
            metadata = json.load(f)
        ref_metadata = fastnda.read_metadata(self.test_file)
        assert metadata == ref_metadata

    def test_convert_metadata_auto_name(self, tmp_path: Path) -> None:
        """Test that converting metadata without explicit output file works."""
        copied_file = tmp_path / self.test_file.name
        shutil.copy(self.test_file, copied_file)
        output = copied_file.with_suffix(".json")
        self.runner.invoke(app, ["convert-metadata", str(copied_file)])
        assert output.exists()
        with output.open("r") as f:
            metadata = json.load(f)
        ref_metadata = fastnda.read_metadata(self.test_file)
        assert metadata == ref_metadata
