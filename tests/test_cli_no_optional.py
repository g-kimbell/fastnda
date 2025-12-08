"""Tests for fastnda CLI without optional dependencies."""

import builtins
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fastnda.cli import app


@pytest.fixture
def _no_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate pandas/tables not installed."""
    del_modules = ["pandas", "tables", "pyarrow"]
    for module in del_modules:
        if module in sys.modules:
            monkeypatch.delitem(sys.modules, module)

    original_import = builtins.__import__

    def _fake_import(module: str, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        """Intercept imports."""
        if module in del_modules:
            msg = f"No module named '{module}'"
            raise ModuleNotFoundError(msg)
        return original_import(module, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


class TestCliNoExtras:
    """Test CLI without extras installed."""

    runner = CliRunner()
    current_folder = current_dir = Path(__file__).parent
    test_file = current_folder / "test_data" / "21_10_7_85.ndax"

    @pytest.mark.usefixtures("_no_extras")
    def test_convert_hdf5(self, tmp_path: Path) -> None:
        """Converting HDF5 without extras raises error."""
        output = tmp_path / self.test_file.with_suffix(".h5").name
        result = self.runner.invoke(
            app,
            [
                "convert",
                str(self.test_file),
                str(output),
                "--format=hdf5",
            ],
        )
        assert result.exit_code == 1
        assert not output.exists()
        assert "pip install fastnda[extras]" in str(result.exception)

    @pytest.mark.usefixtures("_no_extras")
    def test_convert_pandas_parquet(self, tmp_path: Path) -> None:
        """Converting pandas-safe parquet without extras raises error."""
        output = tmp_path / self.test_file.with_suffix(".parquet").name
        result = self.runner.invoke(
            app,
            ["convert", str(self.test_file), str(output), "--format=parquet", "--pandas"],
        )
        assert result.exit_code == 1
        assert "pip install fastnda[extras]" in str(result.exception)

    @pytest.mark.usefixtures("_no_extras")
    def test_convert_pandas_arrow(self, tmp_path: Path) -> None:
        """Converting pandas-safe arrow without extras raises error."""
        output = tmp_path / self.test_file.with_suffix(".arrow").name
        result = self.runner.invoke(
            app,
            ["convert", str(self.test_file), str(output), "--format=arrow", "--pandas"],
        )
        assert result.exit_code == 1
        assert "pip install fastnda[extras]" in str(result.exception)

    @pytest.mark.usefixtures("_no_extras")
    def test_batch_convert_pandas_parquet(self, tmp_path: Path) -> None:
        """Batch converting pandas-safe parquet without extras raises error."""
        result = self.runner.invoke(
            app,
            ["batch-convert", str(tmp_path), "--format=parquet", "--pandas"],
        )
        assert result.exit_code == 1
        assert "pip install fastnda[extras]" in str(result.exception)

    @pytest.mark.usefixtures("_no_extras")
    def test_convert_parquet(self, tmp_path: Path) -> None:
        """Converting polars-style parquet without extras works."""
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

        import polars as pl  # noqa: PLC0415
        from polars.testing import assert_frame_equal  # noqa: PLC0415

        import fastnda  # noqa: PLC0415

        df1 = fastnda.read(self.test_file)
        df2 = pl.read_parquet(output)
        assert_frame_equal(df1, df2)

    @pytest.mark.usefixtures("_no_extras")
    def test_convert_arrow(self, tmp_path: Path) -> None:
        """Converting polars-style arrow without extras works."""
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

        import polars as pl  # noqa: PLC0415
        from polars.testing import assert_frame_equal  # noqa: PLC0415

        import fastnda  # noqa: PLC0415

        df1 = fastnda.read(self.test_file)
        df2 = pl.read_ipc(output)
        assert_frame_equal(df1, df2)
