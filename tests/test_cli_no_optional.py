"""Tests for fastnda CLI without optional dependencies."""

import builtins
import sys
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from fastnda.cli import app


@pytest.fixture
def _no_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate pandas/tables not installed."""
    del_modules = ["pandas", "tables"]
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


class TestCliNoPandas:
    """Test CLI without pandas installed."""

    runner = CliRunner()
    current_folder = current_dir = Path(__file__).parent
    test_file = current_folder / "test_data" / "21_10_7_85.ndax"

    @pytest.mark.usefixtures("_no_pandas")
    def test_convert_hdf5(self, tmp_path: Path) -> None:
        """Converting HDF5 without pandas raises error."""
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
        assert result.exit_code == 1
        assert not output.exists()
        assert "pip install pandas" in str(result.exception)

    @pytest.mark.usefixtures("_no_pandas")
    def test_convert_parquet_pandas(self, tmp_path: Path) -> None:
        """Converting pandas-safe parquet without pandas raises error."""
        output = tmp_path / self.test_file.with_suffix(".parquet").name
        result = self.runner.invoke(
            app,
            ["convert", str(self.test_file), str(output), "--filetype=parquet", "--pandas"],
        )
        assert result.exit_code == 1
        assert "pip install pandas" in str(result.exception)

    @pytest.mark.usefixtures("_no_pandas")
    def test_convert_arrow_pandas(self, tmp_path: Path) -> None:
        """Converting pandas-safe arrow without pandas raises error."""
        output = tmp_path / self.test_file.with_suffix(".arrow").name
        result = self.runner.invoke(
            app,
            ["convert", str(self.test_file), str(output), "--filetype=arrow", "--pandas"],
        )
        assert result.exit_code == 1
        assert "pip install pandas" in str(result.exception)

    @pytest.mark.usefixtures("_no_pandas")
    def test_convert_parquet(self, tmp_path: Path) -> None:
        """Converting polars-style parquet without pandas works."""
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
        import fastnda  # noqa: PLC0415

        df1 = fastnda.read(self.test_file)
        df2 = pl.read_parquet(output)
        assert_frame_equal(df1, df2)
