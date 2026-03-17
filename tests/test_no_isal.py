"""Tests ndax reading without isal dependency."""

import builtins
import sys
from pathlib import Path

import pytest

import fastnda


@pytest.fixture
def no_isal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate isal not installed."""
    del_modules = ["isal"]
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


def test_read_no_isal(no_isal) -> None:  # noqa: ANN001, ARG001
    """Test reading ndax files without isal, using zlib from stdlib."""
    with pytest.raises(ModuleNotFoundError):
        import isal  # noqa: F401, PLC0415
    test_file = Path(__file__).parent / "test_data" / "nw4-120-1-6-53.ndax"
    fastnda.read(test_file)
