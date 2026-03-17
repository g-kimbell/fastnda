"""Tests ndax reading without isal dependency."""

import sys
from pathlib import Path
from types import ModuleType

import pytest

import fastnda


@pytest.fixture
def fastnda_no_isal(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Simulate isal not being installed."""
    # Remove fastnda + isal from cache
    to_remove = [key for key in sys.modules if key in {"isal", "fastnda"} or key.startswith(("isal.", "fastnda."))]
    for module in to_remove:
        sys.modules.pop(module, None)

    # Remove isal from sys.modules
    monkeypatch.setitem(sys.modules, "isal", None)
    monkeypatch.setitem(sys.modules, "isal.isal_zlib", None)

    # Fresh import fastnda
    import fastnda  # noqa: PLC0415
    import fastnda.ndax  # noqa: PLC0415

    return fastnda


def test_ndax_with_isal() -> None:
    """Test isal dependency works is being used."""
    test_file = Path(__file__).parent / "test_data" / "nw4-120-1-6-53.ndax"
    fastnda.read(test_file)
    assert fastnda.ndax.ISAL_AVAILABLE


def test_ndax_no_isal(fastnda_no_isal: ModuleType) -> None:
    """Test importing and reading file without isal dependency."""
    test_file = Path(__file__).parent / "test_data" / "nw4-120-1-6-53.ndax"
    fastnda_no_isal.read(test_file)
    assert not fastnda_no_isal.ndax.ISAL_AVAILABLE
