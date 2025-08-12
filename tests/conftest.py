"""Default to tests/test_data, allow users to change test data folder."""

import re
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command line option to select test data directory."""
    parser.addoption(
        "--data-dir",
        action="store",
        default="tests/test_data",
        help=(
            "Path to the test data directory. "
            "Data directory should contain .nda/.ndax and .parquet files. "
            ".parquet files can be generated with btsda_csv_to_parquet(...). "
            "Files should have same name other than extension. "
            "(default: tests/test_data)"
        ),
    )


@pytest.fixture
def data_dir(request: pytest.FixtureRequest) -> Path:
    """Add test data directory fixture."""
    path = Path(request.config.getoption("--data-dir"))
    if not path.exists():
        pytest.fail(f"Test data directory does not exist: {path}")
    return path


def _remove_extension(filename: str) -> str:
    """Remove file extension from filename.

    Required as filepath.stem would only remove zip from .ndax.zip.
    """
    return re.sub(r"\.(ndax|nda|parquet|nda\.zip)$", "", filename, flags=re.IGNORECASE)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Find matching pairs of files."""
    data_dir = Path(metafunc.config.getoption("--data-dir"))

    if "file_pair" in metafunc.fixturenames:
        # Get data directory
        data_dir = Path(metafunc.config.getoption("--data-dir"))

        # Find all pairs of .ndax and .parquet files with matching stem
        inputs = {
            _remove_extension(f.name): f
            for f in (
                *data_dir.rglob("*.ndax"),  # already zipped
                *data_dir.rglob("*.nda"),
                *data_dir.rglob("*.nda.zip"),  # can save a lot of space
            )
        }
        outputs = {_remove_extension(f.name): f for f in data_dir.rglob("*.parquet")}

        # Create list of (input_file, output_file) tuples
        file_pairs = [(inputs[stem], outputs[stem]) for stem in inputs.keys() & outputs.keys()]

        metafunc.parametrize("file_pair", file_pairs, ids=[f.stem for f, _ in file_pairs])
