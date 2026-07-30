# Copyright © 2026, Empa.
"""Default to tests/test_data, allow users to change test data folder."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.terminal import TerminalReporter


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


@pytest.fixture(scope="session")
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
        file_pairs = [(inputs[stem], outputs.get(stem)) for stem in inputs]

        metafunc.parametrize(
            "file_pair",
            file_pairs,
            ids=[f.stem for f, _ in file_pairs],
            indirect=True,
            scope="module",
        )


@pytest.fixture(scope="module")
def file_pair(request: pytest.FixtureRequest) -> tuple:
    """Return one file_pair from the request as a fixture."""
    return request.param


@pytest.fixture
def test_file() -> Path:
    """Return a single .ndax file, for tests that do not need depend on file internals."""
    return Path(__file__).parent / "test_data" / "nw4-120-1-6-53.ndax"


@pytest.fixture
def note(request: pytest.FixtureRequest) -> Callable[[str], None]:
    """Attach a caveat to a passing test, reported as NOTE instead of a Python warning."""

    def _note(message: str) -> None:
        request.node.user_properties.append(("note", message))

    return _note


def pytest_report_teststatus(
    report: "pytest.CollectReport | pytest.TestReport",
) -> "tuple[str, str, tuple[str, dict[str, bool]]] | None":
    """Report passing tests that called note() as NOTE, in their own 'notes' category."""
    if (
        isinstance(report, pytest.TestReport)
        and report.when == "call"
        and report.passed
        and any(name == "note" for name, _ in report.user_properties)
    ):
        return "notes", "n", ("NOTE", {"yellow": True})
    return None


def pytest_terminal_summary(terminalreporter: "TerminalReporter") -> None:
    """List NOTE messages in their own summary section, like the xfail summary."""
    reports = terminalreporter.stats.get("notes", [])
    if not reports:
        return
    terminalreporter.write_sep("=", "passed with notes", yellow=True)
    for report in reports:
        for name, message in report.user_properties:
            if name == "note":
                terminalreporter.write("NOTE", yellow=True)
                terminalreporter.write_line(f" {report.nodeid} - {message}")
