"""CLI to use fastnda conversion."""

import logging
from pathlib import Path
from typing import Annotated, Literal
from zipfile import BadZipFile

import typer
from tqdm import tqdm

import fastnda

LOGGER = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

OutputFileType = Literal["csv", "parquet", "arrow", "feather", "h5", "hdf5"]

VerbosityOption = Annotated[
    int, typer.Option("--verbose", "-v", count=True, help="Increase verbosity. Use -vv for maximum detail.")
]
QuietOption = Annotated[
    int, typer.Option("--quiet", "-q", count=True, help="Decrease verbosity. Use -qq to remove progress bars.")
]

PandasOption = typer.Option(
    False,
    "--pandas",
    "-p",
    help="(For parquet, arrow, feather) save with pandas-safe column types.",
)


def require_pandas() -> None:
    """Check if pandas is installed."""
    try:
        import pandas as pd  # noqa: F401, PLC0415
        import pyarrow as pa  # noqa: F401, PLC0415
    except ImportError as e:
        msg = (
            "'pandas' and 'pyarrow' optional dependencies are not installed.\n"
            "Install extras with `pip install fastnda[extras]`"
        )
        raise RuntimeError(msg) from e


def require_tables() -> None:
    """Check if pytables is installed for hdf5."""
    try:
        import tables  # noqa: F401, PLC0415
    except ImportError as e:
        msg = "'tables' optional dependency is not installed.\nInstall extras with `pip install fastnda[extras]`"
        raise RuntimeError(msg) from e


class TqdmHandler(logging.Handler):
    """Class to handle logs while using tqdm progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        """Write log to console."""
        msg = self.format(record)
        tqdm.write(msg)


@app.callback()
def main(
    ctx: typer.Context,
    verbose: VerbosityOption = 0,
    quiet: QuietOption = 0,
) -> None:
    """CLI for converting Neware .nda/.ndax files."""
    verbosity = verbose - quiet
    if verbosity <= -2:
        log_level = logging.ERROR
    elif verbosity == -1:
        log_level = logging.CRITICAL
    elif verbosity == 0:
        log_level = logging.WARNING
    elif verbosity == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG
    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    ctx.obj = {"verbosity": verbosity}

    handler = TqdmHandler()
    handler.setFormatter(logging.Formatter("%(name)s:%(levelname)s: %(message)s"))
    root.addHandler(handler)


@app.command()
def convert(
    in_file: Path,
    out_file: Annotated[Path | None, typer.Argument()] = None,
    filetype: OutputFileType = "parquet",
    *,
    pandas: bool = PandasOption,
) -> None:
    """Convert a .nda or .ndax file to another type.

    Args:
        in_file: Path to .nda or .ndax file.
        out_file: Path to the output file.
        filetype: Type of file to convert to, e.g. csv or parquet
        pandas: Whether to save in pandas-safe format

    """
    if filetype in {"h5", "hdf5"}:
        require_pandas()
        require_tables()
    elif pandas:
        require_pandas()
    if out_file is None:
        out_file = in_file.with_suffix("." + filetype)
    _convert_with_type(in_file, out_file, filetype, pandas)


@app.command()
def batch_convert(
    ctx: typer.Context,
    in_folder: Path,
    out_folder: Annotated[Path | None, typer.Argument()] = None,
    filetype: OutputFileType = "parquet",
    *,
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Search for .nda/.ndax files in subfolders"),
    pandas: bool = PandasOption,
) -> None:
    """Convert a .nda or .ndax file to another type."""
    if filetype in {"h5", "hdf5"}:
        require_pandas()
        require_tables()
    elif pandas:
        require_pandas()

    if not in_folder.exists():
        msg = f"Folder {in_folder} does not exist."
        raise FileNotFoundError(msg)

    if not in_folder.is_dir():
        msg = f"{in_folder} is not a folder."
        raise FileNotFoundError(msg)

    if out_folder is None:
        out_folder = in_folder

    in_files = in_folder.rglob("*.nda*") if recursive else in_folder.glob("*.nda*")
    file_list = list(in_files)
    if len(file_list) == 0:
        msg = "No .nda or .ndax files found."
        if not recursive:
            msg += " To search in sub-folders use --recursive or -r."
        raise FileNotFoundError(msg)

    disable_tqdm = ctx.obj.get("verbosity", 0) <= -2
    LOGGER.info("Found %d files to convert in %s.", len(file_list), in_folder)
    for in_file in tqdm(file_list, desc="Converting files", disable=disable_tqdm):
        out_file = out_folder / in_file.relative_to(in_folder).with_suffix("." + filetype)
        out_file.parent.mkdir(exist_ok=True)
        try:
            _convert_with_type(in_file, out_file, filetype, pandas)
        except (ValueError, BadZipFile, KeyError, AttributeError):
            LOGGER.exception("Failed to convert %s.", in_file)


def _convert_with_type(in_file: Path, out_file: Path, filetype: OutputFileType, pandas: bool) -> None:
    df = fastnda.read(in_file)

    match filetype:
        case "csv":
            df.write_csv(out_file)
        case "parquet":
            if pandas:
                df.to_pandas().to_parquet(out_file)
            else:
                df.write_parquet(out_file)
        case "arrow" | "feather":
            if pandas:
                df.to_pandas().to_feather(out_file)
            else:
                df.write_ipc(out_file)
        case "h5" | "hdf5":
            df.to_pandas().to_hdf(out_file, key="data", format="table")
