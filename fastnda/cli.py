"""CLI to use fastnda conversion."""

import logging
from pathlib import Path
from typing import Annotated, Literal

import typer
from tqdm import tqdm

import fastnda

LOGGER = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

OutputFileType = Literal["parquet", "csv"]

VerbosityOption = Annotated[
    int, typer.Option("--verbose", "-v", count=True, help="Increase verbosity. Use -vv for maximum detail.")
]
QuietOption = Annotated[
    int, typer.Option("--quiet", "-q", count=True, help="Decrease verbosity. Use -qq to remove progress bars.")
]


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
) -> None:
    """Convert a .nda or .ndax file to another type.

    Args:
        in_file: Path to .nda or .ndax file.
        out_file: Path to the output file.
        filetype: Type of file to convert to, e.g. csv or parquet

    """
    if out_file is None:
        out_file = in_file.with_suffix("." + filetype)
    _convert_with_type(in_file, out_file, filetype)


@app.command()
def batch_convert(
    ctx: typer.Context,
    in_folder: Path,
    out_folder: Annotated[Path | None, typer.Argument()] = None,
    filetype: OutputFileType = "parquet",
    *,
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Search for .nda/.ndax files in subfolders"),
) -> None:
    """Convert a .nda or .ndax file to another type."""
    if out_folder is None:
        out_folder = in_folder

    in_files = in_folder.rglob("*.nda*") if recursive else in_folder.glob("*.nda*")
    file_list = list(in_files)
    disable_tqdm = ctx.obj.get("verbosity", 0) <= -2
    LOGGER.info("Found %d files to convert in %s.", len(file_list), in_folder)
    for in_file in tqdm(file_list, desc="Converting files", disable=disable_tqdm):
        out_file = out_folder / in_file.relative_to(in_folder).with_suffix("." + filetype)
        out_file.parent.mkdir(exist_ok=True)
        try:
            _convert_with_type(in_file, out_file, filetype)
        except Exception:
            LOGGER.exception("Failed to convert %s.", in_file)


def _convert_with_type(in_file: Path, out_file: Path, filetype: OutputFileType) -> None:
    df = fastnda.read(in_file)

    match filetype:
        case "parquet":
            df.write_parquet(out_file)
        case "csv":
            df.write_csv(out_file)
        case _:
            msg = f"Cannot write to file type {filetype}"
            raise ValueError(msg)
