# Copyright © 2026, Empa.
"""Module to read Neware NDAX files."""

import logging
import re
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from defusedxml import ElementTree

from fastnda._ndc import read_ndc
from fastnda.dicts import AUX_CHL_MAP, AUX_CHL_SCALE_MAP
from fastnda.utils import UnverifiedFormatWarning

try:
    import zlib

    from isal import isal_zlib

    zlib.decompress = isal_zlib.decompress
    zlib.decompressobj = isal_zlib.decompressobj
    ISAL_AVAILABLE = True
except ImportError:
    ISAL_AVAILABLE = False

logger = logging.getLogger(__name__)


def read_ndax(file: str | Path) -> pl.DataFrame:
    """Read data from a Neware .ndax zipped file.

    Args:
        file: Path to .ndax file to read

    Returns:
        DataFrame containing all records in the file

    """
    with zipfile.ZipFile(str(file)) as zf:
        # Get auxiliary channel files and info
        aux_ch_dict = _find_auxiliary_channels(zf)

        # Extract and parse all of the .ndc files into dataframes in parallel
        files_to_read = ["data.ndc", "data_runInfo.ndc", "data_step.ndc", *aux_ch_dict.keys()]
        dfs = _read_ndc_files(zf, files_to_read)

    # Main data (voltage, current) is always called data.ndc
    df = dfs["data.ndc"]

    # 'runInfo' contains times, capacities, energies, and needs to be forward-filled/interpolated
    if "data_runInfo.ndc" in dfs:
        df = df.join(dfs["data_runInfo.ndc"], how="left", on="index")
        df = _data_interpolation(df)

        # 'step' contains cycle count, step index, step_type for each step
        if "data_step.ndc" in dfs:
            df = df.join(dfs["data_step.ndc"], how="left", on="step_count")

    # Merge the aux data if it exists
    for i, (f, aux_dict) in enumerate(aux_ch_dict.items()):
        aux_df = dfs.get(f)
        if aux_df is not None:
            # Get aux ID, use -i if not present to avoid conflicts
            aux_id = aux_dict.get("AuxID", -i)

            # If ? column exists, rename name by ChlType (T, t, H), scaling the value if needed
            if "?" in aux_df.columns and aux_dict.get("ChlType") in AUX_CHL_MAP:
                chltype = aux_dict["ChlType"]
                col = AUX_CHL_MAP[chltype]
                scale = AUX_CHL_SCALE_MAP.get(chltype, 1)
                if scale != 1:
                    aux_df = aux_df.with_columns(pl.col("?") * scale)
                aux_df = aux_df.rename({"?": f"aux{aux_id}_{col}"})
            else:  # Otherwise just append aux ID to column names
                aux_df = aux_df.rename({col: f"aux{aux_id}_{col}" for col in aux_df.columns if col != "index"})
            if len(df) == len(aux_df):
                df = pl.concat([df, aux_df.drop("index")], how="horizontal")
            else:
                df = df.join(aux_df, how="left", on="index")

    return df


def _extract_andbytes_to_df(zf: zipfile.ZipFile, filename: str) -> tuple[str, pl.DataFrame | None]:
    """Extract .ndc from a zipfile and reads it into a DataFrame."""
    if filename in zf.namelist():
        buf = zf.read(filename)
        return filename, read_ndc(buf)
    return filename, None


def _read_ndc_files(zf: zipfile.ZipFile, files_to_read: list[str]) -> dict[str, pl.DataFrame]:
    """Parallel read several ndc files from an open ndax zip.

    Capture any unverified format warnings and collapse into a single warning.

    Args:
        zf: Open ndax zipfile.
        files_to_read: Member filenames to extract and parse.

    Returns:
        Filename -> parsed DataFrame, for members that exist in the zip.

    """
    dfs = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UnverifiedFormatWarning)
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(_extract_andbytes_to_df, zf, fname): fname for fname in files_to_read}
            for future in as_completed(futures):
                fname, df = future.result()
                if df is not None:
                    dfs[fname] = df

    for w in caught:
        if not issubclass(w.category, UnverifiedFormatWarning):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    unverified_messages = sorted({str(w.message) for w in caught if issubclass(w.category, UnverifiedFormatWarning)})
    if unverified_messages:
        warnings.warn("\n".join(unverified_messages), UnverifiedFormatWarning, stacklevel=3)

    return dfs


def read_ndax_metadata(file: str | Path) -> dict[str, str | float]:
    """Read metadata from VersionInfo.xml and Step.xml in a Neware .ndax file."""
    import xmltodict

    metadata = {}
    with zipfile.ZipFile(str(file)) as zf:
        xml_files = [f for f in zf.namelist() if f.endswith(".xml")]
        for xml_file in xml_files:
            name = xml_file.split("/")[-1].split(".")[0]
            xml_tree = ElementTree.fromstring(zf.read(xml_file).decode(errors="ignore")).find("config")
            metadata[name] = xmltodict.parse(ElementTree.tostring(xml_tree).decode(), attr_prefix="")["config"]
    return metadata


def _find_auxiliary_channels(zf: zipfile.ZipFile) -> dict[str, dict]:
    """Find all auxiliary channel files.

    Args:
        zf: open zipfile (ndax)

    Returns:
        dict: keys = filenames, values = dict of attributes of aux channel

    """
    # Auxiliary files files need to be matched to entries in TestInfo.xml
    # Sort by the numbers in the filename, assume same order in TestInfo.xml
    aux_data = []
    for f in zf.namelist():
        m = re.search(r"data_AUX_(\d+)_(\d+)_(\d+)\.ndc", f)
        if m:
            aux_data.append((f, list(map(int, m.groups()))))
        else:
            m = re.search(r".*_(\d+)\.ndc", f)
            if m:
                aux_data.append((f, [int(m.group(1)), 0, 0]))

    # Sort by the three integers
    aux_data.sort(key=lambda x: x[1])
    aux_filenames = [f for f, _ in aux_data]

    # Find all auxiliary channel dicts in TestInfo.xml
    aux_dicts: list[dict] = []
    if aux_filenames:
        try:
            step = zf.read("TestInfo.xml").decode("gb2312")
            test_info = ElementTree.fromstring(step).find("config/TestInfo")
            if test_info is not None:
                aux_dicts.extend(
                    {k: int(v) if v.isdigit() else v for k, v in child.attrib.items()}
                    for child in test_info
                    if "aux" in child.tag.lower()
                )
        except Exception:
            logger.exception("Aux files found, but could not read TestInfo.xml!")

    # ASSUME channel files are in the same order as TestInfo.xml, map filenames to dicts
    if len(aux_dicts) == len(aux_filenames):
        return dict(zip(aux_filenames, aux_dicts, strict=True))
    logger.critical("Found a different number of aux channels in files and TestInfo.xml!")
    return {}


def _data_interpolation(df: pl.DataFrame) -> pl.DataFrame:
    """Forward fill and interpolate missing data in the DataFrame."""
    # Get time by forward filling differences
    df = (
        df.with_columns(
            [
                pl.col("step_time_s").is_null().alias("nan_mask"),
                pl.col("step_time_s").is_not_null().cum_sum().shift(1).fill_null(0).alias("group_idx"),
                pl.col(
                    "dt",
                    "step_count",
                    "step_time_s",
                    "unix_time_s",
                    "charge_capacity_mAh",
                    "discharge_capacity_mAh",
                    "charge_energy_mWh",
                    "discharge_energy_mWh",
                ).fill_null(strategy="forward"),
            ]
        )
        .with_columns(
            [
                (pl.col("dt").cum_sum().over("group_idx") * (pl.col("nan_mask"))).alias("cdt"),
                ((pl.col("dt") * pl.col("current_mA") / 3600).cum_sum().over("group_idx") * pl.col("nan_mask")).alias(
                    "inc_capacity"
                ),
                (
                    (pl.col("dt") * pl.col("voltage_V") * pl.col("current_mA") / 3600).cum_sum().over("group_idx")
                    * pl.col("nan_mask")
                ).alias("inc_energy"),
            ]
        )
        .with_columns(
            [
                (pl.col("step_time_s") + pl.col("cdt")).alias("step_time_s"),
                (pl.col("unix_time_s") + pl.col("cdt")).alias("unix_time_s"),
                (pl.col("charge_capacity_mAh").abs() + pl.col("inc_capacity").clip(lower_bound=0)).alias(
                    "charge_capacity_mAh"
                ),
                (pl.col("discharge_capacity_mAh").abs() - pl.col("inc_capacity").clip(upper_bound=0)).alias(
                    "discharge_capacity_mAh"
                ),
                (pl.col("charge_energy_mWh").abs() + pl.col("inc_energy").clip(lower_bound=0)).alias(
                    "charge_energy_mWh"
                ),
                (pl.col("discharge_energy_mWh").abs() - pl.col("inc_energy").clip(upper_bound=0)).alias(
                    "discharge_energy_mWh"
                ),
            ]
        )
        .drop(["nan_mask", "group_idx", "cdt", "inc_capacity", "inc_energy", "dt"])
    )

    # Sanity checks
    if (df["unix_time_s"].diff() < 0).any():
        logger.warning(
            "IMPORTANT: This ndax has negative jumps in the 'unix_time_s' column! "
            "Use the 'total_time_s' column for analysis.",
        )

    return df
