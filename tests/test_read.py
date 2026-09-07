# Copyright © 2026, Empa.
"""Test read functionality."""

import importlib
import re
from collections.abc import Callable, Generator
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

import fastnda
from fastnda._ndc.ndc import _CONFIRMED_NDC_KEYS, _NDC_READERS
from fastnda.dicts import STEP_TYPE_MAP
from fastnda.nda import _CONFIRMED_READER_NAMES, _NDA_READERS
from fastnda.utils import _generate_cycle_number


@pytest.fixture(scope="module")
def parsed_data(file_pair: tuple[Path, Path | None]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read in the data for each file pair ONCE."""
    test_file, ref_file = file_pair
    if ref_file is None:
        pytest.skip("No reference Parquet file for this input.")
    if test_file.suffix == ".zip":  # Is nda or ndax zipped
        with TemporaryDirectory() as tmp_dir, ZipFile(test_file, "r") as zip_test:
            # unzip file to a temp location and read
            zip_test.extractall(tmp_dir)
            test_file = Path(tmp_dir) / test_file.stem
            df = fastnda.read(test_file, cycle_mode="raw")
    else:
        df = fastnda.read(test_file, cycle_mode="raw")
    df_ref = pl.read_parquet(ref_file)
    return df, df_ref


REV_STEP_TYPE_MAP = {v: k for k, v in STEP_TYPE_MAP.items()}


class TestRead:
    """Compared parsed data to reference from BTSDA."""

    def test_generate_cycle_number(self, test_file: Path) -> None:
        """Test generating cycle numbers on just one file."""
        df1 = fastnda.read(test_file, cycle_mode="raw")
        df2 = fastnda.read(test_file, cycle_mode="chg")
        df1 = df1.with_columns(
            pl.col("step_type").cast(pl.Utf8).replace_strict(REV_STEP_TYPE_MAP, return_dtype=pl.Int32)
        )
        df1 = _generate_cycle_number(df1, "chg")
        assert_series_equal(df1["cycle_count"], df2["cycle_count"])

    def test_wrong_filetype(self) -> None:
        """Test using the wrong file."""
        test_file = Path(r"wrong_file.csv")
        with pytest.raises(ValueError):
            fastnda.read(test_file)

    def test_file_columns(self, parsed_data: tuple) -> None:
        """Check that the expected columns are in the DataFrames."""
        df, df_ref = parsed_data
        df_columns = {
            "index",
            "voltage_V",
            "current_mA",
            "unix_time_s",
            "step_time_s",
            "total_time_s",
            "cycle_count",
            "step_count",
            "step_index",
            "step_type",
            "capacity_mAh",
        }
        # Some old formats never recorded energy - dropped from the reference parquet
        if "Energy(mWs)" in df_ref.columns:
            df_columns.add("energy_mWh")
        assert all(col in df.columns for col in df_columns), (
            f"Missing columns in DataFrame: {df_columns - set(df.columns)}"
        )
        df_ref_columns = {
            "Time",
            "Total Time",
            "Step Index",
            "Step Count",
            "Voltage(mV)",
            "Current(uA)",
            "Capacity(mAs)",
        }
        assert all(col in df_ref.columns for col in df_ref_columns), (
            f"Missing columns in reference DataFrame: {df_ref_columns - set(df_ref.columns)}"
        )
        # Should not be any nulls
        assert any((df.null_count() == 0).row(0)), "DataFrame contains nulls"

    def test_step(self, parsed_data: tuple) -> None:
        """Check that the step column is equal."""
        df, df_ref = parsed_data
        assert_series_equal(
            df["step_index"],
            df_ref["Step Index"],
            check_names=False,
        )
        assert_series_equal(
            df["step_count"],
            df_ref["Step Count"],
            check_names=False,
        )
        # step_type is enum - faster, but not directly comparable to categorical
        # Need to cast both to same dtype, and replace spaces in ref
        # Neware is inconsistent with 'Dchg' and 'DChg' in column names
        assert_series_equal(
            df["step_type"].cast(pl.String),
            (
                df_ref["Step Type"]
                .cast(pl.String)
                .str.replace_all(" ", "_")
                .str.replace_all("Dchg", "DChg")
                .str.replace_all("Pulse_Step", "Pulse")
                .str.replace_all("RI_Chg", "CR_Chg")
                .str.replace_all("RI_DChg", "CR_DChg")
            ),
            check_names=False,
        )

    def test_cycle(self, parsed_data: tuple, note: Callable[[str], None]) -> None:
        """Cycle should be exact when using raw cycle_mode."""
        df, df_ref = parsed_data
        # If the default is wrong, check if cycle_mode auto is correct
        if not (df["cycle_count"] == df_ref["Cycle Index"]).all():
            df2 = df.with_columns(
                pl.col("step_type").cast(pl.Utf8).replace_strict(REV_STEP_TYPE_MAP, return_dtype=pl.Int32)
            )
            df2 = _generate_cycle_number(df2, "auto")
            assert_series_equal(
                df2["cycle_count"],
                df_ref["Cycle Index"],
                check_names=False,
            )
            note("Cycles do not match with 'raw' cycle_mode, only with 'auto'")

    def test_index(self, parsed_data: tuple) -> None:
        """Index should be UInt32 monotonically increasing by 1."""
        df, df_ref = parsed_data
        ref_index = (
            df_ref["DataPoint"]
            if "DataPoint" in df_ref.columns
            else pl.Series("ref_index", range(1, len(df) + 1), dtype=pl.UInt32)
        )
        assert_series_equal(df["index"], ref_index, check_names=False)

    def test_step_time(self, parsed_data: tuple, note: Callable[[str], None]) -> None:
        """Step time should agree within 1 us."""
        df, df_ref = parsed_data
        if len(df) == 0 and len(df_ref) == 0:
            return
        diff = (df["step_time_s"] - df_ref["Time"]).abs()
        max_diff = None
        # BTSDA exported step time changes precision over time
        thresholds = [
            ((1e7, 1e8), 10.1),
            ((1e6, 1e7), 1.01),
            ((1800, 1e6), 0.101),
            ((0, 1800), 0.0101),
        ]
        for (time_min, time_max), threshold in thresholds:
            max_diff = diff.filter((df_ref["Time"] > time_min) & (df_ref["Time"] < time_max)).max()
            if max_diff is not None and max_diff > threshold:
                msg = f"Step time columns differ by up to {max_diff:.2e}"
                raise ValueError(msg)
        # Check earliest time diff, note if over 1 us
        if max_diff is not None and max_diff > 5e-7:
            note(f"Step time only matches within {max_diff:.2e} s")

    def test_total_time(self, parsed_data: tuple, note: Callable[[str], None]) -> None:
        """Total time should agree within 1 us."""
        df, df_ref = parsed_data
        if len(df) == 0 and len(df_ref) == 0:
            return
        diff = (df["total_time_s"] - df_ref["Total Time"]).abs()
        max_diff = None
        # BTSDA exported total time changes precision over time
        thresholds = [
            ((1e7, 1e8), 10.1),
            ((1e6, 1e7), 1.01),
            ((1800, 1e6), 0.101),
            ((0, 1800), 0.0101),
        ]
        for (time_min, time_max), threshold in thresholds:
            max_diff = diff.filter((df_ref["Total Time"] > time_min) & (df_ref["Total Time"] < time_max)).max()
            if max_diff is not None and max_diff > threshold:
                msg = f"Total time columns differ by up to {max_diff:.2e}"
                raise ValueError(msg)
        # Check earliest time diff, note if over 1 us
        if max_diff is not None and max_diff > 5e-7:
            note(f"Total time only matches within {max_diff:.2e} s")

    def test_datetime(self, parsed_data: tuple) -> None:
        """Date should agree within 1 us."""
        df, df_ref = parsed_data
        if len(df) == 0 and len(df_ref) == 0:
            return
        # Cannot cycle cells before Neware was founded in 1998
        assert df["unix_time_s"].min() > 883609200

        # Some old formats never recorded a timestamp - dropped from the reference parquet
        if "Date" not in df_ref.columns:
            # Derived from the test time, so it can only increase
            assert df["unix_time_s"].is_sorted(), "Derived unix_time_s is not monotonically increasing."
            pytest.skip("This format does not record a timestamp.")

        # Cannot compare date directly - Neware datetime is not timezone aware.
        duts = df["unix_time_s"] - df["unix_time_s"][0]
        datetime_ref = df_ref["Date"].cast(pl.Float64) / 1000
        duts_ref = datetime_ref - datetime_ref[0]
        assert_series_equal(
            duts,
            duts_ref,
            check_names=False,
            abs_tol=5e-7,
        )

    def test_voltage(self, parsed_data: tuple) -> None:
        """Voltage usually recorded to 0.1 mV, should agree within 0.05 mV."""
        df, df_ref = parsed_data
        assert_series_equal(
            df["voltage_V"],
            df_ref["Voltage(mV)"] / 1000,
            check_names=False,
            abs_tol=6e-5,
        )

    def test_current(self, parsed_data: tuple) -> None:
        """Current usually recorded to 0.1 mA, should agree within 0.05 mA."""
        df, df_ref = parsed_data
        assert_series_equal(
            df["current_mA"],
            df_ref["Current(uA)"] / 1000,
            check_names=False,
            abs_tol=0.05,
        )

    def test_capacity(self, parsed_data: tuple) -> None:
        """In some nda files, mAs are only recorded to 1 mAs = 3e-4 mAh."""
        df, df_ref = parsed_data
        # Neware capacity can be absolute for both charge and discharge
        # It can also can have negative values for discharge
        abs_diff = (df["capacity_mAh"].abs() - df_ref["Capacity(mAs)"].abs() / 3600).abs()
        rel_diff = 2 * abs_diff / (df["capacity_mAh"].abs() + df_ref["Capacity(mAs)"].abs() / 3600)
        if ((abs_diff > 3e-3) & (rel_diff > 1e-5)).any():
            # If this fails, sometimes Neware does not count negative current during charge towards the capacity
            df = df.with_columns(
                pl.col("capacity_mAh").abs().cum_max().over(pl.col("step_count")).alias("capacity_ignore_negs_mAh")
            )
            abs_diff = (df["capacity_ignore_negs_mAh"].abs() - df_ref["Capacity(mAs)"].abs() / 3600).abs()
            rel_diff = 2 * abs_diff / (df["capacity_ignore_negs_mAh"].abs() + df_ref["Capacity(mAs)"].abs() / 3600)
            if ((abs_diff > 3e-3) & (rel_diff > 1e-5)).any():
                msg = "Capacity columns are different."
                raise ValueError(msg)

    def test_energy(self, parsed_data: tuple, note: Callable[[str], None]) -> None:
        """Neware energy can be recorded 0.1 mWs, check to 3e-5 mWh."""
        df, df_ref = parsed_data
        if "Energy(mWs)" not in df_ref.columns:
            pytest.skip("This format does not record energy.")
        assert "energy_mWh" in df.columns, "Reference records energy but the DataFrame has no energy_mWh."
        # Neware capacity can be absolute for both charge and discharge
        # It can also can have negative values for discharge
        abs_diff = (df["energy_mWh"].abs() - df_ref["Energy(mWs)"].abs() / 3600).abs()
        rel_diff = 2 * abs_diff / (df["energy_mWh"].abs() + df_ref["Energy(mWs)"].abs() / 3600)
        if ((abs_diff > 1e-2) & (rel_diff > 1e-6)).any():
            # If this fails, sometimes Neware does not count negative current during charge towards the energy
            df = df.with_columns(
                pl.col("energy_mWh").abs().cum_max().over(pl.col("step_count")).alias("energy_ignore_negs_mWh")
            )
            abs_diff = (df["energy_ignore_negs_mWh"] - df_ref["Energy(mWs)"].abs() / 3600).abs()
            rel_diff = 2 * abs_diff / (df["energy_ignore_negs_mWh"] + df_ref["Energy(mWs)"].abs() / 3600)
            if ((abs_diff > 6e-3) & (rel_diff > 1e-5)).any():
                msg = "Energy columns are different."
                raise ValueError(msg)
            if ((abs_diff > 3e-4) & (rel_diff > 1e-6)).any():
                msg = f"Energy columns differ by up to {max(abs_diff):.2e} mWh (or {max(rel_diff) * 100:2g}%)."
                note(msg)

    def test_capacity_energy_sign(self, parsed_data: tuple, note: Callable[[str], None]) -> None:
        """Each capacity/energy increment should share sign with current."""
        df, df_ref = parsed_data
        columns = ["capacity_mAh"]
        if "Energy(mWs)" in df_ref.columns and "energy_mWh" in df.columns:
            columns.append("energy_mWh")
        exprs = [pl.col("step_time_s").diff().over("step_count").alias("time_diff"), pl.col("current_mA")]
        for col in columns:
            exprs += [
                pl.col(col).diff().over("step_count").alias(f"{col}_diff"),
                pl.col(col).abs().diff().over("step_count").alias(f"{col}_growth"),
            ]
        diffs = df.select(exprs).drop_nulls()
        static = diffs.filter(pl.col("time_diff") == 0)
        moving = diffs.filter(pl.col("time_diff") != 0)

        # Energy sign is not strictly accurate, but is how Neware treats it - see #77
        for col in columns:
            label = col.rsplit("_", 1)[0]
            n_static = int((static[f"{col}_growth"] > 0).sum())
            assert not n_static, f"{n_static} {label} increments accumulate while step time does not advance."
            mismatch = (moving[f"{col}_diff"].sign() != moving["current_mA"].sign()).mean()
            if mismatch:
                assert mismatch < 0.005, f"{mismatch:.3%} of {label} increments disagree in sign with current."
                if mismatch > 0.001:
                    note(f"{mismatch:.3%} of {label} increments disagree in sign with current.")

    def test_aux_cols(self, parsed_data: tuple) -> None:
        """Dataframes should have matching aux channels."""
        df, df_ref = parsed_data
        df_aux = [c for c in df.columns if c.startswith("aux")]
        df_ref_aux = [c for c in df_ref.columns if re.match(r"^[TtHV]\d+", c)]

        # Check if there are the same number of aux channels
        if len(df_aux) != len(df_ref_aux):
            # Remove empty columns in the ref
            df_ref_aux = [col for col in df_ref_aux if col in df_ref.columns and not (df_ref[col] == 0).all()]
            assert len(df_aux) == len(df_ref_aux), "Number of aux channels does not match."

        for test_col in df_aux:
            if "temp" in test_col:  # temp only recorded to 0.1 degC
                tol = 5e-2
                multiplier = 1.0
            elif "voltage" in test_col:
                tol = 1e-4  # voltage usually accurate to 0.1 mV
                multiplier = 1e-3  # ref is in mV
            else:
                tol = 1e-3
                multiplier = 1.0
            results: dict[str, float] = {}
            for ref_col in df_ref_aux:
                ref_vals = multiplier * df_ref[ref_col]
                # Aux channels do not always sample every row, nulls must match
                if (df[test_col].is_null() != ref_vals.is_null()).any():
                    continue
                mean_diff = (df[test_col] - ref_vals).abs().mean()
                if mean_diff is None:
                    continue
                results[ref_col] = mean_diff
                if mean_diff < tol:
                    break
            else:
                # raise an error
                if not results:
                    msg = f"No reference column is comparable to {test_col}, nulls do not line up with any"
                    raise ValueError(msg)
                closest = min(results, key=lambda x: results[x])
                msg = (
                    f"Could not find any column matching values of {test_col}, "
                    f"closest reference was {closest} with an average difference of {results[closest]}"
                )
                raise ValueError(msg)

    def test_bdf(self, parsed_data: tuple, file_pair: tuple[Path, Path]) -> None:
        """Test bdf column conversion."""
        df, df_ref = parsed_data
        test_file = file_pair[0]
        if test_file.suffix == ".zip":
            with TemporaryDirectory() as tmp_dir, ZipFile(test_file, "r") as zip_test:
                # unzip file to a temp location and read
                zip_test.extractall(tmp_dir)
                test_file = Path(tmp_dir) / test_file.stem
                df_bdf = fastnda.read(test_file, columns="bdf", cycle_mode="raw")
        else:
            df_bdf = fastnda.read(test_file, columns="bdf", cycle_mode="raw")

        assert "record_index" in df_bdf.columns
        assert "voltage_volt" in df_bdf.columns
        assert "current_ampere" in df_bdf.columns
        assert "unix_time_second" in df_bdf.columns
        assert "step_time_second" in df_bdf.columns
        assert "test_time_second" in df_bdf.columns
        assert "cycle_count" in df_bdf.columns
        assert "step_count" in df_bdf.columns
        assert "step_id" in df_bdf.columns
        assert "step_type" in df_bdf.columns
        assert "step_net_capacity_ah" in df_bdf.columns
        if "Energy(mWs)" in df_ref.columns:
            assert "step_net_energy_wh" in df_bdf.columns
            assert_series_equal(df["energy_mWh"], df_bdf["step_net_energy_wh"] * 1e3, check_names=False)

        assert_series_equal(df["index"], df_bdf["record_index"], check_names=False)
        assert_series_equal(df["voltage_V"], df_bdf["voltage_volt"], check_names=False)
        assert_series_equal(df["current_mA"], df_bdf["current_ampere"] * 1e3, check_names=False)
        assert_series_equal(df["unix_time_s"], df_bdf["unix_time_second"], check_names=False)
        assert_series_equal(df["step_time_s"], df_bdf["step_time_second"], check_names=False)
        assert_series_equal(df["total_time_s"], df_bdf["test_time_second"], check_names=False)
        assert_series_equal(df["cycle_count"], df_bdf["cycle_count"], check_names=False)
        assert_series_equal(df["step_count"], df_bdf["step_count"], check_names=False)
        assert_series_equal(df["step_index"], df_bdf["step_id"], check_names=False)
        assert_series_equal(df["step_type"], df_bdf["step_type"], check_names=False)
        assert_series_equal(df["capacity_mAh"], df_bdf["step_net_capacity_ah"] * 1e3, check_names=False)

        # Checking correct order of magnitude, more precise value checks in other tests
        assert_series_equal(
            df_bdf["current_ampere"],
            df_ref["Current(uA)"] * 1e-6,
            abs_tol=5e-5,
            check_names=False,
            check_dtypes=False,
        )

    def test_bdf_pref(self, parsed_data: tuple, file_pair: tuple[Path, Path]) -> None:
        """Test bdf preferred column conversion."""
        df, df_ref = parsed_data
        test_file = file_pair[0]
        if test_file.suffix == ".zip":
            with TemporaryDirectory() as tmp_dir, ZipFile(test_file, "r") as zip_test:
                # unzip file to a temp location and read
                zip_test.extractall(tmp_dir)
                test_file = Path(tmp_dir) / test_file.stem
                df_bdf = fastnda.read(test_file, columns="bdf-pref")
        else:
            df_bdf = fastnda.read(test_file, columns="bdf-pref")

        assert "Record Index / 1" in df_bdf.columns
        assert "Voltage / V" in df_bdf.columns
        assert "Current / A" in df_bdf.columns
        assert "Unix Time / s" in df_bdf.columns
        assert "Step Time / s" in df_bdf.columns
        assert "Test Time / s" in df_bdf.columns
        assert "Cycle Count / 1" in df_bdf.columns
        assert "Step Count / 1" in df_bdf.columns
        assert "Step ID" in df_bdf.columns
        assert "Step Type" in df_bdf.columns
        assert "Step Net Capacity / Ah" in df_bdf.columns
        if "Energy(mWs)" in df_ref.columns:
            assert "Step Net Energy / Wh" in df_bdf.columns
            assert_series_equal(df["energy_mWh"], df_bdf["Step Net Energy / Wh"] * 1e3, check_names=False)

        assert_series_equal(df["current_mA"], df_bdf["Current / A"] * 1e3, check_names=False)
        assert_series_equal(df["capacity_mAh"], df_bdf["Step Net Capacity / Ah"] * 1e3, check_names=False)

        # Checking correct order of magnitude, more precise value checks in other tests
        assert_series_equal(
            df_bdf["Current / A"],
            df_ref["Current(uA)"] * 1e-6,
            abs_tol=5e-5,
            check_names=False,
            check_dtypes=False,
        )

    def test_bad_column_input(self, test_file: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test bad columns input to read."""
        df1 = fastnda.read(test_file, columns="something-wrong")
        df2 = fastnda.read(test_file, columns="default")
        assert_frame_equal(df1, df2)
        assert "not understood" in caplog.text


def _skip_unless_canonical_data_dir(data_dir: Path) -> None:
    """Skip the calling test unless --data-dir points at the default tests/test_data corpus.

    Confirmed-set drift checks only make sense against the full canonical corpus - a --data-dir
    override for testing one file or a subset shouldn't make them fail (or falsely pass).
    """
    canonical_data_dir = Path(__file__).parent / "test_data"
    if data_dir.resolve() != canonical_data_dir.resolve():
        pytest.skip("--data-dir does not point at the default corpus - drift check is not meaningful here.")


def _instrument_reader_dict(module_name: str, dict_attr: str) -> Generator[set[str], None, None]:
    """Wrap every entry of module_name.dict_attr to record which reader functions get called."""
    module = importlib.import_module(module_name)
    reader_dict: dict = getattr(module, dict_attr)
    called: set[str] = set()
    original = dict(reader_dict)
    for key, reader in original.items():
        if reader is None:
            continue
        reader_name = reader.__name__

        def wrapper(*args: object, _reader: Callable = reader, _name: str = reader_name) -> object:
            called.add(_name)
            return _reader(*args)

        reader_dict[key] = wrapper
    try:
        yield called
    finally:
        reader_dict.clear()
        reader_dict.update(original)


@pytest.fixture(scope="module", autouse=True)
def nda_reader_call_tracker() -> Generator[set[str], None, None]:
    """Track which fastnda.nda reader functions get called by real data in this module."""
    yield from _instrument_reader_dict("fastnda.nda", "_NDA_READERS")


_DISTINCT_NDA_READER_NAMES = sorted({r.__name__ for r in _NDA_READERS.values() if r is not None})


class TestNdaCov:
    """Track which NDA reader functions are tested with real data."""

    @pytest.mark.parametrize("reader_name", _DISTINCT_NDA_READER_NAMES)
    def test_reader_validated(self, reader_name: str, nda_reader_call_tracker: set[str]) -> None:
        """Confirm this reader function was actually invoked by TestRead's real-data reads."""
        if reader_name not in nda_reader_call_tracker:
            pytest.xfail(f"{reader_name} was never tested with a real data file.")

    def test_confirmed_reader_names_are_correct(self, data_dir: Path, nda_reader_call_tracker: set[str]) -> None:
        """Check that _CONFIRMED_READER_NAMES matches real data.

        Skips if --data-dir is used.
        """
        _skip_unless_canonical_data_dir(data_dir)
        unbacked = _CONFIRMED_READER_NAMES - nda_reader_call_tracker
        assert not unbacked, (
            f"_CONFIRMED_READER_NAMES claims these reader functions are real-data-verified, but none "
            f"of the real-data tests in this run actually called them: {sorted(unbacked)}"
        )
        newly_verified = nda_reader_call_tracker - _CONFIRMED_READER_NAMES
        assert not newly_verified, (
            f"These reader functions are now checked by real data but not yet in "
            f"_CONFIRMED_READER_NAMES - add them so UnverifiedFormatWarning stops firing "
            f"unnecessarily: {sorted(newly_verified)}"
        )


def _instrument_ndc_readers() -> Generator[set[tuple[tuple[int, int], str]], None, None]:
    """Wrap every NDC_READERS entry to record which (key, reader name) pairs get called."""
    module = importlib.import_module("fastnda._ndc.ndc")
    reader_dict: dict = module._NDC_READERS
    called: set[tuple[tuple[int, int], str]] = set()
    original = dict(reader_dict)
    for key, reader in original.items():
        if reader is None:
            continue
        reader_name = reader.__name__

        def wrapper(
            *args: object, _reader: Callable = reader, _key: tuple[int, int] = key, _name: str = reader_name
        ) -> object:
            called.add((_key, _name))
            return _reader(*args)

        reader_dict[key] = wrapper
    try:
        yield called
    finally:
        reader_dict.clear()
        reader_dict.update(original)


@pytest.fixture(scope="module", autouse=True)
def ndc_reader_call_tracker() -> Generator[set[tuple[tuple[int, int], str]], None, None]:
    """Track which fastnda.ndax (version, filetype) keys and reader functions get called by real data."""
    yield from _instrument_ndc_readers()


_DISTINCT_NDC_READER_NAMES = sorted({r.__name__ for r in _NDC_READERS.values() if r is not None})


class TestNdcCov:
    """Track which NDC reader functions are tested with real data."""

    @pytest.mark.parametrize("reader_name", _DISTINCT_NDC_READER_NAMES)
    def test_reader_validated(
        self, reader_name: str, ndc_reader_call_tracker: set[tuple[tuple[int, int], str]]
    ) -> None:
        """Confirm this reader function was actually invoked by TestRead's real-data reads."""
        called_names = {name for _, name in ndc_reader_call_tracker}
        if reader_name not in called_names:
            pytest.xfail(f"{reader_name} was never tested with a real data file.")

    def test_confirmed_keys_are_correct(
        self, data_dir: Path, ndc_reader_call_tracker: set[tuple[tuple[int, int], str]]
    ) -> None:
        """Check that _CONFIRMED_NDC_KEYS matches real data.

        Skips if --data-dir is used.
        """
        _skip_unless_canonical_data_dir(data_dir)
        called_keys = {key for key, _ in ndc_reader_call_tracker}
        unbacked = _CONFIRMED_NDC_KEYS - called_keys
        assert not unbacked, (
            f"_CONFIRMED_NDC_KEYS claims these (version, filetype) keys are real-data-verified, "
            f"but none of the real-data tests in this run actually read a matching file: {sorted(unbacked)}"
        )
        called_keys = {key for key, _ in ndc_reader_call_tracker}
        newly_verified = called_keys - _CONFIRMED_NDC_KEYS
        assert not newly_verified, (
            f"These (version, filetype) keys are now checked with real data but not yet in "
            f"_CONFIRMED_NDC_KEYS - add them so UnverifiedFormatWarning stops firing "
            f"unnecessarily: {sorted(newly_verified)}"
        )
