"""Test read functionality."""

import re
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import polars as pl
import pytest
from polars.testing import assert_series_equal

import fastnda
from fastnda.dicts import state_dict
from fastnda.main import _generate_cycle_number


@pytest.fixture
def parsed_data(file_pair: tuple[Path, Path]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read in the data for each file pair ONCE."""
    test_file, ref_file = file_pair

    if test_file.suffix == ".zip":  # Is nda or ndax zipped
        with TemporaryDirectory() as tmp_dir, ZipFile(test_file, "r") as zip_test:
            # unzip file to a temp location and read
            zip_test.extractall(tmp_dir)
            test_file = Path(tmp_dir) / test_file.stem
            df = fastnda.read(test_file, software_cycle_number=False)
    else:
        df = fastnda.read(test_file, software_cycle_number=False)
    df_ref = pl.read_parquet(ref_file)
    return df, df_ref


class TestRead:
    """Compared parsed data to reference from BTSDA."""

    def test_generate_cycle_number(self) -> None:
        """Test generating cycle numbers on just one file."""
        test_file = Path(__file__).parent / "test_data" / "nw4-120-1-6-53.ndax"
        df1 = fastnda.read(test_file, software_cycle_number=False)
        df2 = fastnda.read(test_file, software_cycle_number=True, cycle_mode="chg")
        status_mapping = {v: k for k, v in state_dict.items()}
        df1 = df1.with_columns(pl.col("status").replace_strict(status_mapping, return_dtype=pl.Int32))
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
            "timestamp",
            "cycle_count",
            "step_count",
            "step_index",
            "status",
            "capacity_mAh",
            "energy_mWh",
        }
        assert all(col in df.columns for col in df_columns), (
            f"Missing columns in DataFrame: {df_columns - set(df.columns)}"
        )
        df_ref_columns = {
            "Time",
            "Total Time",
            "Date",
            "Step Index",
            "Step Count",
            "Voltage(mV)",
            "Current(uA)",
            "Capacity(mAs)",
            "Energy(mWs)",
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
        # status is enum - faster, but not directly comparable to categorical
        # Need to cast both to same dtype, and replace spaces in ref
        # Neware is inconsistent with 'Dchg' and 'DChg' in column names
        assert_series_equal(
            df["status"].cast(pl.String),
            df_ref["Step Type"].cast(pl.String).str.replace_all(" ", "_").str.replace_all("Dchg", "DChg"),
            check_names=False,
        )

    def test_cycle(self, parsed_data: tuple) -> None:
        """Cycle should be exact when not using software_cycle_number."""
        df, df_ref = parsed_data
        # If the default is wrong, check if software_cycle is correct
        if not (df["cycle_count"] == df_ref["Cycle Index"]).all():
            status_mapping = {v: k for k, v in state_dict.items()}
            df2 = df.with_columns(pl.col("status").replace_strict(status_mapping, return_dtype=pl.Int32))
            df2 = _generate_cycle_number(df2, "auto")
            assert_series_equal(
                df2["cycle_count"],
                df_ref["Cycle Index"],
                check_names=False,
            )
            warnings.warn("Cycles only match when using software_cycle_number", stacklevel=2)

    def test_index(self, parsed_data: tuple) -> None:
        """Index should be UInt32 monotonically increasing by 1."""
        df, df_ref = parsed_data
        assert_series_equal(
            df["index"],
            pl.Series("ref_index", range(1, len(df) + 1), dtype=pl.UInt32),
            check_names=False,
        )

    def test_time(self, parsed_data: tuple) -> None:
        """Time should agree within 1 us."""
        df, df_ref = parsed_data
        assert_series_equal(
            df["step_time_s"],
            df_ref["Time"],
            check_names=False,
            abs_tol=5e-7,
        )

    def test_datetime(self, parsed_data: tuple) -> None:
        """Date should agree within 1 us."""
        df, df_ref = parsed_data
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

        # Datetime should agree with uts
        assert_series_equal(
            df["timestamp"].cast(pl.Float64) * 1e-6,
            df["unix_time_s"],
            check_names=False,
            abs_tol=5e-7,
        )
        # Cannot cycle cells before Neware was founded in 1998
        assert df["unix_time_s"].min() > 883609200

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
        rel_diff = 2 * abs_diff / (df["capacity_mAh"] + df_ref["Capacity(mAs)"].abs() / 3600)
        if ((abs_diff > 3e-4) & (rel_diff > 1e-6)).any():
            # If this fails, sometimes Neware does not count negative current during charge towards the capacity
            df = df.with_columns(
                pl.col("capacity_mAh").abs().cum_max().over(pl.col("step_count")).alias("capacity_ignore_negs_mAh")
            )
            abs_diff = (df["capacity_ignore_negs_mAh"].abs() - df_ref["Capacity(mAs)"].abs() / 3600).abs()
            rel_diff = 2 * abs_diff / (df["capacity_ignore_negs_mAh"].abs() + df_ref["Capacity(mAs)"].abs() / 3600)
            if ((abs_diff > 3e-4) & (rel_diff > 1e-6)).any():
                msg = "Capacity columns are different."
                raise ValueError(msg)

    def test_energy(self, parsed_data: tuple) -> None:
        """Neware energy can be recorded 0.1 mWs, check to 3e-5 mWh."""
        df, df_ref = parsed_data
        # Neware capacity can be absolute for both charge and discharge
        # It can also can have negative values for discharge
        abs_diff = (df["energy_mWh"].abs() - df_ref["Energy(mWs)"].abs() / 3600).abs()
        rel_diff = 2 * abs_diff / (df["energy_mWh"] + df_ref["Energy(mWs)"].abs() / 3600)
        if ((abs_diff > 3e-4) & (rel_diff > 1e-6)).any():
            # If this fails, sometimes Neware does not count negative current during charge towards the energy
            df = df.with_columns(
                pl.col("energy_mWh").abs().cum_max().over(pl.col("step_count")).alias("energy_ignore_negs_mWh")
            )
            abs_diff = (df["energy_ignore_negs_mWh"] - df_ref["Energy(mWs)"].abs() / 3600).abs()
            rel_diff = 2 * abs_diff / (df["energy_ignore_negs_mWh"] + df_ref["Energy(mWs)"].abs() / 3600)
            if ((abs_diff > 3e-4) & (rel_diff > 1e-6)).any():
                msg = "Energy columns are different."
                raise ValueError(msg)

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
                results[ref_col] = sum(abs(df[test_col] - multiplier * df_ref[ref_col])) / len(df)
                if results[ref_col] < tol:
                    break
            else:
                # raise an error
                closest = min(results, key=results.get)
                msg = f"Could not find any column matching values of {test_col}, closest reference was {closest} with an average difference of {results[closest]}"
                raise ValueError(msg)
