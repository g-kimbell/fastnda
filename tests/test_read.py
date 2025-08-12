"""Test read functionality."""

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_series_equal

import fastnda


@pytest.fixture
def parsed_data(file_pair: tuple[Path, Path]) -> tuple[pl.DataFrame, pl.DataFrame, Path, Path]:
    """Read in the data for each file pair ONCE."""
    input_file, output_file = file_pair
    df, _metadata = fastnda.read(input_file, software_cycle_number=False)
    df_ref = pl.read_parquet(output_file)
    return df, df_ref, input_file, output_file


class TestRead:
    """Compared parsed data to reference from BTSDA."""

    def test_file_columns(self, parsed_data: tuple) -> None:
        """Check that the expected columns are in the DataFrames."""
        df, df_ref, input_file, output_file = parsed_data
        df_columns = {
            "step_time_s",
            "unix_time_s",
            "voltage_V",
            "current_mA",
            "charge_capacity_mAh",
            "discharge_capacity_mAh",
        }
        assert all(col in df.columns for col in df_columns), (
            f"Missing columns in DataFrame: {df_columns - set(df.columns)}"
        )
        df_ref_columns = {
            "Time",
            "Step Index",
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
        df, df_ref, input_file, output_file = parsed_data
        assert_series_equal(
            df["step_index"],
            df_ref["Step Index"],
            check_names=False,
        )
        assert_series_equal(
            df["step_count"],
            df_ref["Step Index"].diff().fill_null(0).ne(0).cum_sum() + 1,
            check_names=False,
        )
        # status is enum - faster, but not directly comparable to categorical
        # Need to cast both to same dtype, and replace spaces in ref
        assert_series_equal(
            df["status"].cast(pl.String),
            df_ref["Step Type"].cast(pl.String).str.replace_all(" ", "_"),
            check_names=False,
        )

    def test_cycle(self, parsed_data: tuple) -> None:
        """Cycle should be exact when not using software_cycle_number."""
        df, df_ref, input_file, output_file = parsed_data
        assert_series_equal(
            df["cycle_count"],
            df_ref["Cycle Index"],
            check_names=False,
        )

    def test_index(self, parsed_data: tuple) -> None:
        """Index should be UInt32 monotonically increasing by 1."""
        df, df_ref, input_file, output_file = parsed_data
        assert_series_equal(
            df["index"],
            pl.Series("ref_index", range(1, len(df) + 1), dtype=pl.UInt32),
            check_names=False,
        )

    def test_time(self, parsed_data: tuple) -> None:
        """Time should agree within 1 us."""
        df, df_ref, input_file, output_file = parsed_data
        assert_series_equal(
            df["step_time_s"],
            df_ref["Time"],
            check_names=False,
            atol=5e-7,
        )

    def test_datetime(self, parsed_data: tuple) -> None:
        """Date should agree within 1 us."""
        df, df_ref, input_file, output_file = parsed_data
        # Cannot compare date directly - Neware datetime is not timezone aware.
        duts = df["unix_time_s"] - df["unix_time_s"][0]
        datetime_ref = df_ref["Date"].cast(pl.Float64) / 1000
        duts_ref = datetime_ref - datetime_ref[0]
        assert_series_equal(
            duts,
            duts_ref,
            check_names=False,
            atol=5e-7,
        )

        # Datetime should agree with uts
        assert_series_equal(
            df["timestamp"].cast(pl.Float64) * 1e-6,
            df["unix_time_s"],
            check_names=False,
            atol=5e-7,
        )
        # Cannot cycle cells before Neware was founded in 1998
        assert df["unix_time_s"].min() > 883609200

    def test_voltage(self, parsed_data: tuple) -> None:
        """Voltage usually recorded to 0.1 mV, should agree within 0.05 mV."""
        df, df_ref, input_file, output_file = parsed_data
        assert_series_equal(
            df["voltage_V"],
            df_ref["Voltage(mV)"] / 1000,
            check_names=False,
            atol=5e-5,
        )

    def test_current(self, parsed_data: tuple) -> None:
        """Current usually recorded to 0.1 mA, should agree within 0.05 mA."""
        df, df_ref, input_file, output_file = parsed_data
        assert_series_equal(
            df["current_mA"],
            df_ref["Current(uA)"] / 1000,
            check_names=False,
            atol=0.05,
        )

    def test_capacity(self, parsed_data: tuple) -> None:
        """Neware capacity should be recorded to 1e-6 mAh, check to 5e-7 mAh."""
        df, df_ref, input_file, output_file = parsed_data
        # Neware capacity is absolute and contains both charge and discharge
        assert_series_equal(
            (df["charge_capacity_mAh"] + df["discharge_capacity_mAh"]),
            df_ref["Capacity(mAs)"] / 3600,
            check_names=False,
            atol=5e-7,
        )

    def test_energy(self, parsed_data: tuple) -> None:
        """Neware energy should be recorded to 1e-6 mWh, check to 5e-7 mWh."""
        df, df_ref, input_file, output_file = parsed_data
        # Neware energy is absolute and contains both charge and discharge
        assert_series_equal(
            (df["charge_energy_mWh"] + df["discharge_energy_mWh"]),
            df_ref["Energy(mWs)"] / 3600,
            check_names=False,
            atol=5e-7,
        )
