# Copyright © 2026, Empa.
"""Test btsda module."""

from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
from polars.testing import assert_frame_equal

from fastnda import btsda_csv_to_parquet
from fastnda.btsda import _time_str_to_float


class TestBTSDA:
    """Test module for creating test data from BTSDA."""

    def test_time_str_to_float(self) -> None:
        """Test converting time strings to floats."""
        assert _time_str_to_float("001:23:45.1") == 1 * 3600 + 23 * 60 + 45.1
        assert _time_str_to_float("0:00:00.0000") == 0
        assert _time_str_to_float("0:02:30.0000") == 2 * 60 + 30
        assert _time_str_to_float("12:34:56.7890") == 12 * 3600 + 34 * 60 + 56.7890

    def test_btsda_csv_to_parquet(self) -> None:
        """Test converting BTSDA CSV to Parquet."""
        current_dir = Path(__file__).parent
        csv_path = current_dir / "test_data" / "interp-test.csv"
        ref_file = current_dir / "test_data" / "interp-test.parquet"
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "interp-test.parquet"
            btsda_csv_to_parquet(csv_path, tmp_path)
            df_test = pl.read_parquet(tmp_path)
            df_ref = pl.read_parquet(ref_file)
            assert_frame_equal(df_test, df_ref)


# Columns btsda_csv_to_parquet reads, in BTSDA's export order
_CSV_COLUMNS = (
    "DataPoint",
    "Cycle Index",
    "Step Index",
    "Step Type",
    "Time",
    "Total Time",
    "Current(µA)",
    "Voltage(mV)",
    "Capacity(mAs)",
    "Energy(mWs)",
    "Date",
    "Step start and end identification ",
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a minimal BTSDA-style export, cp1252 encoded like the real thing."""
    lines = [",".join(_CSV_COLUMNS)]
    lines += [",".join(str(row[col]) for col in _CSV_COLUMNS) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="cp1252")


def _row(
    point: int,
    *,
    time_s: int,
    current_ua: float,
    energy_mws: float,
    date: str,
) -> dict[str, object]:
    """Build one export row, all other fields held constant."""
    stamp = f"00:00:{time_s:02d}.000"
    return {
        "DataPoint": point,
        "Cycle Index": 1,
        "Step Index": 1,
        "Step Type": "CC_Chg",
        "Time": stamp,
        "Total Time": stamp,
        "Current(µA)": current_ua,
        "Voltage(mV)": 3600.0,
        "Capacity(mAs)": 10.0 * point,
        "Energy(mWs)": energy_mws,
        "Date": date,
        "Step start and end identification ": 0 if point == 1 else "",
    }


class TestBTSDAPlaceholderColumns:
    """BTSDA reports placeholder values for fields old Neware formats do not record."""

    DATE = "2025-06-17 09:26:39.000"

    def test_drops_all_zero_energy(self, tmp_path: Path) -> None:
        """Energy is dropped when it is zero everywhere but current is not."""
        csv_path = tmp_path / "zero-energy.csv"
        _write_csv(
            csv_path,
            [
                _row(1, time_s=0, current_ua=0.0, energy_mws=0.0, date=self.DATE),
                _row(2, time_s=3, current_ua=1000.0, energy_mws=0.0, date="2025-06-17 09:26:42.000"),
            ],
        )

        df = btsda_csv_to_parquet(csv_path, tmp_path / "zero-energy.parquet")

        assert "Energy(mWs)" not in df.columns
        assert "Date" in df.columns

    def test_keeps_energy_when_current_is_always_zero(self, tmp_path: Path) -> None:
        """A rest-only export has zero energy for real, so energy is kept."""
        csv_path = tmp_path / "rest-only.csv"
        _write_csv(
            csv_path,
            [
                _row(1, time_s=0, current_ua=0.0, energy_mws=0.0, date=self.DATE),
                _row(2, time_s=3, current_ua=0.0, energy_mws=0.0, date="2025-06-17 09:26:42.000"),
            ],
        )

        df = btsda_csv_to_parquet(csv_path, tmp_path / "rest-only.parquet")

        assert "Energy(mWs)" in df.columns

    def test_drops_constant_date(self, tmp_path: Path) -> None:
        """Date is dropped when it never changes across a test that took time."""
        csv_path = tmp_path / "constant-date.csv"
        _write_csv(
            csv_path,
            [
                _row(1, time_s=0, current_ua=1000.0, energy_mws=3.6, date=self.DATE),
                _row(2, time_s=3, current_ua=1000.0, energy_mws=7.2, date=self.DATE),
            ],
        )

        df = btsda_csv_to_parquet(csv_path, tmp_path / "constant-date.parquet")

        assert "Date" not in df.columns
        assert "Energy(mWs)" in df.columns

    def test_keeps_constant_date_at_time_zero(self, tmp_path: Path) -> None:
        """A single-point export has one date for real, so date is kept."""
        csv_path = tmp_path / "one-point.csv"
        _write_csv(csv_path, [_row(1, time_s=0, current_ua=1000.0, energy_mws=3.6, date=self.DATE)])

        df = btsda_csv_to_parquet(csv_path, tmp_path / "one-point.parquet")

        assert "Date" in df.columns

    def test_drops_both_placeholders(self, tmp_path: Path) -> None:
        """Energy and date are dropped independently of each other."""
        csv_path = tmp_path / "both.csv"
        _write_csv(
            csv_path,
            [
                _row(1, time_s=0, current_ua=1000.0, energy_mws=0.0, date=self.DATE),
                _row(2, time_s=3, current_ua=1000.0, energy_mws=0.0, date=self.DATE),
            ],
        )

        df = btsda_csv_to_parquet(csv_path, tmp_path / "both.parquet")

        assert "Energy(mWs)" not in df.columns
        assert "Date" not in df.columns
        assert pl.read_parquet(tmp_path / "both.parquet").columns == df.columns
