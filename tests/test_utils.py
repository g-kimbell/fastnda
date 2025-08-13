"""Test utils module."""

import polars as pl
from polars.testing import assert_series_equal

from fastnda.utils import _count_changes, _generate_cycle_number, _id_first_state


class TestUtils:
    """Test utility functions."""

    def test_count_changes(self) -> None:
        """Test counting cycles from a step index."""
        series = pl.Series([1, 1, 1, 2, 2, 2, 2, 5, 5, 5, 6, 7, 8, 2], dtype=pl.UInt32)
        expect = pl.Series([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 5, 6, 7], dtype=pl.UInt32)
        assert_series_equal(_count_changes(series), expect)

        series = pl.Series([1, 1, 1, 9, 9, 9, 1, 1, 1, 9, 9, 9, 1, 1, 1], dtype=pl.UInt32)
        expect = pl.Series([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], dtype=pl.UInt32)
        assert_series_equal(_count_changes(series), expect)

    def test_id_first_state(self) -> None:
        """Testing finding the first non-rest state."""
        # Codes:
        #     1: "CC_Chg",
        #     2: "CC_DChg",
        #     3: "CV_Chg",
        #     4: "Rest",
        #     5: "cycle_count",
        #     7: "CCCV_Chg",
        #     8: "CP_DChg",
        #     9: "CP_Chg",
        #     10: "CR_DChg",
        #     13: "Pause",
        #     16: "Pulse",
        #     17: "SIM",
        #     19: "CV_DChg",
        #     20: "CCCV_DChg",
        #     21: "Control",
        #     22: "OCV",
        #     26: "CPCV_DChg",
        #     27: "CPCV_Chg",

        df = pl.DataFrame({"status": [4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 2, 2, 2, 1, 1, 1]})
        assert _id_first_state(df) == "chg"
        df = pl.DataFrame({"status": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]})
        assert _id_first_state(df) == "dchg"  # Defaults to discharge
        df = pl.DataFrame({"status": [4, 4, 4, 4, 4, 4, 4, 9, 27, 1, 5, 4, 3, 8, 5]})
        assert _id_first_state(df) == "chg"
        df = pl.DataFrame({"status": [17, 17, 17, 17, 17]})
        assert _id_first_state(df) == "dchg"  # SIM doesn't count as charge or discharge
        df = pl.DataFrame({"status": [22, 22, 22, 22, 22, 21, 13, 13, 4, 4, 45, 20]})
        assert _id_first_state(df) == "dchg"

    def test_generate_cycle_number_chg(self) -> None:
        """Test generating cycle numbers."""
        df = pl.DataFrame(
            {"status": [4, 4, 4, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 4, 4, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2]}
        )
        expect = pl.Series(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5], dtype=pl.UInt32
        )
        assert_series_equal(_generate_cycle_number(df, cycle_mode="chg")["cycle_count"], expect, check_names=False)

    def test_generate_cycle_number_dchg(self) -> None:
        """Test generating cycle numbers."""
        df = pl.DataFrame(
            {"status": [4, 4, 4, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 4, 4, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2]}
        )
        expect = pl.Series(
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6], dtype=pl.UInt32
        )
        assert_series_equal(_generate_cycle_number(df, cycle_mode="dchg")["cycle_count"], expect, check_names=False)

    def test_generate_cycle_number_auto(self) -> None:
        """Test generating cycle numbers."""
        df = pl.DataFrame(
            {"status": [4, 4, 4, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 4, 4, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2]}
        )
        assert_series_equal(
            _generate_cycle_number(df, cycle_mode="auto")["cycle_count"],
            _generate_cycle_number(df, cycle_mode="chg")["cycle_count"],
        )
