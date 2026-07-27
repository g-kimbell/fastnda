"""Unit tests for functions outside of utils."""

import mmap

import polars as pl
import pytest

import fastnda
from fastnda.nda import _merge_aux, _read_nda_130_91


def test_lazy_imports() -> None:
    """Lazy imports resolve known names and rejects unknown ones."""
    assert fastnda.step_type_map is fastnda.dicts.step_type_map
    assert fastnda.btsda_csv_to_parquet is fastnda.btsda.btsda_csv_to_parquet
    with pytest.raises(AttributeError, match="no attribute 'foo'"):
        fastnda.foo  # noqa: B018


def test_nda_aux_merge() -> None:
    """Test nda aux df merge."""
    df = pl.DataFrame(
        {
            "index": [1, 2, 3, 4, 5],
            "voltage_V": [3.3, 3.4, 3.5, 3.6, 3.7],
            "current_mA": [1, 1, 1, 1, 1],
        }
    )
    aux_df = pl.DataFrame(
        {
            "aux": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "index": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "aux_voltage_V": [1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
            "aux_temperature_degC": [2, 2, 2, 2, 2, 4, 4, 4, 4, 4],
        }
    )
    merged_df = _merge_aux(df, aux_df)
    assert len(merged_df) == 5
    assert "aux1_voltage_V" in merged_df.columns
    assert "aux1_temperature_degC" in merged_df.columns
    assert "aux2_voltage_V" in merged_df.columns
    assert "aux2_temperature_degC" in merged_df.columns
    assert merged_df["aux1_voltage_V"].to_list() == [1, 1, 1, 1, 1]
    assert merged_df["aux2_voltage_V"].to_list() == [3, 3, 3, 3, 3]
    assert merged_df["aux1_temperature_degC"].to_list() == [2, 2, 2, 2, 2]
    assert merged_df["aux2_temperature_degC"].to_list() == [4, 4, 4, 4, 4]
    assert "aux" not in merged_df.columns
    assert merged_df["index"].to_list() == [1, 2, 3, 4, 5]


def test_nda_aux_merge_no_aux_col() -> None:
    """Test nda aux df merge."""
    df = pl.DataFrame(
        {
            "index": [1, 2, 3, 4, 5],
            "voltage_V": [3.3, 3.4, 3.5, 3.6, 3.7],
            "current_mA": [1, 1, 1, 1, 1],
        }
    )
    aux_df = pl.DataFrame(
        {
            "index": [1, 2, 3, 4, 5],
            "aux_voltage_V": [1, 1, 1, 1, 1],
            "aux_temperature_degC": [2, 2, 2, 2, 2],
        }
    )
    merged_df = _merge_aux(df, aux_df)
    assert len(merged_df) == 5
    assert "aux_voltage_V" in merged_df.columns
    assert "aux_temperature_degC" in merged_df.columns
    assert merged_df["aux_voltage_V"].to_list() == [1, 1, 1, 1, 1]
    assert merged_df["aux_temperature_degC"].to_list() == [2, 2, 2, 2, 2]
    assert "aux" not in merged_df.columns
    assert merged_df["index"].to_list() == [1, 2, 3, 4, 5]


def test_read_nda_130_91_odd_size() -> None:
    """Test reading nda 130-91 with different record lengths (52,56,60)."""
    header = (
        "4e455741524532303234303430338200130010b70500000000007b010000000000008bb8050000000000e1010000000000006cba05"
    ).ljust(2048, "0")
    records = [
        "550601042b0000000100000000000000809698000000000090727840000000000000000000000000000000002d3e54668006d139",
        "550601042b000000020000000c00000000497f0f00000000d4717840000000000000000000000000000000003a3e54668085b50d",
        "550601042b000000030000000c00000080df1710000000003b76784000000000000000000000000000000000f53f54668033023b",
        "550601042b000000040000004800000080df1710000000004d71784000000000000000000000000000000000314054668033023b",
        "550601042b000000050000008400000080df1710000000008b717840000000000000000000000000000000006d4054668033023b",
    ]
    footer = "550601042b000000"

    def _read_from_hex(data_hex: str) -> pl.DataFrame:
        data = bytes.fromhex(header + data_hex + footer)
        mm = mmap.mmap(-1, len(data))
        mm.write(data)
        mm.seek(0)
        return _read_nda_130_91(mm)

    df = _read_from_hex("".join(records))
    assert len(df) == 5
    assert "temperature_degC" not in df.columns

    df = _read_from_hex("".join(r + "5040b841" for r in records))
    assert len(df) == 5
    assert "aux_temperature_degC" in df.columns

    df = _read_from_hex("".join(r + "5040b841aaaaaaaa" for r in records))
    assert len(df) == 5
    assert "aux_temperature_degC" in df.columns
