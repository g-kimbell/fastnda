"""Mappings used in data processing."""

from collections.abc import Mapping
from types import MappingProxyType

import polars as pl

# Neware step type integer to string codes
STEP_TYPE_MAP = MappingProxyType(
    {
        1: "CC_Chg",
        2: "CC_DChg",
        3: "CV_Chg",
        4: "Rest",
        5: "Cycle",
        7: "CCCV_Chg",
        8: "CP_DChg",
        9: "CP_Chg",
        10: "CR_DChg",
        13: "Pause",
        16: "Pulse",
        17: "SIM",
        18: "PCCCV_Chg",
        19: "CV_DChg",
        20: "CCCV_DChg",
        21: "Control",
        22: "OCV",
        25: "Ramp",
        26: "CPCV_DChg",
        27: "CPCV_Chg",
    }
)

# For _generate_cycle_number, 1 = charge, 0 = discharge
CHARGE_DISCHARGE_MAP = MappingProxyType(
    {
        1: 1,
        2: 0,
        3: 1,
        7: 1,
        8: 0,
        9: 1,
        10: 0,
        18: 1,
        19: 0,
        20: 0,
        26: 0,
        27: 1,
    }
)

# Final column datatypes (excluding aux columns)
DTYPE_MAP: Mapping[str, type[pl.DataType]] = MappingProxyType(
    {
        "index": pl.UInt32,
        "voltage_V": pl.Float32,
        "current_mA": pl.Float32,
        "unix_time_s": pl.Float64,
        "step_time_s": pl.Float64,
        "total_time_s": pl.Float64,
        "cycle_count": pl.UInt32,
        "step_count": pl.UInt32,
        "step_index": pl.UInt32,
        "step_type": pl.Categorical,
        "capacity_mAh": pl.Float32,
        "energy_mWh": pl.Float32,
    }
)

# Current value multiplier based on instrument Range setting
MULTIPLIER_MAP = MappingProxyType(
    {
        -100000000: 1e1,
        -200000: 1e-2,
        -100000: 1e-2,
        -60000: 1e-2,
        -30000: 1e-2,
        -50000: 1e-2,
        -40000: 1e-2,
        -20000: 1e-2,
        -12000: 1e-2,
        -10000: 1e-2,
        -6000: 1e-2,
        -5000: 1e-2,
        -3000: 1e-2,
        -2000: 1e-2,
        -1000: 1e-2,
        -500: 1e-3,
        -100: 1e-3,
        -50: 1e-4,
        -25: 1e-4,
        -20: 1e-4,
        -10: 1e-4,
        -5: 1e-5,
        -2: 1e-5,
        -1: 1e-5,
        0: 0.0,
        1: 1e-4,
        2: 1e-4,
        5: 1e-4,
        10: 1e-3,
        20: 1e-3,
        25: 1e-3,
        50: 1e-3,
        100: 1e-2,
        200: 1e-2,
        250: 1e-2,
        500: 1e-2,
        1000: 1e-1,
        6000: 1e-1,
        10000: 1e-1,
        12000: 1e-1,
        20000: 1e-1,
        30000: 1e-1,
        40000: 1e-1,
        50000: 1e-1,
        60000: 1e-1,
        100000: 1e-1,
        200000: 1e-1,
    }
)

AUX_CHL_MAP = MappingProxyType(
    {
        102: "voltage_V",  # aux voltage
        103: "temperature_degC",  # aux temp
        104: "current_mA",  # aux current in A, gets scaled
        105: "voltage_V",  # 'mult' voltage
        106: "temperature_degC",  # 'mult' temp
        107: "current_mA",  # 'mult' curr in A
        108: "clamp_temperature_degC",  # 'clamp temp'
        109: "clamp_weight_kg",  # 'clamp press'
        110: "clamp_air_pressure_kPa",  # 'clamp air press'
        111: "leader_temperature_degC",  # 'leader temp'
        112: "red_temperature_degC",  # 'red temp'
        113: "humidity",  # not sure of unit
        114: "thickness_mm",
        115: "flow",  # not sure of unit
        116: "voc",  # not sure what this is or what the unit is
        117: "pressure_bar",  # not sure what this relates to
        118: "force_N",  # not sure what this relates to
        335: "temperature_setpoint_degC",
        345: "humidity_%",
    }
)

# Scalings applied to AUX_CHL_MAP
AUX_CHL_SCALE_MAP = MappingProxyType(
    {
        104: 1000,  # A -> mA
        107: 1000,  # A -> mA
    }
)

# User-facing mutable dicts
step_type_map = dict(STEP_TYPE_MAP)

# Kept for backwards compatibility
multiplier_dict = dict(MULTIPLIER_MAP)
aux_chl_type_columns = dict(AUX_CHL_MAP)
dtype_dict = dict(DTYPE_MAP)
state_dict = dict(STEP_TYPE_MAP)
