"""Gedeelde constanten voor de applicatie."""
from typing import Final

APP_TITLE: Final = "Waterstanden naar NAP"
GRAVITY_M_S2: Final = 9.80665
DEFAULT_WATER_DENSITY_KG_M3: Final = 998.2
MAX_FILE_SIZE_MB: Final = 200
NANOSECONDS_PER_HOUR: Final = 3_600_000_000_000
LOGGER_MODES: Final = {"Absolute druk": "absolute_pressure", "Waterkolom": "water_column"}
PRESSURE_UNIT_FACTORS_TO_PA: Final = {
    "Pa": 1.0, "hPa": 100.0, "mbar": 100.0, "0,1 hPa": 10.0,
    "kPa": 1_000.0, "bar": 100_000.0, "psi": 6_894.757293168,
    "mH2O": DEFAULT_WATER_DENSITY_KG_M3 * GRAVITY_M_S2,
    "cmH2O": DEFAULT_WATER_DENSITY_KG_M3 * GRAVITY_M_S2 / 100.0,
}
LENGTH_UNIT_FACTORS_TO_M: Final = {"m": 1.0, "cm": 0.01, "mm": 0.001}
BINARY_VALUE_FORMATS: Final = {
    "Signed integer 8-bit": ("b", 1), "Unsigned integer 8-bit": ("B", 1),
    "Signed integer 16-bit": ("h", 2), "Unsigned integer 16-bit": ("H", 2),
    "Signed integer 32-bit": ("i", 4), "Unsigned integer 32-bit": ("I", 4),
    "Signed integer 64-bit": ("q", 8), "Unsigned integer 64-bit": ("Q", 8),
    "Float 32-bit": ("f", 4), "Float 64-bit": ("d", 8),
}
TEXT_ENCODINGS: Final = ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be", "cp1252", "latin-1")
