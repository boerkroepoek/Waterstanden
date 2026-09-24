"""Domeinmodellen."""
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

TimezoneMode = Literal["Nederlandse lokale tijd", "UTC", "Geen tijdzonecorrectie"]

@dataclass(frozen=True)
class PeilfilterConfiguration:
    """Configuratie voor de waterstandsberekening."""
    filter_id: str
    top_casing_nap_m: float
    cable_length_m: float
    logger_mode: str
    logger_unit: str
    knmi_pressure_unit: str
    default_temperature_c: float
    use_temperature_density: bool

    @property
    def sensor_elevation_nap_m(self) -> float:
        """Geef de hoogte van het sensormembraan ten opzichte van NAP."""
        return self.top_casing_nap_m - self.cable_length_m

@dataclass(frozen=True)
class BinaryFieldDefinition:
    """Definieer een numeriek veld in een binair record."""
    offset: int
    data_type: str
    scale: float = 1.0
    value_offset: float = 0.0

@dataclass(frozen=True)
class BinaryDecoderConfiguration:
    """Configuratie voor vaste binaire records."""
    header_size: int
    record_size: int
    byte_order: str
    timestamp_mode: str
    timestamp_offset: int
    timestamp_data_type: str
    timestamp_unit: str
    timestamp_origin: datetime
    pressure_field: BinaryFieldDefinition
    temperature_field: BinaryFieldDefinition | None
    year_offset: int = 0
    month_offset: int = 2
    day_offset: int = 3
    hour_offset: int = 4
    minute_offset: int = 5
    second_offset: int = 6
    year_data_type: str = "Unsigned integer 16-bit"
