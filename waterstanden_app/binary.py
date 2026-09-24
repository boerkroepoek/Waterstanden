"""Decoder voor binaire loggerrecords."""
import logging
import math
import struct
from datetime import datetime
import numpy as np
import pandas as pd
from .constants import BINARY_VALUE_FORMATS
from .models import BinaryDecoderConfiguration, BinaryFieldDefinition

LOGGER = logging.getLogger(__name__)

def create_hex_preview(file_bytes: bytes, start: int = 0, length: int = 512, bytes_per_line: int = 16) -> str:
    """Maak een hexadecimale en ASCII-preview."""
    data = file_bytes[start:min(len(file_bytes), start + length)]; lines = []
    for line_start in range(0, len(data), bytes_per_line):
        chunk = data[line_start:line_start + bytes_per_line]
        hexadecimal = " ".join(f"{value:02X}" for value in chunk).ljust(bytes_per_line * 3 - 1)
        ascii_text = "".join(chr(value) if 32 <= value <= 126 else "." for value in chunk)
        lines.append(f"{start + line_start:08X}  {hexadecimal}  |{ascii_text}|")
    return "\n".join(lines)

def get_binary_format(data_type: str, byte_order: str) -> tuple[str, int]:
    """Geef struct-formaat en veldlengte."""
    if data_type not in BINARY_VALUE_FORMATS:
        raise ValueError(f"Onbekend binair datatype: {data_type}")
    character, size = BINARY_VALUE_FORMATS[data_type]
    if byte_order not in {"Little-endian", "Big-endian"}:
        raise ValueError(f"Onbekende bytevolgorde: {byte_order}")
    return ("<" if byte_order == "Little-endian" else ">") + character, size

def validate_field(field: BinaryFieldDefinition, record_size: int, byte_order: str, name: str) -> None:
    """Controleer dat een veld volledig binnen een record past."""
    if field.offset < 0:
        raise ValueError(f"De offset van {name} mag niet negatief zijn.")
    _, size = get_binary_format(field.data_type, byte_order)
    if field.offset + size > record_size:
        raise ValueError(f"{name} op offset {field.offset} met lengte {size} past niet in een record van {record_size} bytes.")
    if not math.isfinite(field.scale) or not math.isfinite(field.value_offset):
        raise ValueError(f"Schaal en offset van {name} moeten eindig zijn.")

def unpack_binary_value(record: bytes, field: BinaryFieldDefinition, byte_order: str) -> float:
    """Lees en transformeer één numerieke waarde."""
    format_string, size = get_binary_format(field.data_type, byte_order)
    if field.offset < 0 or field.offset + size > len(record):
        raise ValueError(f"Veld op offset {field.offset} past niet in een record van {len(record)} bytes.")
    raw = struct.unpack_from(format_string, record, field.offset)[0]
    return float(raw) * field.scale + field.value_offset

def decode_timestamp(record: bytes, configuration: BinaryDecoderConfiguration) -> pd.Timestamp:
    """Decodeer één tijdstempel."""
    mode = configuration.timestamp_mode
    if mode == "Losse datumvelden":
        year = int(unpack_binary_value(record, BinaryFieldDefinition(configuration.year_offset, configuration.year_data_type), configuration.byte_order))
        offsets = (configuration.month_offset, configuration.day_offset, configuration.hour_offset, configuration.minute_offset, configuration.second_offset)
        if any(offset < 0 or offset >= len(record) for offset in offsets):
            raise ValueError("Een of meer datumvelden vallen buiten het record.")
        month, day, hour, minute, second = (record[offset] for offset in offsets)
        return pd.Timestamp(datetime(year, month, day, hour, minute, second))
    raw = unpack_binary_value(record, BinaryFieldDefinition(configuration.timestamp_offset, configuration.timestamp_data_type), configuration.byte_order)
    if not math.isfinite(raw):
        return pd.NaT
    if mode == "Unix-tijd":
        units = {"seconden": "s", "milliseconden": "ms", "microseconden": "us", "nanoseconden": "ns"}
        if configuration.timestamp_unit not in units:
            raise ValueError("Ongeldige Unix-tijdseenheid.")
        return pd.to_datetime(raw, unit=units[configuration.timestamp_unit], origin="unix", errors="coerce")
    if mode == "Excel-datum":
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(raw, unit="D")
    if mode == "Tijd sinds aangepaste oorsprong":
        units = {"seconden": "s", "milliseconden": "ms", "microseconden": "us", "dagen": "D"}
        if configuration.timestamp_unit not in units:
            raise ValueError("Ongeldige tijdseenheid voor aangepaste oorsprong.")
        return pd.Timestamp(configuration.timestamp_origin) + pd.to_timedelta(raw, unit=units[configuration.timestamp_unit])
    raise ValueError(f"Onbekende tijdstempelmodus: {mode}")

def validate_binary_configuration(file_bytes: bytes, configuration: BinaryDecoderConfiguration) -> None:
    """Valideer bestand en alle gebruikte velden."""
    if not file_bytes:
        raise ValueError("Het bestand is leeg.")
    if configuration.header_size < 0 or configuration.header_size >= len(file_bytes):
        raise ValueError("De headerlengte valt buiten het bestand.")
    if configuration.record_size <= 0:
        raise ValueError("De recordlengte moet groter zijn dan nul.")
    if len(file_bytes) - configuration.header_size < configuration.record_size:
        raise ValueError("Na de header resteert minder dan één volledig record.")
    validate_field(configuration.pressure_field, configuration.record_size, configuration.byte_order, "Het drukveld")
    if configuration.temperature_field:
        validate_field(configuration.temperature_field, configuration.record_size, configuration.byte_order, "Het temperatuurveld")
    if configuration.timestamp_mode == "Losse datumvelden":
        validate_field(BinaryFieldDefinition(configuration.year_offset, configuration.year_data_type), configuration.record_size, configuration.byte_order, "Het jaarveld")
        for name, offset in (("maand", configuration.month_offset), ("dag", configuration.day_offset), ("uur", configuration.hour_offset), ("minuut", configuration.minute_offset), ("seconde", configuration.second_offset)):
            validate_field(BinaryFieldDefinition(offset, "Unsigned integer 8-bit"), configuration.record_size, configuration.byte_order, f"Het {name}veld")
    else:
        validate_field(BinaryFieldDefinition(configuration.timestamp_offset, configuration.timestamp_data_type), configuration.record_size, configuration.byte_order, "Het tijdstempelveld")

def decode_binary_logger(file_bytes: bytes, configuration: BinaryDecoderConfiguration, maximum_records: int | None = None) -> pd.DataFrame:
    """Decodeer records en behoud fouten voor diagnose."""
    validate_binary_configuration(file_bytes, configuration)
    count = (len(file_bytes) - configuration.header_size) // configuration.record_size
    if maximum_records is not None:
        if maximum_records <= 0:
            raise ValueError("maximum_records moet positief zijn.")
        count = min(count, maximum_records)
    rows = []
    for index in range(count):
        start = configuration.header_size + index * configuration.record_size
        record = file_bytes[start:start + configuration.record_size]
        try:
            timestamp = decode_timestamp(record, configuration)
            pressure = unpack_binary_value(record, configuration.pressure_field, configuration.byte_order)
            temperature = np.nan if configuration.temperature_field is None else unpack_binary_value(record, configuration.temperature_field, configuration.byte_order)
            error = ""
        except (ValueError, OverflowError, struct.error, TypeError) as exc:
            timestamp, pressure, temperature, error = pd.NaT, np.nan, np.nan, str(exc)
        rows.append({"recordnummer": index + 1, "byte_offset": start, "tijd": timestamp, "loggerwaarde": pressure, "temperatuur_c": temperature, "decodeerfout": error})
    frame = pd.DataFrame(rows)
    if frame.empty or not (frame["tijd"].notna() & frame["loggerwaarde"].notna()).any():
        raise ValueError("Geen record leverde een geldig tijdstip en een geldige loggerwaarde op.")
    return frame
