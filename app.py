"""
Streamlit-applicatie voor het omrekenen van loggerdata naar waterstanden
ten opzichte van NAP.

Ondersteunde loggerbestanden:
- CSV, TXT en tekstuele DAT-bestanden;
- UTF-8, Windows-1252, Latin-1 en UTF-16;
- generieke binaire DAT-bestanden met vaste recordstructuur.

De decoder voor binaire bestanden ondersteunt:
- configureerbare headerlengte;
- configureerbare recordlengte;
- little-endian en big-endian;
- Unix-tijdstempels;
- Excel-datums;
- losse datumvelden;
- integers, float32 en float64;
- schaalfactoren en offsets.

Let op:
Een DAT-extensie definieert geen vast bestandsformaat. Fabrikanten kunnen
een propriëtair, gecomprimeerd of versleuteld formaat gebruiken.
"""

from __future__ import annotations

import io
import logging
import math
import re
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# Configuratie
# =============================================================================

APP_TITLE: Final[str] = "Waterstanden naar NAP"
GRAVITY_M_S2: Final[float] = 9.80665
DEFAULT_WATER_DENSITY_KG_M3: Final[float] = 998.2
MAX_FILE_SIZE_MB: Final[int] = 200
NANOSECONDS_PER_HOUR: Final[int] = 3_600_000_000_000

LOGGER_MODES: Final[dict[str, str]] = {
    "Absolute druk": "absolute_pressure",
    "Waterkolom": "water_column",
}

PRESSURE_UNIT_FACTORS_TO_PA: Final[dict[str, float]] = {
    "Pa": 1.0,
    "hPa": 100.0,
    "mbar": 100.0,
    "0,1 hPa": 10.0,
    "kPa": 1_000.0,
    "bar": 100_000.0,
    "psi": 6_894.757293168,
    "mH2O": DEFAULT_WATER_DENSITY_KG_M3 * GRAVITY_M_S2,
    "cmH2O": DEFAULT_WATER_DENSITY_KG_M3 * GRAVITY_M_S2 / 100.0,
}

LENGTH_UNIT_FACTORS_TO_M: Final[dict[str, float]] = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
}

BINARY_VALUE_FORMATS: Final[dict[str, tuple[str, int]]] = {
    "Signed integer 8-bit": ("b", 1),
    "Unsigned integer 8-bit": ("B", 1),
    "Signed integer 16-bit": ("h", 2),
    "Unsigned integer 16-bit": ("H", 2),
    "Signed integer 32-bit": ("i", 4),
    "Unsigned integer 32-bit": ("I", 4),
    "Signed integer 64-bit": ("q", 8),
    "Unsigned integer 64-bit": ("Q", 8),
    "Float 32-bit": ("f", 4),
    "Float 64-bit": ("d", 8),
}

TEXT_ENCODINGS: Final[tuple[str, ...]] = (
    "utf-8-sig",
    "utf-8",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "cp1252",
    "latin-1",
)

TimezoneMode = Literal[
    "Nederlandse lokale tijd",
    "UTC",
    "Geen tijdzonecorrectie",
]


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


# =============================================================================
# Datamodellen
# =============================================================================

@dataclass(frozen=True)
class PeilfilterConfiguration:
    """Configuratie voor de hydrologische berekening."""

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
        """Hoogte van het druksensormembraan ten opzichte van NAP."""
        return self.top_casing_nap_m - self.cable_length_m


@dataclass(frozen=True)
class BinaryFieldDefinition:
    """Definitie van één numeriek veld in een binair record."""

    offset: int
    data_type: str
    scale: float = 1.0
    value_offset: float = 0.0


@dataclass(frozen=True)
class BinaryDecoderConfiguration:
    """Configuratie voor een binair bestand met records van vaste lengte."""

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


# =============================================================================
# Algemene hulpfuncties
# =============================================================================

def water_density_kg_m3(temperature_c: pd.Series) -> pd.Series:
    """Bereken de dichtheid van zoet water op basis van temperatuur."""
    temperature = temperature_c.clip(lower=0.0, upper=40.0)

    numerator = (
        (temperature + 288.9414)
        * (temperature - 3.9863) ** 2
    )
    denominator = 508_929.2 * (temperature + 68.12963)

    return 1_000.0 * (1.0 - numerator / denominator)


def pressure_to_pa(values: pd.Series, unit: str) -> pd.Series:
    """Converteer een drukreeks naar pascal."""
    if unit not in PRESSURE_UNIT_FACTORS_TO_PA:
        raise ValueError(f"Onbekende drukeenheid: {unit}")

    return values * PRESSURE_UNIT_FACTORS_TO_PA[unit]


def length_to_m(values: pd.Series, unit: str) -> pd.Series:
    """Converteer een lengtereeks naar meter."""
    if unit not in LENGTH_UNIT_FACTORS_TO_M:
        raise ValueError(f"Onbekende lengte-eenheid: {unit}")

    return values * LENGTH_UNIT_FACTORS_TO_M[unit]


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """Converteer tekstwaarden robuust naar numerieke waarden."""
    text = series.astype("string").str.strip()

    text = text.mask(
        text.str.lower().isin(
            {
                "",
                "na",
                "n/a",
                "nan",
                "none",
                "null",
                "-999",
                "-9999",
                "-999.9",
            }
        )
    )

    text = text.str.replace("\u00a0", "", regex=False)
    text = text.str.replace(" ", "", regex=False)

    both = (
        text.str.contains(",", na=False)
        & text.str.contains(r"\.", na=False)
    )

    comma_decimal = both & (text.str.rfind(",") > text.str.rfind("."))

    text = text.where(
        ~comma_decimal,
        text.str.replace(".", "", regex=False).str.replace(
            ",",
            ".",
            regex=False,
        ),
    )

    dot_decimal = both & ~comma_decimal

    text = text.where(
        ~dot_decimal,
        text.str.replace(",", "", regex=False),
    )

    only_comma = (
        text.str.contains(",", na=False)
        & ~text.str.contains(r"\.", na=False)
    )

    text = text.where(
        ~only_comma,
        text.str.replace(",", ".", regex=False),
    )

    extracted = text.str.extract(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        expand=False,
    )

    return pd.to_numeric(extracted, errors="coerce")


def normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normaliseer kolomnamen en maak dubbele namen uniek."""
    result = dataframe.copy()
    seen: dict[str, int] = {}
    normalized: list[str] = []

    for column in result.columns:
        name = re.sub(r"\s+", " ", str(column).strip()) or "kolom"
        count = seen.get(name, 0)
        seen[name] = count + 1

        if count:
            name = f"{name}_{count + 1}"

        normalized.append(name)

    result.columns = normalized
    return result


def localize_datetime_series(
    series: pd.Series,
    timezone_mode: TimezoneMode,
) -> pd.Series:
    """
    Zet tijdstempels om naar tijdzonevrije Nederlandse lokale tijd.

    Bij 'UTC' worden de waarden eerst als UTC geïnterpreteerd en daarna
    omgerekend naar Europe/Amsterdam.
    """
    parsed = pd.to_datetime(series, errors="coerce")

    if timezone_mode == "Geen tijdzonecorrectie":
        return parsed

    if timezone_mode == "UTC":
        localized = parsed.dt.tz_localize(
            "UTC",
            ambiguous="NaT",
            nonexistent="NaT",
        )
        return localized.dt.tz_convert(
            "Europe/Amsterdam"
        ).dt.tz_localize(None)

    localized = parsed.dt.tz_localize(
        "Europe/Amsterdam",
        ambiguous="NaT",
        nonexistent="NaT",
    )

    return localized.dt.tz_localize(None)


# =============================================================================
# Detectie tekst of binair
# =============================================================================

def detect_file_kind(file_bytes: bytes) -> tuple[str, str | None, float]:
    """
    Bepaal of een bestand waarschijnlijk tekst of binair is.

    Returns:
        Tuple met bestandstype, vermoedelijke encoding en confidence-score.
    """
    if not file_bytes:
        raise ValueError("Het geüploade bestand is leeg.")

    sample = file_bytes[:65_536]

    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "text", "utf-16", 1.0

    if sample.startswith(b"\xef\xbb\xbf"):
        return "text", "utf-8-sig", 1.0

    if b"\x00" in sample:
        even_nulls = sample[0::2].count(0)
        odd_nulls = sample[1::2].count(0)
        pairs = max(1, len(sample) // 2)

        if even_nulls / pairs > 0.25:
            return "text", "utf-16-be", 0.9

        if odd_nulls / pairs > 0.25:
            return "text", "utf-16-le", 0.9

    for encoding in TEXT_ENCODINGS:
        try:
            decoded = sample.decode(encoding)
        except UnicodeDecodeError:
            continue

        if not decoded:
            continue

        printable = sum(
            character.isprintable() or character in "\r\n\t"
            for character in decoded
        )

        printable_ratio = printable / len(decoded)

        separators = sum(
            decoded.count(separator)
            for separator in (";", ",", "\t", "|", "\n")
        )

        if printable_ratio >= 0.90 and separators >= 2:
            return "text", encoding, printable_ratio

    return "binary", None, 0.95


def create_hex_preview(
    file_bytes: bytes,
    start: int = 0,
    length: int = 512,
    bytes_per_line: int = 16,
) -> str:
    """Maak een hexadecimale en ASCII-preview van binaire data."""
    end = min(len(file_bytes), start + length)
    data = file_bytes[start:end]
    lines: list[str] = []

    for line_start in range(0, len(data), bytes_per_line):
        chunk = data[line_start:line_start + bytes_per_line]

        hexadecimal = " ".join(
            f"{value:02X}"
            for value in chunk
        )

        hexadecimal = hexadecimal.ljust(bytes_per_line * 3 - 1)

        ascii_text = "".join(
            chr(value) if 32 <= value <= 126 else "."
            for value in chunk
        )

        absolute_offset = start + line_start

        lines.append(
            f"{absolute_offset:08X}  {hexadecimal}  |{ascii_text}|"
        )

    return "\n".join(lines)


# =============================================================================
# Tekstbestanden
# =============================================================================

def decode_text_file(
    file_bytes: bytes,
    preferred_encoding: str | None = None,
) -> str:
    """Decodeer een tekstbestand."""
    encodings = list(TEXT_ENCODINGS)

    if preferred_encoding:
        encodings = [
            preferred_encoding,
            *[
                encoding
                for encoding in encodings
                if encoding != preferred_encoding
            ],
        ]

    for encoding in encodings:
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(
        "Het bestand kan niet als tekst worden gelezen. "
        "Selecteer binaire verwerking."
    )


def detect_separator(text: str) -> str:
    """Detecteer het meest waarschijnlijke scheidingsteken."""
    lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ][:30]

    if not lines:
        raise ValueError("Geen gegevensregels gevonden.")

    candidates = (";", "\t", ",", "|")
    scores: dict[str, float] = {}

    for separator in candidates:
        counts = [line.count(separator) for line in lines]
        nonzero = [count for count in counts if count > 0]

        if not nonzero:
            scores[separator] = -1.0
            continue

        scores[separator] = (
            len(nonzero) / len(lines) * 10
            + float(np.mean(nonzero))
            - float(np.std(nonzero))
        )

    selected = max(scores, key=scores.get)

    if scores[selected] < 0:
        return r"\s+"

    return selected


def read_text_table(
    file_bytes: bytes,
    encoding: str | None,
    separator_name: str,
    decimal: str,
    header_row: int,
) -> pd.DataFrame:
    """Lees een tekstueel logger- of KNMI-bestand."""
    text = decode_text_file(file_bytes, encoding)

    separator_mapping = {
        "Automatisch": detect_separator(text),
        "Puntkomma": ";",
        "Komma": ",",
        "Tab": "\t",
        "Pipe": "|",
        "Spaties": r"\s+",
    }

    separator = separator_mapping[separator_name]

    try:
        dataframe = pd.read_csv(
            io.StringIO(text),
            sep=separator,
            decimal=decimal,
            header=header_row,
            engine="python",
            dtype=str,
            on_bad_lines="skip",
        )
    except Exception as exc:
        raise ValueError(
            "Het tekstbestand kon niet als tabel worden gelezen."
        ) from exc

    dataframe = normalize_columns(dataframe)
    dataframe = dataframe.dropna(axis=0, how="all")
    dataframe = dataframe.dropna(axis=1, how="all")

    if dataframe.empty:
        raise ValueError("Het bestand bevat geen bruikbare tabelgegevens.")

    return dataframe


# =============================================================================
# Binaire decoder
# =============================================================================

def get_binary_format(data_type: str, byte_order: str) -> tuple[str, int]:
    """Maak een struct-formaat voor een geselecteerd datatype."""
    if data_type not in BINARY_VALUE_FORMATS:
        raise ValueError(f"Onbekend binair datatype: {data_type}")

    format_character, size = BINARY_VALUE_FORMATS[data_type]
    endian_character = "<" if byte_order == "Little-endian" else ">"

    return endian_character + format_character, size


def unpack_binary_value(
    record: bytes,
    field: BinaryFieldDefinition,
    byte_order: str,
) -> float:
    """Lees een numerieke waarde uit een binair record."""
    format_string, size = get_binary_format(
        field.data_type,
        byte_order,
    )

    if field.offset < 0 or field.offset + size > len(record):
        raise ValueError(
            f"Veld op offset {field.offset} past niet in een record "
            f"van {len(record)} bytes."
        )

    raw_value = struct.unpack_from(
        format_string,
        record,
        field.offset,
    )[0]

    return float(raw_value) * field.scale + field.value_offset


def decode_timestamp(
    record: bytes,
    configuration: BinaryDecoderConfiguration,
) -> pd.Timestamp:
    """Decodeer het tijdstip van één binair record."""
    mode = configuration.timestamp_mode

    if mode == "Losse datumvelden":
        year_field = BinaryFieldDefinition(
            offset=configuration.year_offset,
            data_type=configuration.year_data_type,
        )

        year = int(
            unpack_binary_value(
                record,
                year_field,
                configuration.byte_order,
            )
        )

        byte_offsets = (
            configuration.month_offset,
            configuration.day_offset,
            configuration.hour_offset,
            configuration.minute_offset,
            configuration.second_offset,
        )

        values: list[int] = []

        for offset in byte_offsets:
            if offset < 0 or offset >= len(record):
                raise ValueError(
                    f"Datumveld op offset {offset} valt buiten het record."
                )

            values.append(record[offset])

        month, day, hour, minute, second = values

        return pd.Timestamp(
            datetime(
                year,
                month,
                day,
                hour,
                minute,
                second,
            )
        )

    timestamp_field = BinaryFieldDefinition(
        offset=configuration.timestamp_offset,
        data_type=configuration.timestamp_data_type,
    )

    raw_timestamp = unpack_binary_value(
        record,
        timestamp_field,
        configuration.byte_order,
    )

    if not math.isfinite(raw_timestamp):
        return pd.NaT

    if mode == "Unix-tijd":
        unit_mapping = {
            "seconden": "s",
            "milliseconden": "ms",
            "microseconden": "us",
            "nanoseconden": "ns",
        }

        unit = unit_mapping[configuration.timestamp_unit]

        return pd.to_datetime(
            raw_timestamp,
            unit=unit,
            origin="unix",
            errors="coerce",
        )

    if mode == "Excel-datum":
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(
            raw_timestamp,
            unit="D",
        )

    if mode == "Tijd sinds aangepaste oorsprong":
        unit_mapping = {
            "seconden": "s",
            "milliseconden": "ms",
            "microseconden": "us",
            "dagen": "D",
        }

        unit = unit_mapping[configuration.timestamp_unit]

        return (
            pd.Timestamp(configuration.timestamp_origin)
            + pd.to_timedelta(raw_timestamp, unit=unit)
        )

    raise ValueError(f"Onbekende tijdstempelmodus: {mode}")


def validate_binary_configuration(
    file_bytes: bytes,
    configuration: BinaryDecoderConfiguration,
) -> None:
    """Valideer de binaire decoderconfiguratie."""
    if configuration.header_size < 0:
        raise ValueError("De headerlengte mag niet negatief zijn.")

    if configuration.record_size <= 0:
        raise ValueError("De recordlengte moet groter zijn dan nul.")

    if configuration.header_size >= len(file_bytes):
        raise ValueError(
            "De headerlengte is groter dan of gelijk aan het bestand."
        )

    remaining_bytes = len(file_bytes) - configuration.header_size

    if remaining_bytes < configuration.record_size:
        raise ValueError(
            "Na de header resteert minder dan één volledig record."
        )

    fields = [configuration.pressure_field]

    if configuration.temperature_field is not None:
        fields.append(configuration.temperature_field)

    for field in fields:
        _, size = get_binary_format(
            field.data_type,
            configuration.byte_order,
        )

        if field.offset + size > configuration.record_size:
            raise ValueError(
                f"Het veld op offset {field.offset} met lengte {size} "
                "past niet in de gekozen recordlengte."
            )


def decode_binary_logger(
    file_bytes: bytes,
    configuration: BinaryDecoderConfiguration,
    maximum_records: int | None = None,
) -> pd.DataFrame:
    """Decodeer een binair loggerbestand met records van vaste lengte."""
    validate_binary_configuration(file_bytes, configuration)

    data_size = len(file_bytes) - configuration.header_size
    record_count = data_size // configuration.record_size
    remainder = data_size % configuration.record_size

    if remainder:
        LOGGER.warning(
            "%s resterende bytes vormen geen volledig record.",
            remainder,
        )

    if maximum_records is not None:
        record_count = min(record_count, maximum_records)

    rows: list[dict[str, object]] = []
    decoding_errors = 0

    for record_index in range(record_count):
        record_start = (
            configuration.header_size
            + record_index * configuration.record_size
        )
        record_end = record_start + configuration.record_size
        record = file_bytes[record_start:record_end]

        try:
            timestamp = decode_timestamp(record, configuration)

            pressure = unpack_binary_value(
                record,
                configuration.pressure_field,
                configuration.byte_order,
            )

            temperature: float | None = None

            if configuration.temperature_field is not None:
                temperature = unpack_binary_value(
                    record,
                    configuration.temperature_field,
                    configuration.byte_order,
                )

            rows.append(
                {
                    "recordnummer": record_index + 1,
                    "byte_offset": record_start,
                    "tijd": timestamp,
                    "loggerwaarde": pressure,
                    "temperatuur_c": temperature,
                    "decodeerfout": "",
                }
            )

        except (
            ValueError,
            OverflowError,
            struct.error,
            TypeError,
        ) as exc:
            decoding_errors += 1

            rows.append(
                {
                    "recordnummer": record_index + 1,
                    "byte_offset": record_start,
                    "tijd": pd.NaT,
                    "loggerwaarde": np.nan,
                    "temperatuur_c": np.nan,
                    "decodeerfout": str(exc),
                }
            )

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        raise ValueError("Er konden geen binaire records worden gelezen.")

    valid_rows = dataframe["tijd"].notna() & dataframe["loggerwaarde"].notna()

    if not valid_rows.any():
        raise ValueError(
            "Geen enkel binair record leverde een geldig tijdstip en een "
            "geldige loggerwaarde op. Controleer header, recordlengte, "
            "bytevolgorde, offsets en datatypes."
        )

    LOGGER.info(
        "Binair bestand gedecodeerd: %s records, %s fouten.",
        len(dataframe),
        decoding_errors,
    )

    return dataframe


# =============================================================================
# Tijdreeksen voorbereiden
# =============================================================================

def prepare_text_logger(
    dataframe: pd.DataFrame,
    date_column: str,
    time_column: str | None,
    value_column: str,
    temperature_column: str | None,
    timezone_mode: TimezoneMode,
    day_first: bool,
) -> pd.DataFrame:
    """Zet een tekstueel loggerbestand om naar de standaardstructuur."""
    date_values = dataframe[date_column].astype("string").str.strip()

    if time_column:
        time_values = dataframe[time_column].astype("string").str.strip()
        datetime_values = date_values + " " + time_values
    else:
        datetime_values = date_values

    result = pd.DataFrame(
        {
            "tijd": pd.to_datetime(
                datetime_values,
                errors="coerce",
                dayfirst=day_first,
                format="mixed",
            ),
            "loggerwaarde": clean_numeric_series(
                dataframe[value_column]
            ),
        }
    )

    if temperature_column:
        result["temperatuur_c"] = clean_numeric_series(
            dataframe[temperature_column]
        )
    else:
        result["temperatuur_c"] = np.nan

    result["tijd"] = localize_datetime_series(
        result["tijd"],
        timezone_mode,
    )

    result = result.dropna(subset=["tijd"])
    result = result.sort_values("tijd")
    result = result.drop_duplicates("tijd", keep="last")
    result = result.reset_index(drop=True)

    if result.empty:
        raise ValueError("Geen geldige logger-tijdstippen gevonden.")

    return result


def prepare_binary_logger(
    dataframe: pd.DataFrame,
    timezone_mode: TimezoneMode,
) -> pd.DataFrame:
    """Normaliseer reeds gedecodeerde binaire loggerdata."""
    result = dataframe.copy()

    result["tijd"] = localize_datetime_series(
        result["tijd"],
        timezone_mode,
    )

    result["loggerwaarde"] = pd.to_numeric(
        result["loggerwaarde"],
        errors="coerce",
    )

    result["temperatuur_c"] = pd.to_numeric(
        result["temperatuur_c"],
        errors="coerce",
    )

    result = result.dropna(subset=["tijd"])
    result = result.sort_values("tijd")
    result = result.drop_duplicates("tijd", keep="last")
    result = result.reset_index(drop=True)

    if result.empty:
        raise ValueError("Geen geldige binaire loggerrecords gevonden.")

    return result


def prepare_knmi_data(
    dataframe: pd.DataFrame,
    date_column: str,
    time_column: str | None,
    pressure_column: str,
    timezone_mode: TimezoneMode,
    day_first: bool,
) -> pd.DataFrame:
    """Normaliseer een KNMI-tabel."""
    date_values = dataframe[date_column].astype("string").str.strip()

    if time_column:
        time_values = dataframe[time_column].astype("string").str.strip()
        datetime_values = date_values + " " + time_values
    else:
        datetime_values = date_values

    result = pd.DataFrame(
        {
            "tijd": pd.to_datetime(
                datetime_values,
                errors="coerce",
                dayfirst=day_first,
                format="mixed",
            ),
            "knmi_druk_bronwaarde": clean_numeric_series(
                dataframe[pressure_column]
            ),
        }
    )

    result["tijd"] = localize_datetime_series(
        result["tijd"],
        timezone_mode,
    )

    result = result.dropna(
        subset=["tijd", "knmi_druk_bronwaarde"]
    )

    result = (
        result.groupby("tijd", as_index=False)
        .agg({"knmi_druk_bronwaarde": "mean"})
        .sort_values("tijd")
        .reset_index(drop=True)
    )

    if len(result) < 2:
        raise ValueError(
            "Minimaal twee geldige KNMI-metingen zijn noodzakelijk."
        )

    return result


# =============================================================================
# Interpolatie en waterstandsberekening
# =============================================================================

def interpolate_knmi_pressure(
    logger_data: pd.DataFrame,
    knmi_data: pd.DataFrame,
    knmi_pressure_unit: str,
    maximum_gap_hours: float,
) -> pd.DataFrame:
    """Interpoleer KNMI-luchtdruk naar de logger-tijdstippen."""
    logger = logger_data.copy()
    knmi = knmi_data.copy()

    knmi["luchtdruk_pa"] = pressure_to_pa(
        knmi["knmi_druk_bronwaarde"],
        knmi_pressure_unit,
    )

    logger_times = logger["tijd"].astype("int64").to_numpy()
    knmi_times = knmi["tijd"].astype("int64").to_numpy()
    pressures = knmi["luchtdruk_pa"].to_numpy(dtype=float)

    interpolated = np.full(len(logger), np.nan)
    source_intervals = np.full(len(logger), np.nan)
    outside_range = np.zeros(len(logger), dtype=bool)

    positions = np.searchsorted(
        knmi_times,
        logger_times,
        side="left",
    )

    for index, position in enumerate(positions):
        logger_time = logger_times[index]

        if position == 0:
            if logger_time == knmi_timesinterpolated[index] = pressures[0]
                source_intervals[index] = 0.0
            else:
                outside_range[index] = True
            continue

        if position >= len(knmi_times):
            if logger_time == knmi_times[-1]:
                interpolated[index] = pressures[-1]
                source_intervals[index] = 0.0
            else:
                outside_range[index] = True
            continue

        previous_time = knmi_times[position - 1]
        next_time = knmi_times[position]
        previous_pressure = pressures[position - 1]
        next_pressure = pressures[position]

        if logger_time == previous_time:
            interpolated[index] = previous_pressure
            source_intervals[index] = 0.0
            continue

        if logger_time == next_time:
            interpolated[index] = next_pressure
            source_intervals[index] = 0.0
            continue

        interval_ns = next_time - previous_time

        if interval_ns <= 0:
            continue

        interval_hours = interval_ns / NANOSECONDS_PER_HOUR
        source_intervals[index] = interval_hours

        if interval_hours > maximum_gap_hours:
            continue

        fraction = (logger_time - previous_time) / interval_ns

        interpolated[index] = (
            previous_pressure
            + fraction * (next_pressure - previous_pressure)
        )

    logger["luchtdruk_pa"] = interpolated
    logger["knmi_broninterval_uur"] = source_intervals
    logger["buiten_knmi_periode"] = outside_range

    return logger


def calculate_water_levels(
    dataframe: pd.DataFrame,
    configuration: PeilfilterConfiguration,
) -> pd.DataFrame:
    """Bereken de waterkolom en waterstand ten opzichte van NAP."""
    result = dataframe.copy()

    temperature = result["temperatuur_c"].fillna(
        configuration.default_temperature_c
    )

    result["gebruikte_temperatuur_c"] = temperature

    if configuration.use_temperature_density:
        result["waterdichtheid_kg_m3"] = water_density_kg_m3(
            temperature
        )
    else:
        result["waterdichtheid_kg_m3"] = (
            DEFAULT_WATER_DENSITY_KG_M3
        )

    if configuration.logger_mode == "absolute_pressure":
        result["loggerdruk_pa"] = pressure_to_pa(
            result["loggerwaarde"],
            configuration.logger_unit,
        )

        result["wateroverdruk_pa"] = (
            result["loggerdruk_pa"] - result["luchtdruk_pa"]
        )

        result["waterkolom_m"] = (
            result["wateroverdruk_pa"]
            / (
                result["waterdichtheid_kg_m3"]
                * GRAVITY_M_S2
            )
        )
    else:
        result["loggerdruk_pa"] = np.nan
        result["wateroverdruk_pa"] = np.nan
        result["waterkolom_m"] = length_to_m(
            result["loggerwaarde"],
            configuration.logger_unit,
        )

    result["filter_id"] = configuration.filter_id
    result["bovenkant_peilbuis_nap_m"] = (
        configuration.top_casing_nap_m
    )
    result["kabellengte_m"] = configuration.cable_length_m
    result["sensorhoogte_nap_m"] = (
        configuration.sensor_elevation_nap_m
    )

    result["waterstand_nap_m"] = (
        result["sensorhoogte_nap_m"]
        + result["waterkolom_m"]
    )

    result["waterdiepte_tov_bovenkant_m"] = (
        configuration.top_casing_nap_m
        - result["waterstand_nap_m"]
    )

    result["luchtdruk_hpa"] = result["luchtdruk_pa"] / 100.0

    result["kwaliteitscode"] = "OK"

    result.loc[
        result["loggerwaarde"].isna(),
        "kwaliteitscode",
    ] = "LOGGER_ONTBREEKT"

    result.loc[
        result["luchtdruk_pa"].isna()
        & (configuration.logger_mode == "absolute_pressure"),
        "kwaliteitscode",
    ] = "LUCHTDRUK_ONTBREEKT"

    result.loc[
        result["waterkolom_m"] < 0,
        "kwaliteitscode",
    ] = "NEGATIEVE_WATERKOLOM"

    result.loc[
        (result["waterkolom_m"] > 100)
        | (result["waterkolom_m"] < -5),
        "kwaliteitscode",
    ] = "BUITEN_BEREIK"

    return result


# =============================================================================
# Export en grafieken
# =============================================================================

def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Exporteer resultaten naar een Nederlandse CSV."""
    export = dataframe.copy()

    for column in export.columns:
        if pd.api.types.is_datetime64_any_dtype(export[column]):
            export[column] = export[column].dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    return export.to_csv(
        index=False,
        sep=";",
        decimal=",",
        na_rep="",
        lineterminator="\n",
    ).encode("utf-8-sig")


def create_water_level_figure(dataframe: pd.DataFrame) -> go.Figure:
    """Maak een interactieve waterstandgrafiek."""
    valid = dataframe.dropna(
        subset=["tijd", "waterstand_nap_m"]
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=valid["tijd"],
            y=valid["waterstand_nap_m"],
            mode="lines",
            name="Waterstand",
            line={"color": "#0078D4", "width": 2},
            hovertemplate=(
                "%{x|%d-%m-%Y %H:%M}<br>"
                "%{y:.3f} m NAP"
                "<extra></extra>"
            ),
        )
    )

    if not dataframe.empty:
        figure.add_hline(
            y=float(
                dataframe["bovenkant_peilbuis_nap_m"].iloc[0]
            ),
            line_dash="dash",
            line_color="#D83B01",
            annotation_text="Bovenkant peilbuis",
        )

    figure.update_layout(
        title="Berekende grondwaterstand",
        xaxis_title="Datum en tijd",
        yaxis_title="Waterstand (m NAP)",
        hovermode="x unified",
    )

    return figure


# =============================================================================
# Streamlit-componenten
# =============================================================================

def render_timezone_selector(
    key: str,
    default: TimezoneMode,
) -> TimezoneMode:
    """Toon een tijdzoneselectie."""
    options: list[TimezoneMode] = [
        "Nederlandse lokale tijd",
        "UTC",
        "Geen tijdzonecorrectie",
    ]

    return st.selectbox(
        "Tijdzone in het bestand",
        options=options,
        index=options.index(default),
        key=key,
    )


def render_binary_field(
    label: str,
    key: str,
    default_offset: int,
    default_type: str,
    default_scale: float,
) -> BinaryFieldDefinition:
    """Toon invoervelden voor één binair numeriek veld."""
    st.markdown(f"**{label}**")

    columns = st.columns(4)

    offset = columns[0].number_input(
        "Byte-offset",
        min_value=0,
        value=default_offset,
        step=1,
        key=f"{key}_offset",
    )

    data_type = columns[1].selectbox(
        "Datatype",
        options=list(BINARY_VALUE_FORMATS),
        index=list(BINARY_VALUE_FORMATS).index(default_type),
        key=f"{key}_datatype",
    )

    scale = columns[2].number_input(
        "Schaalfactor",
        value=default_scale,
        format="%.10f",
        key=f"{key}_scale",
    )

    value_offset = columns[3].number_input(
        "Waarde-offset",
        value=0.0,
        format="%.10f",
        key=f"{key}_value_offset",
    )

    return BinaryFieldDefinition(
        offset=int(offset),
        data_type=data_type,
        scale=float(scale),
        value_offset=float(value_offset),
    )


def render_binary_configuration(
    file_bytes: bytes,
) -> BinaryDecoderConfiguration:
    """Toon alle configuratievelden voor de binaire decoder."""
    st.subheader("Binaire bestandsstructuur")

    st.warning(
        "Controleer de technische documentatie van de logger. "
        "Onjuiste offsets of datatypes kunnen plausibele maar foutieve "
        "meetwaarden veroorzaken."
    )

    preview_columns = st.columns(3)

    preview_start = preview_columns[0].number_input(
        "Startpositie hex-preview",
        min_value=0,
        max_value=max(0, len(file_bytes) - 1),
        value=0,
        step=16,
    )

    preview_length = preview_columns[1].number_input(
        "Aantal previewbytes",
        min_value=16,
        max_value=min(8_192, len(file_bytes)),
        value=min(512, len(file_bytes)),
        step=16,
    )

    preview_columns[2].metric(
        "Bestandsgrootte",
        f"{len(file_bytes):,} bytes",
    )

    st.code(
        create_hex_preview(
            file_bytes,
            start=int(preview_start),
            length=int(preview_length),
        ),
        language="text",
    )

    structure_columns = st.columns(3)

    header_size = structure_columns[0].number_input(
        "Headerlengte in bytes",
        min_value=0,
        max_value=max(0, len(file_bytes) - 1),
        value=0,
        step=1,
    )

    record_size = structure_columns[1].number_input(
        "Recordlengte in bytes",
        min_value=1,
        max_value=65_536,
        value=16,
        step=1,
    )

    byte_order = structure_columns[2].selectbox(
        "Bytevolgorde",
        options=["Little-endian", "Big-endian"],
    )

    remaining = len(file_bytes) - int(header_size)
    possible_records = (
        remaining // int(record_size)
        if record_size
        else 0
    )
    remainder = (
        remaining % int(record_size)
        if record_size
        else 0
    )

    st.info(
        f"Deze configuratie levert maximaal {possible_records:,} records "
        f"op, met {remainder} resterende bytes."
    )

    st.markdown("**Tijdstempel**")

    timestamp_mode = st.selectbox(
        "Tijdstempelopslag",
        options=[
            "Unix-tijd",
            "Excel-datum",
            "Tijd sinds aangepaste oorsprong",
            "Losse datumvelden",
        ],
    )

    timestamp_offset = 0
    timestamp_data_type = "Unsigned integer 32-bit"
    timestamp_unit = "seconden"
    timestamp_origin = datetime(1970, 1, 1)

    year_offset = 0
    month_offset = 2
    day_offset = 3
    hour_offset = 4
    minute_offset = 5
    second_offset = 6
    year_data_type = "Unsigned integer 16-bit"

    if timestamp_mode == "Losse datumvelden":
        date_columns_1 = st.columns(4)
        date_columns_2 = st.columns(3)

        year_offset = int(
            date_columns_1[0].number_input(
                "Offset jaar",
                min_value=0,
                value=0,
                step=1,
            )
        )

        year_data_type = date_columns_1[1].selectbox(
            "Datatype jaar",
            options=[
                "Unsigned integer 16-bit",
                "Signed integer 16-bit",
                "Unsigned integer 32-bit",
            ],
        )

        month_offset = int(
            date_columns_1[2].number_input(
                "Offset maand",
                min_value=0,
                value=2,
                step=1,
            )
        )

        day_offset = int(
            date_columns_1[3].number_input(
                "Offset dag",
                min_value=0,
                value=3,
                step=1,
            )
        )

        hour_offset = int(
            date_columns_2[0].number_input(
                "Offset uur",
                min_value=0,
                value=4,
                step=1,
            )
        )

        minute_offset = int(
            date_columns_2[1].number_input(
                "Offset minuut",
                min_value=0,
                value=5,
                step=1,
            )
        )

        second_offset = int(
            date_columns_2[2].number_input(
                "Offset seconde",
                min_value=0,
                value=6,
                step=1,
            )
        )

    else:
        timestamp_columns = st.columns(4)

        timestamp_offset = int(
            timestamp_columns[0].number_input(
                "Offset tijdstempel",
                min_value=0,
                value=0,
                step=1,
            )
        )

        timestamp_data_type = timestamp_columns[1].selectbox(
            "Datatype tijdstempel",
            options=list(BINARY_VALUE_FORMATS),
            index=list(BINARY_VALUE_FORMATS).index(
                "Unsigned integer 32-bit"
            ),
        )

        if timestamp_mode == "Excel-datum":
            timestamp_unit = "dagen"
            timestamp_columns[2].text_input(
                "Eenheid",
                value="dagen",
                disabled=True,
            )
        else:
            unit_options = [
                "seconden",
                "milliseconden",
                "microseconden",
            ]

            if timestamp_mode == "Tijd sinds aangepaste oorsprong":
                unit_options.append("dagen")

            timestamp_unit = timestamp_columns[2].selectbox(
                "Tijdseenheid",
                options=unit_options,
            )

        if timestamp_mode == "Tijd sinds aangepaste oorsprong":
            origin_date = timestamp_columns[3].date_input(
                "Oorsprongsdatum",
                value=datetime(1970, 1, 1).date(),
            )
            timestamp_origin = datetime.combine(
                origin_date,
                datetime.min.time(),
            )

    pressure_field = render_binary_field(
        label="Loggerdruk of waterkolom",
        key="binary_pressure",
        default_offset=4,
        default_type="Float 32-bit",
        default_scale=1.0,
    )

    has_temperature = st.checkbox(
        "Het binaire record bevat temperatuur",
        value=True,
    )

    temperature_field: BinaryFieldDefinition | None = None

    if has_temperature:
        temperature_field = render_binary_field(
            label="Temperatuur",
            key="binary_temperature",
            default_offset=8,
            default_type="Float 32-bit",
            default_scale=1.0,
        )

    return BinaryDecoderConfiguration(
        header_size=int(header_size),
        record_size=int(record_size),
        byte_order=byte_order,
        timestamp_mode=timestamp_mode,
        timestamp_offset=timestamp_offset,
        timestamp_data_type=timestamp_data_type,
        timestamp_unit=timestamp_unit,
        timestamp_origin=timestamp_origin,
        pressure_field=pressure_field,
        temperature_field=temperature_field,
        year_offset=year_offset,
        month_offset=month_offset,
        day_offset=day_offset,
        hour_offset=hour_offset,
        minute_offset=minute_offset,
        second_offset=second_offset,
        year_data_type=year_data_type,
    )


def render_text_table_settings(
    prefix: str,
) -> tuple[str, str, int]:
    """Toon instellingen voor een teksttabel."""
    columns = st.columns(3)

    separator = columns[0].selectbox(
        "Scheidingsteken",
        options=[
            "Automatisch",
            "Puntkomma",
            "Komma",
            "Tab",
            "Pipe",
            "Spaties",
        ],
        key=f"{prefix}_separator",
    )

    decimal = columns[1].selectbox(
        "Decimaalteken",
        options=[",", "."],
        key=f"{prefix}_decimal",
    )

    header_row = columns[2].number_input(
        "Regelnummer tabelkop",
        min_value=0,
        value=0,
        step=1,
        key=f"{prefix}_header",
    )

    return separator, decimal, int(header_row)


def main() -> None:
    """Start de Streamlit-applicatie."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="💧",
        layout="wide",
    )

    st.title("💧 Waterstanden omrekenen naar NAP")

    st.markdown(
        """
        Deze applicatie verwerkt tekstuele en binaire loggerbestanden,
        compenseert absolute loggerdruk met KNMI-luchtdruk en berekent
        waterstanden ten opzichte van NAP.
        """
    )

    st.header("1. Meetopstelling")

    configuration_columns = st.columns(4)

    filter_id = configuration_columns[0].text_input(
        "Peilfilter-ID",
        value="PB01",
    )

    top_casing_nap_m = configuration_columns[1].number_input(
        "Bovenkant peilbuis (m NAP)",
        value=1.000,
        step=0.001,
        format="%.3f",
    )

    cable_length_m = configuration_columns[2].number_input(
        "Kabellengte tot druksensor (m)",
        min_value=0.0,
        value=5.000,
        step=0.001,
        format="%.3f",
    )

    logger_mode_label = configuration_columns[3].selectbox(
        "Type loggerwaarde",
        options=list(LOGGER_MODES),
    )

    logger_mode = LOGGER_MODES[logger_mode_label]

    unit_columns = st.columns(4)

    if logger_mode == "absolute_pressure":
        logger_unit = unit_columns[0].selectbox(
            "Eenheid loggerdruk",
            options=list(PRESSURE_UNIT_FACTORS_TO_PA),
            index=list(PRESSURE_UNIT_FACTORS_TO_PA).index("hPa"),
        )
    else:
        logger_unit = unit_columns[0].selectbox(
            "Eenheid waterkolom",
            options=list(LENGTH_UNIT_FACTORS_TO_M),
        )

    knmi_unit = unit_columns[1].selectbox(
        "Eenheid KNMI-luchtdruk",
        options=list(PRESSURE_UNIT_FACTORS_TO_PA),
        index=list(PRESSURE_UNIT_FACTORS_TO_PA).index("0,1 hPa"),
        disabled=logger_mode != "absolute_pressure",
    )

    default_temperature = unit_columns[2].number_input(
        "Standaard watertemperatuur (°C)",
        min_value=0.0,
        max_value=40.0,
        value=12.0,
        step=0.1,
    )

    use_temperature_density = unit_columns[3].checkbox(
        "Temperatuurcorrectie",
        value=True,
    )

    sensor_elevation = top_casing_nap_m - cable_length_m

    st.info(
        f"Sensorhoogte: **{sensor_elevation:.3f} m NAP**"
    )

    st.header("2. Loggerbestand")

    logger_file = st.file_uploader(
        "Upload loggerbestand",
        type=["dat", "bin", "csv", "txt"],
    )

    if logger_file is None:
        st.stop()

    logger_bytes = logger_file.getvalue()

    if len(logger_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(
            f"Het bestand is groter dan {MAX_FILE_SIZE_MB} MB."
        )
        st.stop()

    detected_kind, detected_encoding, confidence = detect_file_kind(
        logger_bytes
    )

    kind_label = (
        "Tekstbestand"
        if detected_kind == "text"
        else "Binair bestand"
    )

    st.info(
        f"Automatische detectie: **{kind_label}** "
        f"met betrouwbaarheid {confidence:.0%}."
    )

    file_mode = st.radio(
        "Verwerkingsmethode",
        options=[
            "Automatisch",
            "Tekstbestand",
            "Binair bestand met vaste records",
        ],
        horizontal=True,
    )

    if file_mode == "Automatisch":
        effective_mode = (
            "Tekstbestand"
            if detected_kind == "text"
            else "Binair bestand met vaste records"
        )
    else:
        effective_mode = file_mode

    logger_data: pd.DataFrame | None = None

    if effective_mode == "Tekstbestand":
        separator, decimal, header_row = render_text_table_settings(
            "logger"
        )

        try:
            logger_raw = read_text_table(
                file_bytes=logger_bytes,
                encoding=detected_encoding,
                separator_name=separator,
                decimal=decimal,
                header_row=header_row,
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        st.dataframe(
            logger_raw.head(20),
            use_container_width=True,
        )

        text_columns = list(logger_raw.columns)

        mapping_columns = st.columns(4)

        date_column = mapping_columns[0].selectbox(
            "Datum- of tijdkolom",
            options=text_columns,
        )

        time_selection = mapping_columns[1].selectbox(
            "Aparte tijdkolom",
            options=["Geen"] + text_columns,
        )

        value_column = mapping_columns[2].selectbox(
            "Loggerwaarde",
            options=text_columns,
        )

        temperature_selection = mapping_columns[3].selectbox(
            "Temperatuurkolom",
            options=["Geen"] + text_columns,
        )

        logger_timezone = render_timezone_selector(
            "logger_text_timezone",
            "Nederlandse lokale tijd",
        )

        day_first = st.checkbox(
            "Datum gebruikt dag-maand-jaar",
            value=True,
        )

        try:
            logger_data = prepare_text_logger(
                dataframe=logger_raw,
                date_column=date_column,
                time_column=(
                    None
                    if time_selection == "Geen"
                    else time_selection
                ),
                value_column=value_column,
                temperature_column=(
                    None
                    if temperature_selection == "Geen"
                    else temperature_selection
                ),
                timezone_mode=logger_timezone,
                day_first=day_first,
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    else:
        binary_configuration = render_binary_configuration(
            logger_bytes
        )

        logger_timezone = render_timezone_selector(
            "logger_binary_timezone",
            "Nederlandse lokale tijd",
        )

        if st.button(
            "Test binaire decoder",
            use_container_width=True,
        ):
            try:
                preview_data = decode_binary_logger(
                    logger_bytes,
                    binary_configuration,
                    maximum_records=100,
                )

                st.success(
                    "De eerste binaire records zijn gedecodeerd."
                )

                st.dataframe(
                    preview_data,
                    use_container_width=True,
                )
            except ValueError as exc:
                st.error(str(exc))

        try:
            decoded_binary = decode_binary_logger(
                logger_bytes,
                binary_configuration,
            )

            logger_data = prepare_binary_logger(
                decoded_binary,
                logger_timezone,
            )

        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    if logger_data is None or logger_data.empty:
        st.error("Er zijn geen loggergegevens beschikbaar.")
        st.stop()

    st.subheader("Preview genormaliseerde loggerdata")

    st.dataframe(
        logger_data.head(50),
        use_container_width=True,
    )

    if logger_data["tijd"].notna().any():
        st.caption(
            "Periode logger: "
            f"{logger_data['tijd'].min():%d-%m-%Y %H:%M:%S} tot "
            f"{logger_data['tijd'].max():%d-%m-%Y %H:%M:%S}"
        )

    if logger_mode == "absolute_pressure":
        st.header("3. KNMI-luchtdruk")

        knmi_file = st.file_uploader(
            "Upload KNMI-bestand",
            type=["dat", "csv", "txt"],
        )

        if knmi_file is None:
            st.stop()

        knmi_bytes = knmi_file.getvalue()
        knmi_kind, knmi_encoding, _ = detect_file_kind(knmi_bytes)

        if knmi_kind != "text":
            st.error(
                "Het KNMI-bestand moet een tekstueel tabelbestand zijn."
            )
            st.stop()

        knmi_separator, knmi_decimal, knmi_header = (
            render_text_table_settings("knmi")
        )

        try:
            knmi_raw = read_text_table(
                file_bytes=knmi_bytes,
                encoding=knmi_encoding,
                separator_name=knmi_separator,
                decimal=knmi_decimal,
                header_row=knmi_header,
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        st.dataframe(
            knmi_raw.head(20),
            use_container_width=True,
        )

        knmi_columns = list(knmi_raw.columns)
        knmi_mapping = st.columns(3)

        knmi_date_column = knmi_mapping[0].selectbox(
            "KNMI datum- of tijdkolom",
            options=knmi_columns,
        )

        knmi_time_selection = knmi_mapping[1].selectbox(
            "KNMI aparte tijdkolom",
            options=["Geen"] + knmi_columns,
        )

        knmi_pressure_column = knmi_mapping[2].selectbox(
            "KNMI-luchtdrukkrom",
            options=knmi_columns,
        )

        knmi_timezone = render_timezone_selector(
            "knmi_timezone",
            "UTC",
        )

        knmi_day_first = st.checkbox(
            "KNMI-datum gebruikt dag-maand-jaar",
            value=True,
        )

        maximum_gap_hours = st.number_input(
            "Maximaal te interpoleren KNMI-datagat in uren",
            min_value=1.0,
            max_value=168.0,
            value=6.0,
            step=1.0,
        )

        try:
            knmi_data = prepare_knmi_data(
                dataframe=knmi_raw,
                date_column=knmi_date_column,
                time_column=(
                    None
                    if knmi_time_selection == "Geen"
                    else knmi_time_selection
                ),
                pressure_column=knmi_pressure_column,
                timezone_mode=knmi_timezone,
                day_first=knmi_day_first,
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    else:
        knmi_data = pd.DataFrame()
        maximum_gap_hours = 6.0

    st.header("4. Berekening")

    if not st.button(
        "Bereken waterstanden",
        type="primary",
        use_container_width=True,
    ):
        st.stop()

    configuration = PeilfilterConfiguration(
        filter_id=filter_id.strip() or "Onbekend",
        top_casing_nap_m=float(top_casing_nap_m),
        cable_length_m=float(cable_length_m),
        logger_mode=logger_mode,
        logger_unit=logger_unit,
        knmi_pressure_unit=knmi_unit,
        default_temperature_c=float(default_temperature),
        use_temperature_density=use_temperature_density,
    )

    try:
        if logger_mode == "absolute_pressure":
            combined = interpolate_knmi_pressure(
                logger_data=logger_data,
                knmi_data=knmi_data,
                knmi_pressure_unit=knmi_unit,
                maximum_gap_hours=float(maximum_gap_hours),
            )
        else:
            combined = logger_data.copy()
            combined["luchtdruk_pa"] = np.nan
            combined["knmi_broninterval_uur"] = np.nan
            combined["buiten_knmi_periode"] = False

        result = calculate_water_levels(
            combined,
            configuration,
        )

    except Exception as exc:
        LOGGER.exception("Berekening mislukt")
        st.error(f"De berekening is mislukt: {exc}")
        st.stop()

    valid = result.dropna(subset=["waterstand_nap_m"])

    if valid.empty:
        st.error(
            "Er zijn geen geldige waterstanden berekend. Controleer "
            "vooral de binaire decoder, drukeenheden en tijdzones."
        )
        st.stop()

    st.success(
        f"{len(valid):,} geldige waterstanden berekend."
    )

    metrics = st.columns(4)

    metrics[0].metric(
        "Minimum",
        f"{valid['waterstand_nap_m'].min():.3f} m NAP",
    )

    metrics[1].metric(
        "Gemiddelde",
        f"{valid['waterstand_nap_m'].mean():.3f} m NAP",
    )

    metrics[2].metric(
        "Maximum",
        f"{valid['waterstand_nap_m'].max():.3f} m NAP",
    )

    metrics[3].metric(
        "Geldige records",
        f"{len(valid):,}",
    )

    st.plotly_chart(
        create_water_level_figure(result),
        use_container_width=True,
    )

    output_columns = [
        column
        for column in [
            "filter_id",
            "tijd",
            "recordnummer",
            "byte_offset",
            "loggerwaarde",
            "loggerdruk_pa",
            "luchtdruk_hpa",
            "wateroverdruk_pa",
            "gebruikte_temperatuur_c",
            "waterdichtheid_kg_m3",
            "waterkolom_m",
            "sensorhoogte_nap_m",
            "waterstand_nap_m",
            "waterdiepte_tov_bovenkant_m",
            "knmi_broninterval_uur",
            "buiten_knmi_periode",
            "kwaliteitscode",
            "decodeerfout",
        ]
        if column in result.columns
    ]

    st.dataframe(
        result[output_columns],
        use_container_width=True,
        hide_index=True,
    )

    safe_filter_id = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        configuration.filter_id,
    ).strip("_") or "peilfilter"

    st.download_button(
        "Download berekende waterstanden",
        data=dataframe_to_csv_bytes(result[output_columns]),
        file_name=f"{safe_filter_id}_waterstanden_nap.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
