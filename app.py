"""
Streamlit-applicatie voor het omrekenen van drukloggergegevens naar
grondwaterstanden ten opzichte van NAP.

De applicatie ondersteunt:

1. Absolute druklogger met atmosferische compensatie.
2. Logger die al een waterkolom registreert.
3. Verschillende scheidingstekens, decimalen en tekstcoderingen.
4. Interpolatie van lokale KNMI-luchtdruk naar logger-tijdstippen.
5. Correctie voor temperatuurafhankelijke waterdichtheid.
6. Tijdzonecorrectie voor UTC en lokale Nederlandse tijd.
7. Kwaliteitscontrole en export naar CSV.

Berekening bij een absolute-druklogger:

    sensor_hoogte_nap = bovenkant_peilbuis_nap - kabellengte
    waterkolom = (loggerdruk_absoluut - luchtdruk) / (rho * g)
    waterstand_nap = sensor_hoogte_nap + waterkolom

De gebruikte kabellengte moet worden gemeten van de bovenkant van de
peilbuis tot het drukpunt of sensormembraan van de logger.
"""

from __future__ import annotations

import io
import logging
import math
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================================
# Configuratie
# ============================================================================

APP_TITLE: Final[str] = "Grondwaterstand naar NAP"
GRAVITY: Final[float] = 9.80665
DEFAULT_WATER_DENSITY: Final[float] = 998.2
MAX_FILE_SIZE_MB: Final[int] = 100

DATETIME_COLUMN_HINTS: Final[tuple[str, ...]] = (
    "datetime",
    "date time",
    "timestamp",
    "tijdstip",
    "datumtijd",
    "datum tijd",
    "date",
    "datum",
    "time",
    "tijd",
)

PRESSURE_COLUMN_HINTS: Final[tuple[str, ...]] = (
    "pressure",
    "druk",
    "press",
    "p abs",
    "p_abs",
    "absolute pressure",
    "luchtdruk",
    "barometer",
    "barometric",
    "p",
)

TEMPERATURE_COLUMN_HINTS: Final[tuple[str, ...]] = (
    "temperature",
    "temperatuur",
    "temp",
    "water temperature",
    "watertemperatuur",
    "t",
)

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
    "mH₂O": DEFAULT_WATER_DENSITY * GRAVITY,
    "cmH₂O": DEFAULT_WATER_DENSITY * GRAVITY / 100.0,
}

LENGTH_UNIT_FACTORS_TO_M: Final[dict[str, float]] = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
}

DATETIME_FORMAT_OPTIONS: Final[dict[str, str | None]] = {
    "Automatisch herkennen": None,
    "DD-MM-YYYY HH:MM:SS": "%d-%m-%Y %H:%M:%S",
    "DD/MM/YYYY HH:MM:SS": "%d/%m/%Y %H:%M:%S",
    "YYYY-MM-DD HH:MM:SS": "%Y-%m-%d %H:%M:%S",
    "YYYY/MM/DD HH:MM:SS": "%Y/%m/%d %H:%M:%S",
    "YYYYMMDDHHMM": "%Y%m%d%H%M",
    "YYYYMMDDHH": "%Y%m%d%H",
}

TimezoneMode = Literal[
    "Nederlandse lokale tijd",
    "UTC",
    "Geen tijdzonecorrectie",
]


# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


# ============================================================================
# Datamodellen
# ============================================================================

@dataclass(frozen=True)
class PeilfilterConfiguration:
    """Configuratie van een peilfilter en de daarin geplaatste logger."""

    filter_id: str
    top_casing_nap_m: float
    cable_length_m: float
    logger_mode: str
    logger_pressure_unit: str
    knmi_pressure_unit: str
    water_column_unit: str = "m"
    default_water_temperature_c: float = 20.0
    use_temperature_density: bool = True

    @property
    def sensor_elevation_nap_m(self) -> float:
        """Bereken de hoogte van de druksensor ten opzichte van NAP."""
        return self.top_casing_nap_m - self.cable_length_m


@dataclass(frozen=True)
class DataQualitySummary:
    """Samenvatting van de uitgevoerde kwaliteitscontrole."""

    total_rows: int
    valid_rows: int
    missing_logger_values: int
    missing_knmi_values: int
    negative_water_columns: int
    extrapolated_rows: int
    implausible_rows: int


# ============================================================================
# Bestandsverwerking
# ============================================================================

def decode_file(file_bytes: bytes) -> str:
    """
    Decodeer een tekstbestand met een aantal gebruikelijke coderingen.

    Raises:
        ValueError: Als geen ondersteunde codering werkt.
    """
    encodings = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

    for encoding in encodings:
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(
        "Het bestand kon niet als tekst worden gelezen. "
        "Sla het bestand op als UTF-8, Windows-1252 of Latin-1."
    )


def remove_comment_and_metadata_lines(text: str) -> str:
    """
    Verwijder lege regels en veelvoorkomende metadataregels.

    Een kommentaarregel begint met #, // of !. Regels voor de daadwerkelijke
    tabelkop worden alleen verwijderd als ze duidelijk geen tabelstructuur
    bevatten.
    """
    lines = text.splitlines()
    cleaned_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith(("#", "//", "!")):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def detect_separator(text: str) -> str:
    """
    Detecteer het meest waarschijnlijke scheidingsteken.

    Ondersteunde scheidingstekens:
    - puntkomma
    - tab
    - komma
    - pipe
    """
    candidate_lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "//", "!"))
    ]

    if not candidate_lines:
        raise ValueError("Het bestand bevat geen leesbare gegevens.")

    sample_lines = candidate_lines[:20]
    candidates = (";", "\t", ",", "|")

    scores: dict[str, float] = {}

    for separator in candidates:
        counts = [line.count(separator) for line in sample_lines]
        nonzero_counts = [count for count in counts if count > 0]

        if not nonzero_counts:
            scores[separator] = -1.0
            continue

        mean_count = float(np.mean(nonzero_counts))
        variability = float(np.std(nonzero_counts))
        coverage = len(nonzero_counts) / len(sample_lines)

        scores[separator] = coverage * 10.0 + mean_count - variability

    detected = max(scores, key=scores.get)

    if scores[detected] < 0:
        return r"\s+"

    return detected


def find_probable_header_row(text: str, separator: str) -> int:
    """
    Zoek de meest waarschijnlijke tabelkop.

    Dit helpt bij DAT- en KNMI-bestanden die enkele metadataregels boven de
    kolomnamen bevatten.
    """
    lines = text.splitlines()
    best_row = 0
    best_score = -math.inf

    for index, line in enumerate(lines[:100]):
        stripped = line.strip()

        if not stripped:
            continue

        if separator == r"\s+":
            parts = re.split(r"\s+", stripped)
        else:
            parts = [part.strip() for part in stripped.split(separator)]

        if len(parts) < 2:
            continue

        letters = sum(
            bool(re.search(r"[A-Za-zÀ-ÿ]", part))
            for part in parts
        )
        unique_parts = len(set(parts))
        unnamed_parts = sum(not part for part in parts)

        score = (
            len(parts)
            + letters * 2
            + unique_parts * 0.2
            - unnamed_parts * 2
        )

        lower_line = stripped.lower()

        if any(hint in lower_line for hint in DATETIME_COLUMN_HINTS):
            score += 5

        if any(hint in lower_line for hint in PRESSURE_COLUMN_HINTS):
            score += 5

        if score > best_score:
            best_score = score
            best_row = index

    return best_row


def normalize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Maak kolomnamen uniek en verwijder overtollige spaties."""
    result = dataframe.copy()
    new_columns: list[str] = []
    seen: dict[str, int] = {}

    for column in result.columns:
        name = str(column).strip()
        name = re.sub(r"\s+", " ", name)
        name = name or "kolom"

        count = seen.get(name, 0)

        if count:
            unique_name = f"{name}_{count + 1}"
        else:
            unique_name = name

        seen[name] = count + 1
        new_columns.append(unique_name)

    result.columns = new_columns
    return result


@st.cache_data(show_spinner=False)
def read_tabular_file(
    file_bytes: bytes,
    separator_option: str,
    decimal_option: str,
    skip_rows: int,
) -> pd.DataFrame:
    """
    Lees een logger- of KNMI-tekstbestand in als DataFrame.

    Args:
        file_bytes: Inhoud van het geüploade bestand.
        separator_option: Gekozen scheidingsteken of 'Automatisch'.
        decimal_option: Decimaalteken, punt of komma.
        skip_rows: Aantal regels dat voor de tabel moet worden overgeslagen.

    Returns:
        Ingelezen DataFrame.
    """
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(
            f"Het bestand is groter dan {MAX_FILE_SIZE_MB} MB."
        )

    text = decode_file(file_bytes)
    text = remove_comment_and_metadata_lines(text)

    if not text.strip():
        raise ValueError("Het bestand bevat geen gegevensregels.")

    separator_map = {
        "Automatisch": None,
        "Puntkomma": ";",
        "Tab": "\t",
        "Komma": ",",
        "Pipe": "|",
        "Spaties": r"\s+",
    }

    separator = separator_map[separator_option]

    if separator is None:
        separator = detect_separator(text)

    if skip_rows < 0:
        header_row = find_probable_header_row(text, separator)
    else:
        header_row = skip_rows

    try:
        dataframe = pd.read_csv(
            io.StringIO(text),
            sep=separator,
            header=header_row,
            decimal=decimal_option,
            engine="python",
            dtype=str,
            on_bad_lines="skip",
        )
    except Exception as exc:
        raise ValueError(
            "Het bestand kon niet als tabel worden gelezen. "
            "Controleer het scheidingsteken, het decimaalteken en het aantal "
            "over te slaan regels."
        ) from exc

    dataframe = normalize_column_names(dataframe)
    dataframe = dataframe.dropna(axis=0, how="all")
    dataframe = dataframe.dropna(axis=1, how="all")

    if dataframe.empty:
        raise ValueError(
            "Na het inlezen zijn geen gegevens overgebleven."
        )

    return dataframe


# ============================================================================
# Conversiehulpfuncties
# ============================================================================

def clean_numeric_series(series: pd.Series) -> pd.Series:
    """
    Converteer een tekstkolom robuust naar numerieke waarden.

    De functie verwerkt onder meer:
    - decimale komma's;
    - duizendtallen;
    - witruimte;
    - eenheden achter numerieke waarden;
    - KNMI missings zoals -9999.
    """
    text = series.astype("string").str.strip()

    missing_values = {
        "",
        "na",
        "n/a",
        "nan",
        "none",
        "null",
        "-9999",
        "-999.9",
        "-999",
    }

    text = text.mask(text.str.lower().isin(missing_values))

    text = text.str.replace("\u00a0", "", regex=False)
    text = text.str.replace(" ", "", regex=False)

    both_separators = text.str.contains(",", na=False) & text.str.contains(
        r"\.", na=False
    )

    comma_after_dot = both_separators & (
        text.str.rfind(",") > text.str.rfind(".")
    )

    text = text.where(
        ~comma_after_dot,
        text.str.replace(".", "", regex=False).str.replace(
            ",", ".", regex=False
        ),
    )

    dot_after_comma = both_separators & ~comma_after_dot

    text = text.where(
        ~dot_after_comma,
        text.str.replace(",", "", regex=False),
    )

    only_comma = text.str.contains(",", na=False) & ~text.str.contains(
        r"\.", na=False
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


def parse_datetime_series(
    dataframe: pd.DataFrame,
    date_column: str,
    time_column: str | None,
    datetime_format: str | None,
    day_first: bool,
) -> pd.Series:
    """
    Maak één tijdreeks uit een datumkolom en optionele tijdkolom.
    """
    if date_column not in dataframe.columns:
        raise ValueError(f"Datumkolom '{date_column}' bestaat niet.")

    date_values = dataframe[date_column].astype("string").str.strip()

    if time_column and time_column != "(geen aparte tijdkolom)":
        if time_column not in dataframe.columns:
            raise ValueError(f"Tijdkolom '{time_column}' bestaat niet.")

        time_values = dataframe[time_column].astype("string").str.strip()
        combined = date_values + " " + time_values
    else:
        combined = date_values

    try:
        if datetime_format:
            parsed = pd.to_datetime(
                combined,
                format=datetime_format,
                errors="coerce",
            )
        else:
            parsed = pd.to_datetime(
                combined,
                errors="coerce",
                dayfirst=day_first,
                format="mixed",
            )
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "De datum- en tijdwaarden konden niet worden verwerkt."
        ) from exc

    return parsed


def localize_datetime_series(
    series: pd.Series,
    timezone_mode: TimezoneMode,
) -> pd.Series:
    """
    Zet tijdstempels om naar tijdzone-onafhankelijke Nederlandse lokale tijd.

    Daardoor zijn logger- en KNMI-reeksen goed vergelijkbaar.
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
        return localized.dt.tz_convert("Europe/Amsterdam").dt.tz_localize(None)

    localized = parsed.dt.tz_localize(
        "Europe/Amsterdam",
        ambiguous="NaT",
        nonexistent="NaT",
    )
    return localized.dt.tz_localize(None)


def pressure_to_pa(values: pd.Series, unit: str) -> pd.Series:
    """Converteer drukwaarden naar pascal."""
    if unit not in PRESSURE_UNIT_FACTORS_TO_PA:
        raise ValueError(f"Onbekende drukeenheid: {unit}")

    return values * PRESSURE_UNIT_FACTORS_TO_PA[unit]


def length_to_m(values: pd.Series, unit: str) -> pd.Series:
    """Converteer lengtes naar meter."""
    if unit not in LENGTH_UNIT_FACTORS_TO_M:
        raise ValueError(f"Onbekende lengte-eenheid: {unit}")

    return values * LENGTH_UNIT_FACTORS_TO_M[unit]


def water_density_kg_m3(temperature_c: pd.Series) -> pd.Series:
    """
    Bereken de dichtheid van zoet water als functie van de temperatuur.

    De formule is geschikt voor normale grondwatertemperaturen en geeft de
    dichtheid in kg/m³.
    """
    temperature = temperature_c.clip(lower=0.0, upper=40.0)

    numerator = (
        (temperature + 288.9414)
        * (temperature - 3.9863) ** 2
    )
    denominator = 508_929.2 * (temperature + 68.12963)

    return 1_000.0 * (1.0 - numerator / denominator)


def detect_suggested_column(
    columns: list[str],
    hints: tuple[str, ...],
) -> str:
    """Geef de meest waarschijnlijke kolom op basis van naamherkenning."""
    lower_columns = [column.lower().strip() for column in columns]

    for hint in hints:
        for original, lower in zip(columns, lower_columns):
            if lower == hint:
                return original

    for hint in hints:
        for original, lower in zip(columns, lower_columns):
            if hint in lower:
                return original

    return columns[0]


# ============================================================================
# Verwerking logger- en KNMI-gegevens
# ============================================================================

def prepare_logger_data(
    dataframe: pd.DataFrame,
    date_column: str,
    time_column: str | None,
    value_column: str,
    temperature_column: str | None,
    datetime_format: str | None,
    day_first: bool,
    timezone_mode: TimezoneMode,
) -> pd.DataFrame:
    """Normaliseer loggergegevens naar een standaardstructuur."""
    result = pd.DataFrame()

    result["tijd"] = parse_datetime_series(
        dataframe=dataframe,
        date_column=date_column,
        time_column=time_column,
        datetime_format=datetime_format,
        day_first=day_first,
    )

    result["tijd"] = localize_datetime_series(
        result["tijd"],
        timezone_mode,
    )

    result["loggerwaarde"] = clean_numeric_series(dataframe[value_column])

    if temperature_column and temperature_column != "(geen temperatuurkolom)":
        result["temperatuur_c"] = clean_numeric_series(
            dataframe[temperature_column]
        )
    else:
        result["temperatuur_c"] = np.nan

    result["logger_rijnummer"] = np.arange(1, len(result) + 1)

    result = result.dropna(subset=["tijd"])
    result = result.sort_values("tijd")
    result = result.drop_duplicates(subset=["tijd"], keep="last")
    result = result.reset_index(drop=True)

    if result.empty:
        raise ValueError(
            "Er zijn geen geldige logger-tijdstempels gevonden."
        )

    return result


def prepare_knmi_data(
    dataframe: pd.DataFrame,
    date_column: str,
    time_column: str | None,
    pressure_column: str,
    datetime_format: str | None,
    day_first: bool,
    timezone_mode: TimezoneMode,
) -> pd.DataFrame:
    """Normaliseer KNMI-gegevens naar een standaardstructuur."""
    result = pd.DataFrame()

    result["tijd"] = parse_datetime_series(
        dataframe=dataframe,
        date_column=date_column,
        time_column=time_column,
        datetime_format=datetime_format,
        day_first=day_first,
    )

    result["tijd"] = localize_datetime_series(
        result["tijd"],
        timezone_mode,
    )

    result["knmi_druk_bronwaarde"] = clean_numeric_series(
        dataframe[pressure_column]
    )

    result = result.dropna(
        subset=["tijd", "knmi_druk_bronwaarde"]
    )
    result = result.sort_values("tijd")

    if result.empty:
        raise ValueError(
            "Er zijn geen geldige KNMI-tijdstempels en drukwaarden gevonden."
        )

    # Gemiddelde gebruiken als meerdere metingen hetzelfde tijdstip hebben.
    result = (
        result.groupby("tijd", as_index=False)
        .agg({"knmi_druk_bronwaarde": "mean"})
        .sort_values("tijd")
        .reset_index(drop=True)
    )

    return result


def interpolate_knmi_pressure(
    logger_data: pd.DataFrame,
    knmi_data: pd.DataFrame,
    knmi_pressure_unit: str,
    maximum_gap_hours: float,
) -> pd.DataFrame:
    """
    Interpoleer de KNMI-luchtdruk lineair naar logger-tijdstippen.

    Meetpunten buiten het KNMI-tijdsbereik worden niet geëxtrapoleerd.
    Daarnaast worden interpolaties over een te groot KNMI-datagat afgekeurd.
    """
    logger = logger_data.copy()
    knmi = knmi_data.copy()

    knmi["luchtdruk_pa"] = pressure_to_pa(
        knmi["knmi_druk_bronwaarde"],
        knmi_pressure_unit,
    )

    logger_times_ns = logger["tijd"].astype("int64").to_numpy()
    knmi_times_ns = knmi["tijd"].astype("int64").to_numpy()
    pressure_values = knmi["luchtdruk_pa"].to_numpy(dtype=float)

    interpolated = np.full(len(logger), np.nan, dtype=float)
    source_gap_hours = np.full(len(logger), np.nan, dtype=float)
    is_outside_range = np.zeros(len(logger), dtype=bool)

    if len(knmi) < 2:
        raise ValueError(
            "Voor interpolatie zijn minimaal twee geldige KNMI-metingen nodig."
        )

    positions = np.searchsorted(knmi_times_ns, logger_times_ns)

    for index, position in enumerate(positions):
        logger_time = logger_times_ns[index]

        if position == 0:
            if logger_time == knmi_times_ns[0]:
                source_gap_hours[index] = 0.0
            else:
                is_outside_range[index] = True
            continue

        if position >= len(knmi_times_ns):
            if logger_time == knmi_times_ns[-1]:
                interpolated[index] = pressure_values[-1]
                source_gap_hours[index] = 0.0
            else:
                is_outside_range[index] = True
            continue

        previous_time = knmi_times_ns[position - 1]
        next_time = knmi_times_ns[position]
        previous_pressure = pressure_values[position - 1]
        next_pressure = pressure_values[position]

        if logger_time == previous_time:
            interpolated[index] = previous_pressure
            source_gap_hours[index] = 0.0
            continue

        if logger_time == next_time:
            interpolated[index] = next_pressure
            source_gap_hours[index] = 0.0
            continue

        interval_ns = next_time - previous_time
        interval_hours = interval_ns / 3_600_000_000_000

        source_gap_hours[index] = interval_hours

        if interval_hours > maximum_gap_hours:
            continue

        fraction = (logger_time - previous_time) / interval_ns

        interpolated[index] = (
            previous_pressure
            + fraction * (next_pressure - previous_pressure)
        )

    logger["luchtdruk_pa"] = interpolated
    logger["knmi_broninterval_uur"] = source_gap_hours
    logger["buiten_knmi_periode"] = is_outside_range

    return logger


def calculate_groundwater_levels(
    combined_data: pd.DataFrame,
    configuration: PeilfilterConfiguration,
) -> pd.DataFrame:
    """
    Bereken waterkolom, waterstand NAP en stijghoogte vanaf bovenkant peilbuis.
    """
    result = combined_data.copy()

    temperature = result["temperatuur_c"].fillna(
        configuration.default_water_temperature_c
    )

    if configuration.use_temperature_density:
        result["waterdichtheid_kg_m3"] = water_density_kg_m3(temperature)
    else:
        result["waterdichtheid_kg_m3"] = DEFAULT_WATER_DENSITY

    result["gebruikte_temperatuur_c"] = temperature

    if configuration.logger_mode == "absolute_pressure":
        result["loggerdruk_pa"] = pressure_to_pa(
            result["loggerwaarde"],
            configuration.logger_pressure_unit,
        )

        result["wateroverdruk_pa"] = (
            result["loggerdruk_pa"] - result["luchtdruk_pa"]
        )

        result["waterkolom_m"] = (
            result["wateroverdruk_pa"]
            / (result["waterdichtheid_kg_m3"] * GRAVITY)
        )

    elif configuration.logger_mode == "water_column":
        result["loggerdruk_pa"] = np.nan
        result["wateroverdruk_pa"] = np.nan
        result["waterkolom_m"] = length_to_m(
            result["loggerwaarde"],
            configuration.water_column_unit,
        )

    else:
        raise ValueError(
            f"Onbekende loggermodus: {configuration.logger_mode}"
        )

    result["filter_id"] = configuration.filter_id
    result["bovenkant_peilbuis_nap_m"] = configuration.top_casing_nap_m
    result["kabellengte_m"] = configuration.cable_length_m
    result["sensorhoogte_nap_m"] = configuration.sensor_elevation_nap_m

    result["waterstand_nap_m"] = (
        result["sensorhoogte_nap_m"] + result["waterkolom_m"]
    )

    # Positieve waarde betekent: water bevindt zich onder de bovenkant.
    result["waterdiepte_tov_bovenkant_m"] = (
        configuration.top_casing_nap_m - result["waterstand_nap_m"]
    )

    result["luchtdruk_hpa"] = result["luchtdruk_pa"] / 100.0

    result["kwaliteitscode"] = build_quality_codes(result)

    return result


def build_quality_codes(dataframe: pd.DataFrame) -> pd.Series:
    """
    Stel per meetregel een compacte kwaliteitscode samen.

    Codes:
        OK      Geldige berekening.
        L-MIS   Loggerwaarde ontbreekt.
        K-MIS   KNMI-luchtdruk ontbreekt.
        K-EXT   Tijdstip ligt buiten KNMI-periode.
        NEG     Negatieve waterkolom.
        RANGE   Onwaarschijnlijke waterstand of waterkolom.
    """
    codes: list[str] = []

    for row in dataframe.itertuples(index=False):
        row_codes: list[str] = []

        if pd.isna(row.loggerwaarde):
            row_codes.append("L-MIS")

        if pd.isna(row.luchtdruk_pa):
            row_codes.append("K-MIS")

        if bool(row.buiten_knmi_periode):
            row_codes.append("K-EXT")

        if pd.notna(row.waterkolom_m) and row.waterkolom_m < 0:
            row_codes.append("NEG")

        implausible = (
            pd.notna(row.waterkolom_m)
            and (
                row.waterkolom_m > 100.0
                or row.waterkolom_m < -5.0
            )
        )

        if implausible:
            row_codes.append("RANGE")

        codes.append("|".join(row_codes) if row_codes else "OK")

    return pd.Series(codes, index=dataframe.index, dtype="string")


def summarize_quality(dataframe: pd.DataFrame) -> DataQualitySummary:
    """Maak een samenvatting van de datakwaliteit."""
    valid_mask = (
        dataframe["waterstand_nap_m"].notna()
        & dataframe["loggerwaarde"].notna()
    )

    implausible_mask = (
        (dataframe["waterkolom_m"] > 100.0)
        | (dataframe["waterkolom_m"] < -5.0)
    ).fillna(False)

    return DataQualitySummary(
        total_rows=len(dataframe),
        valid_rows=int(valid_mask.sum()),
        missing_logger_values=int(
            dataframe["loggerwaarde"].isna().sum()
        ),
        missing_knmi_values=int(
            dataframe["luchtdruk_pa"].isna().sum()
        ),
        negative_water_columns=int(
            (dataframe["waterkolom_m"] < 0).fillna(False).sum()
        ),
        extrapolated_rows=int(
            dataframe["buiten_knmi_periode"].fillna(False).sum()
        ),
        implausible_rows=int(implausible_mask.sum()),
    )


# ============================================================================
# Presentatie en export
# ============================================================================

def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Exporteer een DataFrame als Nederlandstalige CSV."""
    export_data = dataframe.copy()

    datetime_columns = export_data.select_dtypes(
        include=["datetime64[ns]", "datetimetz"]
    ).columns

    for column in datetime_columns:
        export_data[column] = export_data[column].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    csv_text = export_data.to_csv(
        index=False,
        sep=";",
        decimal=",",
        na_rep="",
        lineterminator="\n",
    )

    return csv_text.encode("utf-8-sig")


def create_water_level_figure(dataframe: pd.DataFrame) -> go.Figure:
    """Maak een interactieve figuur van de berekende waterstand."""
    valid = dataframe.dropna(subset=["tijd", "waterstand_nap_m"])

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=valid["tijd"],
            y=valid["waterstand_nap_m"],
            mode="lines",
            name="Waterstand NAP",
            line={"color": "#0078D4", "width": 2},
            hovertemplate=(
                "Tijd: %{x|%d-%m-%Y %H:%M}<br>"
                "Waterstand: %{y:.3f} m NAP"
                "<extra></extra>"
            ),
        )
    )

    figure.add_hline(
        y=float(dataframe["bovenkant_peilbuis_nap_m"].iloc[0]),
        line_dash="dash",
        line_color="#D83B01",
        annotation_text="Bovenkant peilbuis",
        annotation_position="top left",
    )

    figure.update_layout(
        title="Grondwaterstand ten opzichte van NAP",
        xaxis_title="Datum en tijd",
        yaxis_title="Waterstand (m NAP)",
        hovermode="x unified",
        legend_title="Reeks",
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
    )

    return figure


def create_pressure_figure(dataframe: pd.DataFrame) -> go.Figure:
    """Maak een figuur van loggerdruk en geïnterpoleerde luchtdruk."""
    figure = go.Figure()

    if dataframe["loggerdruk_pa"].notna().any():
        figure.add_trace(
            go.Scatter(
                x=dataframe["tijd"],
                y=dataframe["loggerdruk_pa"] / 100.0,
                mode="lines",
                name="Absolute loggerdruk",
                line={"color": "#107C10", "width": 1.5},
            )
        )

    figure.add_trace(
        go.Scatter(
            x=dataframe["tijd"],
            y=dataframe["luchtdruk_hpa"],
            mode="lines",
            name="KNMI-luchtdruk",
            line={"color": "#5C2D91", "width": 1.5},
        )
    )

    figure.update_layout(
        title="Loggerdruk en atmosferische luchtdruk",
        xaxis_title="Datum en tijd",
        yaxis_title="Druk (hPa)",
        hovermode="x unified",
        legend_title="Reeks",
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
    )

    return figure


def show_quality_summary(summary: DataQualitySummary) -> None:
    """Toon de kwaliteitssamenvatting in de Streamlit-interface."""
    columns = st.columns(4)

    columns[0].metric("Meetregels", f"{summary.total_rows:,}")
    columns[1].metric("Geldige resultaten", f"{summary.valid_rows:,}")
    columns[2].metric(
        "Ontbrekende KNMI-waarden",
        f"{summary.missing_knmi_values:,}",
    )
    columns[3].metric(
        "Negatieve waterkolommen",
        f"{summary.negative_water_columns:,}",
    )

    warnings: list[str] = []

    if summary.missing_logger_values:
        warnings.append(
            f"{summary.missing_logger_values} loggerwaarden ontbreken."
        )

    if summary.missing_knmi_values:
        warnings.append(
            f"{summary.missing_knmi_values} tijden konden niet aan geldige "
            "KNMI-luchtdruk worden gekoppeld."
        )

    if summary.extrapolated_rows:
        warnings.append(
            f"{summary.extrapolated_rows} tijden liggen buiten de periode "
            "van het KNMI-bestand."
        )

    if summary.negative_water_columns:
        warnings.append(
            f"{summary.negative_water_columns} berekende waterkolommen zijn "
            "negatief. Controleer de drukeenheden en de luchtdrukreeks."
        )

    if summary.implausible_rows:
        warnings.append(
            f"{summary.implausible_rows} uitkomsten vallen buiten de "
            "standaard plausibiliteitsgrenzen."
        )

    if warnings:
        st.warning("\n\n".join(warnings))
    else:
        st.success("De automatische kwaliteitscontrole heeft geen problemen gevonden.")


# ============================================================================
# Streamlit-interface
# ============================================================================

def configure_page() -> None:
    """Configureer de Streamlit-pagina."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="💧",
        layout="wide",
    )

    st.title("💧 Grondwaterstanden omrekenen naar NAP")

    st.markdown(
        """
        Upload een loggerbestand en een bestand met lokale KNMI-luchtdruk.
        De applicatie compenseert de absolute loggerdruk voor atmosferische
        druk en berekent vervolgens de grondwaterstand in meter NAP.
        """
    )

    with st.expander("Berekeningsmethode en belangrijke aandachtspunten"):
        st.markdown(
            """
            **Berekening**

            1. Sensorhoogte NAP = bovenkant peilbuis NAP − kabellengte
            2. Wateroverdruk = absolute loggerdruk − atmosferische druk
            3. Waterkolom = wateroverdruk ÷ (waterdichtheid × zwaartekracht)
            4. Waterstand NAP = sensorhoogte NAP + waterkolom

            **Belangrijk**

            - Gebruik de kabellengte tot het drukpunt van de logger.
            - Gebruik bij voorkeur KNMI-luchtdruk op stationsniveau.
            - Controleer of de logger absolute druk of al gecompenseerde
              waterkolom registreert.
            - KNMI-luchtdruk kan in 0,1 hPa zijn opgeslagen.
            - Controleer of beide bestanden UTC of Nederlandse lokale tijd
              gebruiken.
            """
        )


def file_reading_controls(
    prefix: str,
) -> tuple[str, str, int]:
    """Maak herbruikbare instellingen voor het inlezen van een bestand."""
    left, middle, right = st.columns(3)

    separator = left.selectbox(
        "Scheidingsteken",
        options=[
            "Automatisch",
            "Puntkomma",
            "Tab",
            "Komma",
            "Pipe",
            "Spaties",
        ],
        key=f"{prefix}_separator",
    )

    decimal = middle.selectbox(
        "Decimaalteken",
        options=[",", "."],
        index=0,
        key=f"{prefix}_decimal",
    )

    automatic_header = right.checkbox(
        "Tabelkop automatisch zoeken",
        value=True,
        key=f"{prefix}_automatic_header",
    )

    if automatic_header:
        skip_rows = -1
    else:
        skip_rows = right.number_input(
            "Regels overslaan",
            min_value=0,
            max_value=500,
            value=0,
            step=1,
            key=f"{prefix}_skip_rows",
        )

    return separator, decimal, int(skip_rows)


def datetime_mapping_controls(
    dataframe: pd.DataFrame,
    prefix: str,
    default_timezone: TimezoneMode,
) -> tuple[str, str | None, str | None, bool, TimezoneMode]:
    """Toon instellingen voor datum-, tijd- en tijdzonekolommen."""
    columns = list(dataframe.columns)

    suggested_date = detect_suggested_column(
        columns,
        DATETIME_COLUMN_HINTS,
    )

    date_column = st.selectbox(
        "Datum- of datum/tijdkolom",
        options=columns,
        index=columns.index(suggested_date),
        key=f"{prefix}_date_column",
    )

    time_options = ["(geen aparte tijdkolom)"] + columns

    time_column = st.selectbox(
        "Optionele aparte tijdkolom",
        options=time_options,
        index=0,
        key=f"{prefix}_time_column",
    )

    datetime_format_label = st.selectbox(
        "Datumformaat",
        options=list(DATETIME_FORMAT_OPTIONS.keys()),
        index=0,
        key=f"{prefix}_datetime_format",
    )

    day_first = st.checkbox(
        "Dag staat vóór maand",
        value=True,
        key=f"{prefix}_day_first",
    )

    timezone_options: list[TimezoneMode] = [
        "Nederlandse lokale tijd",
        "UTC",
        "Geen tijdzonecorrectie",
    ]

    timezone_mode = st.selectbox(
        "Tijdzone van het bestand",
        options=timezone_options,
        index=timezone_options.index(default_timezone),
        key=f"{prefix}_timezone",
    )

    normalized_time_column: str | None = time_column

    if time_column == "(geen aparte tijdkolom)":
        normalized_time_column = None

    return (
        date_column,
        normalized_time_column,
        DATETIME_FORMAT_OPTIONS[datetime_format_label],
        day_first,
        timezone_mode,
    )


def main() -> None:
    """Start de Streamlit-applicatie."""
    configure_page()

    st.header("1. Peilfilter en logger")

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
        help=(
            "De ingemeten hoogte van de bovenkant van de peilbuis "
            "ten opzichte van NAP."
        ),
    )

    cable_length_m = configuration_columns[2].number_input(
        "Kabellengte tot druksensor (m)",
        min_value=0.0,
        value=5.000,
        step=0.001,
        format="%.3f",
        help=(
            "Afstand vanaf de bovenkant van de peilbuis tot het drukpunt "
            "van de logger."
        ),
    )

    logger_mode_label = configuration_columns[3].selectbox(
        "Type loggerwaarde",
        options=list(LOGGER_MODES.keys()),
        index=0,
    )

    logger_mode = LOGGER_MODES[logger_mode_label]

    unit_columns = st.columns(4)

    logger_pressure_unit = unit_columns[0].selectbox(
        "Eenheid loggerdruk",
        options=list(PRESSURE_UNIT_FACTORS_TO_PA.keys()),
        index=list(PRESSURE_UNIT_FACTORS_TO_PA.keys()).index("hPa"),
        disabled=logger_mode != "absolute_pressure",
    )

    water_column_unit = unit_columns[1].selectbox(
        "Eenheid logger-waterkolom",
        options=list(LENGTH_UNIT_FACTORS_TO_M.keys()),
        index=0,
        disabled=logger_mode != "water_column",
    )

    knmi_pressure_unit = unit_columns[2].selectbox(
        "Eenheid KNMI-luchtdruk",
        options=list(PRESSURE_UNIT_FACTORS_TO_PA.keys()),
        index=list(PRESSURE_UNIT_FACTORS_TO_PA.keys()).index("0,1 hPa"),
        disabled=logger_mode != "absolute_pressure",
        help=(
            "Veel KNMI-bestanden slaan luchtdruk op in 0,1 hPa. "
            "Een waarde 10134 betekent dan 1013,4 hPa."
        ),
    )

    default_temperature_c = unit_columns[3].number_input(
        "Standaard watertemperatuur (°C)",
        min_value=0.0,
        max_value=40.0,
        value=12.0,
        step=0.1,
        format="%.1f",
    )

    use_temperature_density = st.checkbox(
        "Corrigeer waterdichtheid op basis van watertemperatuur",
        value=True,
    )

    sensor_elevation = top_casing_nap_m - cable_length_m

    st.info(
        f"De berekende sensorhoogte is **{sensor_elevation:.3f} m NAP**."
    )

    st.header("2. Loggerbestand")

    logger_file = st.file_uploader(
        "Upload het loggerbestand",
        type=["dat", "csv", "txt"],
        key="logger_file",
    )

    if logger_file is None:
        st.info("Upload eerst een loggerbestand om door te gaan.")
        return

    logger_separator, logger_decimal, logger_skip_rows = (
        file_reading_controls("logger")
    )

    try:
        logger_raw = read_tabular_file(
            file_bytes=logger_file.getvalue(),
            separator_option=logger_separator,
            decimal_option=logger_decimal,
            skip_rows=logger_skip_rows,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    st.caption(
        f"Loggerbestand ingelezen: {len(logger_raw):,} regels en "
        f"{len(logger_raw.columns)} kolommen."
    )

    with st.expander("Voorbeeld loggerbestand", expanded=True):
        st.dataframe(logger_raw.head(20), use_container_width=True)

    st.subheader("Kolomkoppeling loggerbestand")

    (
        logger_date_column,
        logger_time_column,
        logger_datetime_format,
        logger_day_first,
        logger_timezone,
    ) = datetime_mapping_controls(
        dataframe=logger_raw,
        prefix="logger",
        default_timezone="Nederlandse lokale tijd",
    )

    logger_columns = list(logger_raw.columns)

    suggested_logger_value = detect_suggested_column(
        logger_columns,
        PRESSURE_COLUMN_HINTS,
    )

    logger_value_column = st.selectbox(
        "Kolom met loggerdruk of waterkolom",
        options=logger_columns,
        index=logger_columns.index(suggested_logger_value),
    )

    temperature_options = ["(geen temperatuurkolom)"] + logger_columns

    suggested_temperature = detect_suggested_column(
        logger_columns,
        TEMPERATURE_COLUMN_HINTS,
    )

    temperature_index = (
        temperature_options.index(suggested_temperature)
        if suggested_temperature in temperature_options
        else 0
    )

    temperature_column_selection = st.selectbox(
        "Optionele watertemperatuurkolom",
        options=temperature_options,
        index=temperature_index,
    )

    temperature_column: str | None = temperature_column_selection

    if temperature_column_selection == "(geen temperatuurkolom)":
        temperature_column = None

    if logger_mode == "absolute_pressure":
        st.header("3. KNMI-luchtdrukbestand")

        knmi_file = st.file_uploader(
            "Upload het lokale KNMI-bestand",
            type=["dat", "csv", "txt"],
            key="knmi_file",
        )

        if knmi_file is None:
            st.info(
                "Upload een KNMI-bestand met datum, tijd en luchtdruk."
            )
            return

        knmi_separator, knmi_decimal, knmi_skip_rows = (
            file_reading_controls("knmi")
        )

        try:
            knmi_raw = read_tabular_file(
                file_bytes=knmi_file.getvalue(),
                separator_option=knmi_separator,
                decimal_option=knmi_decimal,
                skip_rows=knmi_skip_rows,
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        st.caption(
            f"KNMI-bestand ingelezen: {len(knmi_raw):,} regels en "
            f"{len(knmi_raw.columns)} kolommen."
        )

        with st.expander("Voorbeeld KNMI-bestand", expanded=True):
            st.dataframe(knmi_raw.head(20), use_container_width=True)

        st.subheader("Kolomkoppeling KNMI-bestand")

        (
            knmi_date_column,
            knmi_time_column,
            knmi_datetime_format,
            knmi_day_first,
            knmi_timezone,
        ) = datetime_mapping_controls(
            dataframe=knmi_raw,
            prefix="knmi",
            default_timezone="UTC",
        )

        knmi_columns = list(knmi_raw.columns)

        suggested_knmi_pressure = detect_suggested_column(
            knmi_columns,
            PRESSURE_COLUMN_HINTS,
        )

        knmi_pressure_column = st.selectbox(
            "Kolom met KNMI-luchtdruk",
            options=knmi_columns,
            index=knmi_columns.index(suggested_knmi_pressure),
        )

        maximum_gap_hours = st.number_input(
            "Maximaal KNMI-datagat voor interpolatie (uur)",
            min_value=1.0,
            max_value=168.0,
            value=6.0,
            step=1.0,
            help=(
                "Loggerwaarden worden niet berekend als de omliggende "
                "KNMI-metingen verder uit elkaar liggen dan deze grens."
            ),
        )

    else:
        knmi_raw = pd.DataFrame()
        knmi_date_column = ""
        knmi_time_column = None
        knmi_datetime_format = None
        knmi_day_first = True
        knmi_timezone = "Geen tijdzonecorrectie"
        knmi_pressure_column = ""
        maximum_gap_hours = 6.0

    st.header("4. Berekening")

    if not st.button(
        "Bereken waterstanden",
        type="primary",
        use_container_width=True,
    ):
        return

    configuration = PeilfilterConfiguration(
        filter_id=filter_id.strip() or "Onbekend",
        top_casing_nap_m=float(top_casing_nap_m),
        cable_length_m=float(cable_length_m),
        logger_mode=logger_mode,
        logger_pressure_unit=logger_pressure_unit,
        knmi_pressure_unit=knmi_pressure_unit,
        water_column_unit=water_column_unit,
        default_water_temperature_c=float(default_temperature_c),
        use_temperature_density=use_temperature_density,
    )

    try:
        with st.spinner("Gegevens verwerken en waterstanden berekenen..."):
            logger_data = prepare_logger_data(
                dataframe=logger_raw,
                date_column=logger_date_column,
                time_column=logger_time_column,
                value_column=logger_value_column,
                temperature_column=temperature_column,
                datetime_format=logger_datetime_format,
                day_first=logger_day_first,
                timezone_mode=logger_timezone,
            )

            if logger_mode == "absolute_pressure":
                knmi_data = prepare_knmi_data(
                    dataframe=knmi_raw,
                    date_column=knmi_date_column,
                    time_column=knmi_time_column,
                    pressure_column=knmi_pressure_column,
                    datetime_format=knmi_datetime_format,
                    day_first=knmi_day_first,
                    timezone_mode=knmi_timezone,
                )

                combined_data = interpolate_knmi_pressure(
                    logger_data=logger_data,
                    knmi_data=knmi_data,
                    knmi_pressure_unit=knmi_pressure_unit,
                    maximum_gap_hours=float(maximum_gap_hours),
                )
            else:
                combined_data = logger_data.copy()
                combined_data["luchtdruk_pa"] = np.nan
                combined_data["knmi_broninterval_uur"] = np.nan
                combined_data["buiten_knmi_periode"] = False

            result = calculate_groundwater_levels(
                combined_data=combined_data,
                configuration=configuration,
            )

    except ValueError as exc:
        LOGGER.exception("Validatiefout tijdens verwerking")
        st.error(str(exc))
        return
    except Exception as exc:
        LOGGER.exception("Onverwachte fout tijdens verwerking")
        st.error(
            "Er is een onverwachte fout opgetreden bij de berekening: "
            f"{exc}"
        )
        return

    st.success("De waterstanden zijn berekend.")

    summary = summarize_quality(result)
    show_quality_summary(summary)

    valid_results = result.dropna(subset=["waterstand_nap_m"])

    if not valid_results.empty:
        metric_columns = st.columns(4)

        metric_columns[0].metric(
            "Minimum",
            f"{valid_results['waterstand_nap_m'].min():.3f} m NAP",
        )
        metric_columns[1].metric(
            "Gemiddelde",
            f"{valid_results['waterstand_nap_m'].mean():.3f} m NAP",
        )
        metric_columns[2].metric(
            "Maximum",
            f"{valid_results['waterstand_nap_m'].max():.3f} m NAP",
        )
        metric_columns[3].metric(
            "Meetperiode",
            (
                f"{valid_results['tijd'].min():%d-%m-%Y} t/m "
                f"{valid_results['tijd'].max():%d-%m-%Y}"
            ),
        )

        st.plotly_chart(
            create_water_level_figure(result),
            use_container_width=True,
        )

        if logger_mode == "absolute_pressure":
            st.plotly_chart(
                create_pressure_figure(result),
                use_container_width=True,
            )
    else:
        st.error(
            "Er zijn geen geldige waterstanden berekend. Controleer de "
            "eenheden, tijdzones, KNMI-periode en kolomkoppelingen."
        )

    st.subheader("Resultaten")

    preferred_columns = [
        "filter_id",
        "tijd",
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
        "kwaliteitscode",
    ]

    display_columns = [
        column
        for column in preferred_columns
        if column in result.columns
    ]

    st.dataframe(
        result[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    safe_filter_id = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        configuration.filter_id,
    ).strip("_")

    if not safe_filter_id:
        safe_filter_id = "peilfilter"

    st.download_button(
        label="Download resultaten als CSV",
        data=dataframe_to_csv_bytes(result[display_columns]),
        file_name=f"{safe_filter_id}_waterstanden_nap.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

    with st.expander("Betekenis kwaliteitscodes"):
        st.markdown(
            """
            - **OK**: geldige berekening zonder automatische waarschuwing
            - **L-MIS**: loggerwaarde ontbreekt
            - **K-MIS**: geen bruikbare KNMI-luchtdruk beschikbaar
            - **K-EXT**: logger-tijdstip ligt buiten de KNMI-periode
            - **NEG**: berekende waterkolom is negatief
            - **RANGE**: waterkolom ligt buiten de standaard plausibiliteitsgrens
            """
        )


if __name__ == "__main__":
    main()
