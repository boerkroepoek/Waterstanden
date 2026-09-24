"""Hydrologische conversies en berekeningen."""
import numpy as np
import pandas as pd
from .constants import DEFAULT_WATER_DENSITY_KG_M3, GRAVITY_M_S2, LENGTH_UNIT_FACTORS_TO_M, NANOSECONDS_PER_HOUR, PRESSURE_UNIT_FACTORS_TO_PA
from .models import PeilfilterConfiguration

def water_density_kg_m3(temperature_c: pd.Series) -> pd.Series:
    """Bereken de dichtheid van zoet water tussen 0 en 40 graden Celsius."""
    temperature = temperature_c.clip(0.0, 40.0)
    numerator = (temperature + 288.9414) * (temperature - 3.9863) ** 2
    denominator = 508_929.2 * (temperature + 68.12963)
    return 1_000.0 * (1.0 - numerator / denominator)

def pressure_to_pa(values: pd.Series, unit: str) -> pd.Series:
    """Converteer druk naar pascal."""
    if unit not in PRESSURE_UNIT_FACTORS_TO_PA:
        raise ValueError(f"Onbekende drukeenheid: {unit}")
    return values * PRESSURE_UNIT_FACTORS_TO_PA[unit]

def length_to_m(values: pd.Series, unit: str) -> pd.Series:
    """Converteer lengte naar meter."""
    if unit not in LENGTH_UNIT_FACTORS_TO_M:
        raise ValueError(f"Onbekende lengte-eenheid: {unit}")
    return values * LENGTH_UNIT_FACTORS_TO_M[unit]

def interpolate_knmi_pressure(logger_data: pd.DataFrame, knmi_data: pd.DataFrame, knmi_pressure_unit: str, maximum_gap_hours: float) -> pd.DataFrame:
    """Interpoleer KNMI-druk zonder extrapolatie."""
    if logger_data.empty:
        raise ValueError("De loggergegevens zijn leeg.")
    if len(knmi_data) < 2:
        raise ValueError("Minimaal twee KNMI-metingen zijn nodig.")
    if maximum_gap_hours <= 0:
        raise ValueError("De maximale KNMI-datagat moet positief zijn.")
    logger = logger_data.sort_values("tijd").reset_index(drop=True).copy()
    knmi = knmi_data.sort_values("tijd").reset_index(drop=True).copy()
    knmi["luchtdruk_pa"] = pressure_to_pa(knmi["knmi_druk_bronwaarde"], knmi_pressure_unit)
    pressure_hpa = knmi["luchtdruk_pa"] / 100.0
    if not pressure_hpa.between(850.0, 1100.0).all():
        raise ValueError("De omgerekende luchtdruk valt buiten 850 tot 1100 hPa. Controleer de KNMI-eenheid.")
    logger_times = logger["tijd"].astype("int64").to_numpy()
    knmi_times = knmi["tijd"].astype("int64").to_numpy()
    pressures = knmi["luchtdruk_pa"].to_numpy(float)
    interpolated = np.full(len(logger), np.nan); intervals = np.full(len(logger), np.nan); outside = np.zeros(len(logger), dtype=bool)
    for index, position in enumerate(np.searchsorted(knmi_times, logger_times, side="left")):
        sample = logger_times[index]
        if position < len(knmi_times) and sample == knmi_times[position]:
            interpolated[index] = pressures[position]; intervals[index] = 0.0; continue
        if position == 0 or position == len(knmi_times):
            outside[index] = True; continue
        previous_time, next_time = knmi_times[position - 1], knmi_times[position]
        interval_ns = next_time - previous_time
        if interval_ns <= 0:
            continue
        intervals[index] = interval_ns / NANOSECONDS_PER_HOUR
        if intervals[index] > maximum_gap_hours:
            continue
        fraction = (sample - previous_time) / interval_ns
        interpolated[index] = pressures[position - 1] + fraction * (pressures[position] - pressures[position - 1])
    logger["luchtdruk_pa"] = interpolated; logger["knmi_broninterval_uur"] = intervals; logger["buiten_knmi_periode"] = outside
    return logger

def calculate_water_levels(dataframe: pd.DataFrame, configuration: PeilfilterConfiguration) -> pd.DataFrame:
    """Bereken waterkolom, NAP-waterstand en kwaliteitsvlaggen."""
    result = dataframe.copy()
    temperature = pd.to_numeric(result["temperatuur_c"], errors="coerce").fillna(configuration.default_temperature_c)
    result["gebruikte_temperatuur_c"] = temperature
    result["qc_temperatuur_buiten_bereik"] = ~temperature.between(0.0, 40.0)
    result["waterdichtheid_kg_m3"] = water_density_kg_m3(temperature) if configuration.use_temperature_density else DEFAULT_WATER_DENSITY_KG_M3
    if configuration.logger_mode == "absolute_pressure":
        result["loggerdruk_pa"] = pressure_to_pa(result["loggerwaarde"], configuration.logger_unit)
        result["wateroverdruk_pa"] = result["loggerdruk_pa"] - result["luchtdruk_pa"]
        result["waterkolom_m"] = result["wateroverdruk_pa"] / (result["waterdichtheid_kg_m3"] * GRAVITY_M_S2)
    elif configuration.logger_mode == "water_column":
        result["loggerdruk_pa"] = np.nan; result["wateroverdruk_pa"] = np.nan
        result["waterkolom_m"] = length_to_m(result["loggerwaarde"], configuration.logger_unit)
    else:
        raise ValueError("Onbekende loggermodus.")
    result["filter_id"] = configuration.filter_id; result["bovenkant_peilbuis_nap_m"] = configuration.top_casing_nap_m
    result["kabellengte_m"] = configuration.cable_length_m; result["sensorhoogte_nap_m"] = configuration.sensor_elevation_nap_m
    result["waterstand_nap_m"] = result["sensorhoogte_nap_m"] + result["waterkolom_m"]
    result["waterdiepte_tov_bovenkant_m"] = configuration.top_casing_nap_m - result["waterstand_nap_m"]
    result["luchtdruk_hpa"] = result["luchtdruk_pa"] / 100.0
    result["qc_logger_ontbreekt"] = result["loggerwaarde"].isna()
    result["qc_luchtdruk_ontbreekt"] = result["luchtdruk_pa"].isna() & (configuration.logger_mode == "absolute_pressure")
    result["qc_negatieve_waterkolom"] = result["waterkolom_m"] < 0
    result["qc_buiten_bereik"] = (result["waterkolom_m"] > 100) | (result["waterkolom_m"] < -5)
    labels = [("qc_logger_ontbreekt", "LOGGER_ONTBREEKT"), ("qc_luchtdruk_ontbreekt", "LUCHTDRUK_ONTBREEKT"), ("qc_negatieve_waterkolom", "NEGATIEVE_WATERKOLOM"), ("qc_buiten_bereik", "BUITEN_BEREIK"), ("qc_temperatuur_buiten_bereik", "TEMPERATUUR_BUITEN_BEREIK")]
    result["kwaliteitscode"] = ["|".join(label for column, label in labels if bool(row[column])) or "OK" for _, row in result.iterrows()]
    return result
