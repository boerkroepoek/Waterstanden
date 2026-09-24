"""Normalisatie van logger- en KNMI-tijdreeksen."""
import numpy as np
import pandas as pd
from .models import TimezoneMode
from .parsing import clean_numeric_series

def localize_datetime_series(series: pd.Series, timezone_mode: TimezoneMode) -> pd.Series:
    """Normaliseer naar tijdzonevrije Nederlandse lokale tijd."""
    if timezone_mode == "UTC":
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        return parsed.dt.tz_convert("Europe/Amsterdam").dt.tz_localize(None)
    parsed = pd.to_datetime(series, errors="coerce")
    aware = isinstance(parsed.dtype, pd.DatetimeTZDtype)
    if timezone_mode == "Geen tijdzonecorrectie":
        return parsed.dt.tz_localize(None) if aware else parsed
    if aware:
        return parsed.dt.tz_convert("Europe/Amsterdam").dt.tz_localize(None)
    return parsed.dt.tz_localize("Europe/Amsterdam", ambiguous="NaT", nonexistent="NaT").dt.tz_localize(None)

def _datetimes(dataframe: pd.DataFrame, date_column: str, time_column: str | None, day_first: bool) -> pd.Series:
    values = dataframe[date_column].astype("string").str.strip()
    if time_column:
        values = values + " " + dataframe[time_column].astype("string").str.strip()
    return pd.to_datetime(values, errors="coerce", dayfirst=day_first, format="mixed")

def prepare_text_logger(dataframe: pd.DataFrame, date_column: str, time_column: str | None, value_column: str, temperature_column: str | None, timezone_mode: TimezoneMode, day_first: bool) -> pd.DataFrame:
    """Zet tekstuele loggerdata om naar de standaardstructuur."""
    result = pd.DataFrame({"tijd": _datetimes(dataframe, date_column, time_column, day_first), "loggerwaarde": clean_numeric_series(dataframe[value_column])})
    result["temperatuur_c"] = clean_numeric_series(dataframe[temperature_column]) if temperature_column else np.nan
    result["tijd"] = localize_datetime_series(result["tijd"], timezone_mode)
    result = result.dropna(subset=["tijd"]).sort_values("tijd").drop_duplicates("tijd", keep="last").reset_index(drop=True)
    if result.empty:
        raise ValueError("Geen geldige logger-tijdstippen gevonden.")
    return result

def prepare_binary_logger(dataframe: pd.DataFrame, timezone_mode: TimezoneMode) -> pd.DataFrame:
    """Normaliseer gedecodeerde binaire loggerdata."""
    result = dataframe.copy(); result["tijd"] = localize_datetime_series(result["tijd"], timezone_mode)
    result["loggerwaarde"] = pd.to_numeric(result["loggerwaarde"], errors="coerce")
    result["temperatuur_c"] = pd.to_numeric(result["temperatuur_c"], errors="coerce")
    result = result.dropna(subset=["tijd"]).sort_values("tijd").drop_duplicates("tijd", keep="last").reset_index(drop=True)
    if result.empty:
        raise ValueError("Geen geldige binaire loggerrecords gevonden.")
    return result

def prepare_knmi_data(dataframe: pd.DataFrame, date_column: str, time_column: str | None, pressure_column: str, timezone_mode: TimezoneMode, day_first: bool) -> pd.DataFrame:
    """Normaliseer en aggregeer KNMI-data."""
    result = pd.DataFrame({"tijd": _datetimes(dataframe, date_column, time_column, day_first), "knmi_druk_bronwaarde": clean_numeric_series(dataframe[pressure_column])})
    result["tijd"] = localize_datetime_series(result["tijd"], timezone_mode)
    result = result.dropna().groupby("tijd", as_index=False).agg({"knmi_druk_bronwaarde": "mean"}).sort_values("tijd").reset_index(drop=True)
    if len(result) < 2:
        raise ValueError("Minimaal twee geldige KNMI-metingen zijn noodzakelijk.")
    return result
