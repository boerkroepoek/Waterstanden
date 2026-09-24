"""Tests voor hydrologische berekeningen."""
import numpy as np
import pandas as pd
import pytest
from waterstanden_app.calculations import calculate_water_levels, interpolate_knmi_pressure
from waterstanden_app.models import PeilfilterConfiguration

def test_exact_first_knmi_measurement_regression() -> None:
    """Verwerk een exacte match met de eerste KNMI-meting."""
    logger = pd.DataFrame({"tijd": pd.to_datetime(["2026-01-01 00:00"]), "loggerwaarde": [1100.0], "temperatuur_c": [12.0]})
    knmi = pd.DataFrame({"tijd": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]), "knmi_druk_bronwaarde": [1000.0, 1001.0]})
    result = interpolate_knmi_pressure(logger, knmi, "hPa", 6.0)
    assert result.loc[0, "luchtdruk_pa"] == pytest.approx(100_000.0)
    assert result.loc[0, "knmi_broninterval_uur"] == 0.0

def test_interpolation_and_large_gap() -> None:
    """Interpoleer binnen en weiger boven de maximale datagat."""
    logger = pd.DataFrame({"tijd": pd.to_datetime(["2026-01-01 00:30"])})
    knmi = pd.DataFrame({"tijd": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]), "knmi_druk_bronwaarde": [1000.0, 1002.0]})
    assert interpolate_knmi_pressure(logger, knmi, "hPa", 6.0).loc[0, "luchtdruk_pa"] == pytest.approx(100_100.0)
    assert np.isnan(interpolate_knmi_pressure(logger, knmi, "hPa", 0.25).loc[0, "luchtdruk_pa"])

def test_water_column_level_and_quality_codes() -> None:
    """Bereken waterstand en behoud meerdere kwaliteitscodes."""
    data = pd.DataFrame({"tijd": pd.to_datetime(["2026-01-01"]), "loggerwaarde": [-600.0], "temperatuur_c": [50.0], "luchtdruk_pa": [np.nan]})
    config = PeilfilterConfiguration("PB01", 1.0, 5.0, "water_column", "cm", "hPa", 12.0, True)
    result = calculate_water_levels(data, config)
    assert result.loc[0, "waterstand_nap_m"] == pytest.approx(-10.0)
    assert "NEGATIEVE_WATERKOLOM" in result.loc[0, "kwaliteitscode"]
    assert "BUITEN_BEREIK" in result.loc[0, "kwaliteitscode"]
    assert "TEMPERATUUR_BUITEN_BEREIK" in result.loc[0, "kwaliteitscode"]

def test_implausible_pressure_unit_is_rejected() -> None:
    """Stop wanneer de KNMI-eenheid een onrealistische druk oplevert."""
    logger = pd.DataFrame({"tijd": pd.to_datetime(["2026-01-01 00:30"])})
    knmi = pd.DataFrame({"tijd": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]), "knmi_druk_bronwaarde": [1013.0, 1014.0]})
    with pytest.raises(ValueError, match="KNMI-eenheid"):
        interpolate_knmi_pressure(logger, knmi, "0,1 hPa", 6.0)
