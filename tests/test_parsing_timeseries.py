"""Tests voor tekstparsing en tijdzones."""
import pandas as pd
import pytest
from waterstanden_app.parsing import clean_numeric_series, detect_file_kind, read_text_table
from waterstanden_app.timeseries import localize_datetime_series

def test_numeric_locales_and_strict_invalid_text() -> None:
    """Ondersteun lokale getallen zonder willekeurige tekst te accepteren."""
    result = clean_numeric_series(pd.Series(["1.234,5", "1,234.5", "druk=12"]))
    assert result.iloc[0] == pytest.approx(1234.5)
    assert result.iloc[1] == pytest.approx(1234.5)
    assert pd.isna(result.iloc[2])

def test_utc_conversion() -> None:
    """Converteer UTC naar Nederlandse lokale wintertijd."""
    result = localize_datetime_series(pd.Series(["2026-01-01 12:00:00"]), "UTC")
    assert result.iloc[0] == pd.Timestamp("2026-01-01 13:00:00")

def test_empty_file_detection() -> None:
    """Weiger lege uploads."""
    with pytest.raises(ValueError, match="leeg"):
        detect_file_kind(b"")

def test_read_text_table() -> None:
    """Lees een eenvoudige puntkommatabel."""
    result = read_text_table(b"tijd;waarde\n2026-01-01;1,2\n", "utf-8", "Puntkomma", ",", 0)
    assert list(result.columns) == ["tijd", "waarde"]
