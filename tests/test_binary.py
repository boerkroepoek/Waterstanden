"""Tests voor de binaire decoder."""
import struct
from datetime import datetime
import pytest
from waterstanden_app.binary import decode_binary_logger, validate_binary_configuration
from waterstanden_app.models import BinaryDecoderConfiguration, BinaryFieldDefinition

def configuration(timestamp_offset: int = 0, timestamp_type: str = "Unsigned integer 32-bit") -> BinaryDecoderConfiguration:
    """Maak een standaard testconfiguratie."""
    return BinaryDecoderConfiguration(0, 12, "Little-endian", "Unix-tijd", timestamp_offset, timestamp_type, "seconden", datetime(1970, 1, 1), BinaryFieldDefinition(4, "Float 32-bit"), BinaryFieldDefinition(8, "Float 32-bit"))

def test_decode_binary_record() -> None:
    """Decodeer Unix-tijd, druk en temperatuur."""
    raw = struct.pack("<Iff", 1_735_689_600, 1013.25, 12.5)
    result = decode_binary_logger(raw, configuration())
    assert result.loc[0, "loggerwaarde"] == pytest.approx(1013.25)
    assert result.loc[0, "temperatuur_c"] == pytest.approx(12.5)

def test_timestamp_field_must_fit_record() -> None:
    """Valideer ook het tijdstempelveld."""
    with pytest.raises(ValueError, match="tijdstempelveld"):
        validate_binary_configuration(bytes(24), configuration(8, "Unsigned integer 64-bit"))

def test_empty_file_is_rejected() -> None:
    """Weiger een leeg binair bestand."""
    with pytest.raises(ValueError, match="leeg"):
        validate_binary_configuration(b"", configuration())
