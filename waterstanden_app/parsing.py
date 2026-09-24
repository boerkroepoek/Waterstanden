"""Detectie en verwerking van tekstbestanden."""
import io
import re
import numpy as np
import pandas as pd
from .constants import TEXT_ENCODINGS

def clean_numeric_series(series: pd.Series) -> pd.Series:
    """Converteer lokale getalnotaties en ontbrekende waarden."""
    text = series.astype("string").str.strip()
    text = text.mask(text.str.lower().isin({"", "na", "n/a", "nan", "none", "null", "-999", "-9999", "-999.9"}))
    text = text.str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
    both = text.str.contains(",", na=False) & text.str.contains(r"\.", na=False)
    comma_decimal = both & (text.str.rfind(",") > text.str.rfind("."))
    text = text.where(~comma_decimal, text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    text = text.where(~(both & ~comma_decimal), text.str.replace(",", "", regex=False))
    only_comma = text.str.contains(",", na=False) & ~text.str.contains(r"\.", na=False)
    text = text.where(~only_comma, text.str.replace(",", ".", regex=False))
    number = text.str.extract(r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$", expand=False)
    return pd.to_numeric(number, errors="coerce")

def normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normaliseer kolomnamen en maak dubbele namen uniek."""
    result = dataframe.copy(); seen: dict[str, int] = {}; names = []
    for column in result.columns:
        base = re.sub(r"\s+", " ", str(column).strip()) or "kolom"
        seen[base] = seen.get(base, 0) + 1
        names.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    result.columns = names
    return result

def detect_file_kind(file_bytes: bytes) -> tuple[str, str | None, float]:
    """Classificeer een bestand heuristisch als tekst of binair."""
    if not file_bytes:
        raise ValueError("Het geüploade bestand is leeg.")
    sample = file_bytes[:65_536]
    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "text", "utf-16", 1.0
    if sample.startswith(b"\xef\xbb\xbf"):
        return "text", "utf-8-sig", 1.0
    if b"\x00" in sample:
        pairs = max(1, len(sample) // 2)
        if sample[0::2].count(0) / pairs > 0.25:
            return "text", "utf-16-be", 0.9
        if sample[1::2].count(0) / pairs > 0.25:
            return "text", "utf-16-le", 0.9
    for encoding in TEXT_ENCODINGS:
        try:
            decoded = sample.decode(encoding)
        except UnicodeDecodeError:
            continue
        if decoded:
            ratio = sum(c.isprintable() or c in "\r\n\t" for c in decoded) / len(decoded)
            separators = sum(decoded.count(s) for s in (";", ",", "\t", "|", "\n"))
            if ratio >= 0.90 and separators >= 2:
                return "text", encoding, ratio
    return "binary", None, 0.8

def decode_text_file(file_bytes: bytes, preferred_encoding: str | None = None) -> str:
    """Decodeer tekst met bekende encodings."""
    encodings = ([preferred_encoding] if preferred_encoding else []) + [e for e in TEXT_ENCODINGS if e != preferred_encoding]
    for encoding in encodings:
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Het bestand kan niet als tekst worden gelezen.")

def detect_separator(text: str) -> str:
    """Detecteer het meest consistente scheidingsteken."""
    lines = [line for line in text.splitlines() if line.strip()][:30]
    if not lines:
        raise ValueError("Geen gegevensregels gevonden.")
    scores = {}
    for separator in (";", "\t", ",", "|"):
        counts = [line.count(separator) for line in lines]; nonzero = [n for n in counts if n]
        scores[separator] = -1.0 if not nonzero else len(nonzero) / len(lines) * 10 + float(np.mean(nonzero)) - float(np.std(nonzero))
    selected = max(scores, key=scores.get)
    return r"\s+" if scores[selected] < 0 else selected

def read_text_table(file_bytes: bytes, encoding: str | None, separator_name: str, decimal: str, header_row: int) -> pd.DataFrame:
    """Lees een teksttabel streng in."""
    text = decode_text_file(file_bytes, encoding)
    mapping = {"Automatisch": detect_separator(text), "Puntkomma": ";", "Komma": ",", "Tab": "\t", "Pipe": "|", "Spaties": r"\s+"}
    try:
        frame = pd.read_csv(io.StringIO(text), sep=mapping[separator_name], decimal=decimal, header=header_row, engine="python", dtype=str, on_bad_lines="error")
    except Exception as exc:
        raise ValueError("Het tekstbestand kon niet als tabel worden gelezen.") from exc
    frame = normalize_columns(frame).dropna(how="all").dropna(axis=1, how="all")
    if frame.empty:
        raise ValueError("Het bestand bevat geen bruikbare tabelgegevens.")
    return frame
