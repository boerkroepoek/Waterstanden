"""Streamlit-interface voor conversie van loggerdata naar NAP-waterstanden."""
import logging
import re
from datetime import datetime
import pandas as pd
import streamlit as st
from waterstanden_app.binary import create_hex_preview, decode_binary_logger
from waterstanden_app.calculations import calculate_water_levels, interpolate_knmi_pressure
from waterstanden_app.constants import APP_TITLE, BINARY_VALUE_FORMATS, LENGTH_UNIT_FACTORS_TO_M, LOGGER_MODES, MAX_FILE_SIZE_MB, PRESSURE_UNIT_FACTORS_TO_PA
from waterstanden_app.export import create_water_level_figure, dataframe_to_csv_bytes
from waterstanden_app.models import BinaryDecoderConfiguration, BinaryFieldDefinition, PeilfilterConfiguration, TimezoneMode
from waterstanden_app.parsing import detect_file_kind, read_text_table
from waterstanden_app.timeseries import prepare_binary_logger, prepare_knmi_data, prepare_text_logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger(__name__)

@st.cache_data(show_spinner="Binair loggerbestand decoderen...")
def cached_decode(file_bytes: bytes, configuration: BinaryDecoderConfiguration) -> pd.DataFrame:
    """Decodeer en cache een binair loggerbestand."""
    return decode_binary_logger(file_bytes, configuration)

def table_settings(prefix: str) -> tuple[str, str, int]:
    """Render instellingen voor een teksttabel."""
    columns = st.columns(3)
    separator = columns[0].selectbox("Scheidingsteken", ["Automatisch", "Puntkomma", "Komma", "Tab", "Pipe", "Spaties"], key=f"{prefix}_separator")
    decimal = columns[1].selectbox("Decimaalteken", [",", "."], key=f"{prefix}_decimal")
    header = int(columns[2].number_input("Index tabelkop, beginnend bij 0", min_value=0, value=0, step=1, key=f"{prefix}_header"))
    return separator, decimal, header

def timezone_selector(key: str, default: TimezoneMode) -> TimezoneMode:
    """Render een tijdzoneselectie."""
    options: list[TimezoneMode] = ["Nederlandse lokale tijd", "UTC", "Geen tijdzonecorrectie"]
    return st.selectbox("Tijdzone in het bestand", options, index=options.index(default), key=key)

def binary_field(label: str, key: str, offset: int, data_type: str) -> BinaryFieldDefinition:
    """Render één configureerbaar binair veld."""
    st.markdown(f"**{label}**"); columns = st.columns(4)
    return BinaryFieldDefinition(
        offset=int(columns[0].number_input("Byte-offset", min_value=0, value=offset, key=f"{key}_offset")),
        data_type=columns[1].selectbox("Datatype", list(BINARY_VALUE_FORMATS), index=list(BINARY_VALUE_FORMATS).index(data_type), key=f"{key}_type"),
        scale=float(columns[2].number_input("Schaalfactor", value=1.0, format="%.10f", key=f"{key}_scale")),
        value_offset=float(columns[3].number_input("Waarde-offset", value=0.0, format="%.10f", key=f"{key}_value_offset")),
    )

def binary_configuration(file_bytes: bytes) -> BinaryDecoderConfiguration:
    """Render de binaire decoderconfiguratie."""
    st.warning("Gebruik de technische documentatie van de logger. Verkeerde instellingen kunnen plausibele maar onjuiste waarden opleveren.")
    with st.expander("Hex-preview", expanded=True):
        columns = st.columns(2)
        start = int(columns[0].number_input("Startpositie", 0, max(0, len(file_bytes) - 1), 0, 16))
        length = int(columns[1].number_input("Aantal bytes", 16, min(8192, len(file_bytes)), min(512, len(file_bytes)), 16))
        st.code(create_hex_preview(file_bytes, start, length), language="text")
    columns = st.columns(3)
    header = int(columns[0].number_input("Headerlengte", 0, max(0, len(file_bytes) - 1), 0))
    record = int(columns[1].number_input("Recordlengte", 1, 65536, 16))
    byte_order = columns[2].selectbox("Bytevolgorde", ["Little-endian", "Big-endian"])
    mode = st.selectbox("Tijdstempelopslag", ["Unix-tijd", "Excel-datum", "Tijd sinds aangepaste oorsprong", "Losse datumvelden"])
    timestamp_offset, timestamp_type, timestamp_unit = 0, "Unsigned integer 32-bit", "seconden"
    origin = datetime(1970, 1, 1); year_offset, month_offset, day_offset, hour_offset, minute_offset, second_offset = 0, 2, 3, 4, 5, 6
    year_type = "Unsigned integer 16-bit"
    if mode == "Losse datumvelden":
        c = st.columns(4); year_offset = int(c[0].number_input("Offset jaar", 0, value=0)); year_type = c[1].selectbox("Datatype jaar", ["Unsigned integer 16-bit", "Signed integer 16-bit", "Unsigned integer 32-bit"]); month_offset = int(c[2].number_input("Offset maand", 0, value=2)); day_offset = int(c[3].number_input("Offset dag", 0, value=3))
        c = st.columns(3); hour_offset = int(c[0].number_input("Offset uur", 0, value=4)); minute_offset = int(c[1].number_input("Offset minuut", 0, value=5)); second_offset = int(c[2].number_input("Offset seconde", 0, value=6))
    else:
        c = st.columns(4); timestamp_offset = int(c[0].number_input("Offset tijdstempel", 0, value=0)); timestamp_type = c[1].selectbox("Datatype tijdstempel", list(BINARY_VALUE_FORMATS), index=list(BINARY_VALUE_FORMATS).index("Unsigned integer 32-bit"))
        units = ["seconden", "milliseconden", "microseconden"] + (["dagen"] if mode == "Tijd sinds aangepaste oorsprong" else [])
        timestamp_unit = "dagen" if mode == "Excel-datum" else c[2].selectbox("Tijdseenheid", units)
        if mode == "Tijd sinds aangepaste oorsprong":
            origin = datetime.combine(c[3].date_input("Oorsprongsdatum", datetime(1970, 1, 1)), datetime.min.time())
    pressure = binary_field("Loggerdruk of waterkolom", "pressure", 4, "Float 32-bit")
    temperature = binary_field("Temperatuur", "temperature", 8, "Float 32-bit") if st.checkbox("Record bevat temperatuur", True) else None
    return BinaryDecoderConfiguration(header, record, byte_order, mode, timestamp_offset, timestamp_type, timestamp_unit, origin, pressure, temperature, year_offset, month_offset, day_offset, hour_offset, minute_offset, second_offset, year_type)

def main() -> None:
    """Start de Streamlit-app."""
    st.set_page_config(page_title=APP_TITLE, page_icon="💧", layout="wide")
    st.title("💧 Waterstanden omrekenen naar NAP")
    st.caption("Tekstuele en binaire loggerdata, luchtdrukcompensatie en export naar Nederlandse CSV.")
    st.header("1. Meetopstelling")
    c = st.columns(4); filter_id = c[0].text_input("Peilfilter-ID", "PB01"); top = float(c[1].number_input("Bovenkant peilbuis (m NAP)", value=1.0, step=0.001, format="%.3f")); cable = float(c[2].number_input("Verticaal hoogteverschil tot sensormembraan (m)", min_value=0.0, value=5.0, step=0.001, format="%.3f")); mode_label = c[3].selectbox("Type loggerwaarde", list(LOGGER_MODES)); logger_mode = LOGGER_MODES[mode_label]
    c = st.columns(4); logger_unit = c[0].selectbox("Eenheid loggerwaarde", list(PRESSURE_UNIT_FACTORS_TO_PA if logger_mode == "absolute_pressure" else LENGTH_UNIT_FACTORS_TO_M), index=list(PRESSURE_UNIT_FACTORS_TO_PA).index("hPa") if logger_mode == "absolute_pressure" else 0); knmi_unit = c[1].selectbox("Eenheid KNMI-luchtdruk", list(PRESSURE_UNIT_FACTORS_TO_PA), index=list(PRESSURE_UNIT_FACTORS_TO_PA).index("0,1 hPa"), disabled=logger_mode != "absolute_pressure"); default_temp = float(c[2].number_input("Standaard watertemperatuur (°C)", 0.0, 40.0, 12.0, 0.1)); temp_correction = c[3].checkbox("Temperatuurcorrectie", True)
    st.info(f"Sensorhoogte: **{top - cable:.3f} m NAP**")
    st.header("2. Loggerbestand"); upload = st.file_uploader("Upload loggerbestand", type=["dat", "bin", "csv", "txt"])
    if upload is None: st.stop()
    logger_bytes = upload.getvalue()
    if len(logger_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024: st.error(f"Het bestand is groter dan {MAX_FILE_SIZE_MB} MB."); st.stop()
    kind, encoding, confidence = detect_file_kind(logger_bytes); st.info(f"Detectie: **{'tekst' if kind == 'text' else 'binair'}**, indicatieve score {confidence:.0%}.")
    chosen = st.radio("Verwerkingsmethode", ["Automatisch", "Tekstbestand", "Binair bestand met vaste records"], horizontal=True)
    effective = ("Tekstbestand" if kind == "text" else "Binair bestand met vaste records") if chosen == "Automatisch" else chosen
    try:
        if effective == "Tekstbestand":
            separator, decimal, header = table_settings("logger"); raw = read_text_table(logger_bytes, encoding, separator, decimal, header); st.dataframe(raw.head(20), use_container_width=True)
            columns = list(raw.columns); c = st.columns(4); date = c[0].selectbox("Datum- of tijdkolom", columns); time = c[1].selectbox("Aparte tijdkolom", ["Geen"] + columns); value = c[2].selectbox("Loggerwaarde", columns); temperature = c[3].selectbox("Temperatuurkolom", ["Geen"] + columns)
            tz = timezone_selector("logger_text_timezone", "Nederlandse lokale tijd"); day_first = st.checkbox("Datum gebruikt dag-maand-jaar", True)
            logger_data = prepare_text_logger(raw, date, None if time == "Geen" else time, value, None if temperature == "Geen" else temperature, tz, day_first)
        else:
            config = binary_configuration(logger_bytes); tz = timezone_selector("logger_binary_timezone", "Nederlandse lokale tijd")
            if st.button("Test eerste 100 binaire records"):
                st.dataframe(decode_binary_logger(logger_bytes, config, 100), use_container_width=True)
            logger_data = prepare_binary_logger(cached_decode(logger_bytes, config), tz)
    except ValueError as exc:
        st.error(str(exc)); st.stop()
    st.dataframe(logger_data.head(50), use_container_width=True)
    if logger_mode == "absolute_pressure":
        st.header("3. KNMI-luchtdruk"); knmi_upload = st.file_uploader("Upload KNMI-bestand", type=["dat", "csv", "txt"])
        if knmi_upload is None: st.stop()
        knmi_bytes = knmi_upload.getvalue(); knmi_kind, knmi_encoding, _ = detect_file_kind(knmi_bytes)
        if knmi_kind != "text": st.error("Het KNMI-bestand moet tekstueel zijn."); st.stop()
        separator, decimal, header = table_settings("knmi"); raw = read_text_table(knmi_bytes, knmi_encoding, separator, decimal, header); st.dataframe(raw.head(20), use_container_width=True)
        columns = list(raw.columns); c = st.columns(3); date = c[0].selectbox("KNMI datum- of tijdkolom", columns); time = c[1].selectbox("KNMI aparte tijdkolom", ["Geen"] + columns); pressure = c[2].selectbox("KNMI-luchtdrukkolom", columns)
        tz = timezone_selector("knmi_timezone", "UTC"); day_first = st.checkbox("KNMI-datum gebruikt dag-maand-jaar", True); maximum_gap = float(st.number_input("Maximale KNMI-datagat (uren)", 1.0, 168.0, 6.0))
        try: knmi_data = prepare_knmi_data(raw, date, None if time == "Geen" else time, pressure, tz, day_first)
        except ValueError as exc: st.error(str(exc)); st.stop()
    else:
        knmi_data = pd.DataFrame(); maximum_gap = 6.0
    st.header("4. Berekening")
    if not st.button("Bereken waterstanden", type="primary", use_container_width=True): st.stop()
    configuration = PeilfilterConfiguration(filter_id.strip() or "Onbekend", top, cable, logger_mode, logger_unit, knmi_unit, default_temp, temp_correction)
    try:
        if logger_mode == "absolute_pressure": combined = interpolate_knmi_pressure(logger_data, knmi_data, knmi_unit, maximum_gap)
        else:
            combined = logger_data.copy(); combined["luchtdruk_pa"] = pd.NA; combined["knmi_broninterval_uur"] = pd.NA; combined["buiten_knmi_periode"] = False
        result = calculate_water_levels(combined, configuration)
    except (ValueError, TypeError, KeyError) as exc:
        LOGGER.exception("Berekening mislukt"); st.error(f"De berekening is mislukt: {exc}"); st.stop()
    valid = result.dropna(subset=["waterstand_nap_m"])
    if valid.empty: st.error("Er zijn geen geldige waterstanden berekend."); st.stop()
    c = st.columns(4); c[0].metric("Minimum", f"{valid['waterstand_nap_m'].min():.3f} m NAP"); c[1].metric("Gemiddelde", f"{valid['waterstand_nap_m'].mean():.3f} m NAP"); c[2].metric("Maximum", f"{valid['waterstand_nap_m'].max():.3f} m NAP"); c[3].metric("Geldige records", f"{len(valid):,}")
    st.plotly_chart(create_water_level_figure(result), use_container_width=True)
    st.dataframe(result, use_container_width=True, hide_index=True)
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", configuration.filter_id).strip("_") or "peilfilter"
    st.download_button("Download berekende waterstanden", dataframe_to_csv_bytes(result), f"{safe_id}_waterstanden_nap.csv", "text/csv", type="primary", use_container_width=True)

if __name__ == "__main__":
    main()
