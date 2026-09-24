"""Export en visualisatie."""
import pandas as pd
import plotly.graph_objects as go

def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Exporteer als Nederlandse CSV."""
    export = dataframe.copy()
    for column in export.columns:
        if pd.api.types.is_datetime64_any_dtype(export[column]):
            export[column] = export[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    return export.to_csv(index=False, sep=";", decimal=",", na_rep="", lineterminator="\n").encode("utf-8-sig")

def create_water_level_figure(dataframe: pd.DataFrame) -> go.Figure:
    """Maak de interactieve waterstandgrafiek."""
    valid = dataframe.dropna(subset=["tijd", "waterstand_nap_m"])
    figure = go.Figure(go.Scatter(x=valid["tijd"], y=valid["waterstand_nap_m"], mode="lines", name="Waterstand", line={"color": "#0078D4", "width": 2}, hovertemplate="%{x|%d-%m-%Y %H:%M}<br>%{y:.3f} m NAP<extra></extra>"))
    if not dataframe.empty:
        figure.add_hline(y=float(dataframe["bovenkant_peilbuis_nap_m"].iloc[0]), line_dash="dash", line_color="#D83B01", annotation_text="Bovenkant peilbuis")
    figure.update_layout(title="Berekende grondwaterstand", xaxis_title="Datum en tijd", yaxis_title="Waterstand (m NAP)", hovermode="x unified")
    return figure
