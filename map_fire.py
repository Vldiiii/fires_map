import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

BASE_DIR    = Path(__file__).parent
PRED_FILE   = BASE_DIR / "predictions_primorsky.csv"
BORDER_FILE = BASE_DIR / "primorsky_coords.csv"

st.set_page_config(page_title="Прогноз пожарной опасности", layout="wide")
st.title("Прогноз пожарной опасности — Приморский край")

@st.cache_resource
def load_border(path: Path):
    if not path.exists():
        st.warning("Файл границы не найден — карта покажет все данные")
        return None, None, None

    df = pd.read_csv(path)
    lon_col = "longitude" if "longitude" in df.columns else "lon"
    lat_col = "latitude"  if "latitude"  in df.columns else "lat"

    all_coords = list(zip(df[lon_col], df[lat_col]))

    parts, current = [], [all_coords[0]]
    for i in range(1, len(all_coords)):
        dlat = abs(all_coords[i][1] - all_coords[i-1][1])
        dlon = abs(all_coords[i][0] - all_coords[i-1][0])
        if dlat > 1.0 or dlon > 1.0:
            parts.append(current)
            current = []
        current.append(all_coords[i])
    parts.append(current)

    polys = []
    for pts in parts:
        if len(pts) < 3:
            continue
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        try:
            p = make_valid(Polygon(pts))
            if not p.is_empty:
                polys.append(p)
        except Exception:
            pass

    if not polys:
        return None, None, None

    region = make_valid(unary_union(polys))

    b = region.bounds
    bbox = (b[0], b[1], b[2], b[3])

    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": region.__geo_interface__,
            "properties": {}
        }]
    }

    return region, geojson, bbox


@st.cache_data(show_spinner=False)
def load_predictions(path: Path) -> pd.DataFrame:

    try:
        gdrive_url = st.secrets["GDRIVE_URL"]
        import requests, io
        if "drive.google.com" in gdrive_url:
            file_id = gdrive_url.split("/d/")[1].split("/")[0]
            url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
        else:
            url = gdrive_url
        with st.spinner("Загрузка данных с Google Drive..."):
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.content.decode("utf-8")), low_memory=False)
    except (KeyError, Exception):
        if not path.exists():
            st.error(f"Файл прогнозов не найден: {path}")
            return pd.DataFrame()
        df = pd.read_csv(path, low_memory=False)

    df["date"]             = pd.to_datetime(df["date"], errors="coerce")
    df["lat"]              = pd.to_numeric(df["lat"],  errors="coerce")
    df["lon"]              = pd.to_numeric(df["lon"],  errors="coerce")
    df["fire_probability"] = pd.to_numeric(df["fire_probability"], errors="coerce")
    df["fire"]             = df.get("fire", 0).fillna(0).astype(int)
    df = df.dropna(subset=["date", "lat", "lon", "fire_probability"])
    df["date_only"]        = df["date"].dt.date
    return df


def filter_by_region(df: pd.DataFrame, region, bbox):

    if region is None:
        return df

    lon_min, lat_min, lon_max, lat_max = bbox
    mask_bbox = (
        (df["lon"] >= lon_min) & (df["lon"] <= lon_max) &
        (df["lat"] >= lat_min) & (df["lat"] <= lat_max)
    )
    candidates = df[mask_bbox].copy()

    pts   = [Point(x, y) for x, y in zip(candidates["lon"], candidates["lat"])]
    inside = [region.contains(p) for p in pts]
    return candidates[inside].reset_index(drop=True)


def risk_color(p: float) -> list:
    if p < 0.2: return [80,  180, 80,  170]
    if p < 0.4: return [180, 200, 80,  175]
    if p < 0.6: return [230, 180, 60,  180]
    if p < 0.8: return [230, 100, 50,  190]
    return               [220, 40,  40,  210]

with st.spinner("Загрузка данных..."):
    region, border_geojson, bbox = load_border(BORDER_FILE)
    df_all = load_predictions(PRED_FILE)

if df_all.empty:
    st.stop()

st.sidebar.header("Фильтры")

view_mode = st.sidebar.radio("Режим", ["Один день", "Неделя"])

available_dates = sorted(df_all["date_only"].unique())
selected_date   = st.sidebar.selectbox(
    "Дата", available_dates, index=len(available_dates) - 1
)

if view_mode == "Один день":
    view = df_all[df_all["date_only"] == selected_date].copy()
    subtitle = f"{selected_date}"
else:
    end_date = selected_date + timedelta(days=6)
    week = df_all[
        (df_all["date_only"] >= selected_date) &
        (df_all["date_only"] <= end_date)
    ]
    view = week.groupby(["lat","lon"], as_index=False).agg(
        fire_probability=("fire_probability", "max"),
        fire=("fire", "max")
    )
    subtitle = f"{selected_date} — {end_date} (макс. за неделю)"

if view.empty:
    st.warning("Нет данных для выбранных фильтров")
    st.stop()

view["color"] = view["fire_probability"].apply(risk_color)

layers = []

if border_geojson:
    layers.append(pdk.Layer(
        "GeoJsonLayer",
        data=border_geojson,
        get_fill_color=[0, 0, 0, 0],
        get_line_color=[200, 0, 0, 255],
        line_width_min_pixels=2,
        stroked=True,
        filled=False,
        pickable=False,
    ))

layers.append(pdk.Layer(
    "ScatterplotLayer",
    data=view,
    get_position="[lon, lat]",
    get_fill_color="color",
    get_radius=4,
    radius_units="pixels",
    radius_min_pixels=2,
    radius_max_pixels=12,
    pickable=True,
    opacity=0.85,
))

view_state = pdk.ViewState(
    latitude=df_all["lat"].mean(),
    longitude=df_all["lon"].mean(),
    zoom=6, pitch=0,
)

tooltip = {
    "html": (
        "<b>Вероятность пожара:</b> {prob_percent}%<br/>"
        "<b>Координаты:</b> {lat}, {lon}"
    ),
    "style": {"background": "white", "padding": "8px", "borderRadius": "4px"}
}

st.pydeck_chart(
    pdk.Deck(layers=layers, initial_view_state=view_state,
             tooltip=tooltip, map_style="light"),
    use_container_width=True,
)
