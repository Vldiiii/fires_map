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

    b = region.bounds  # (minx, miny, maxx, maxy)
    bbox = (b[0], b[1], b[2], b[3])

    # GeoJSON для pydeck
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

def risk_label(p: float) -> str:
    if p < 0.2: return "очень низкий"
    if p < 0.4: return "низкий"
    if p < 0.6: return "средний"
    if p < 0.8: return "высокий"
    return "очень высокий"

with st.spinner("Загрузка данных..."):
    region, border_geojson, bbox = load_border(BORDER_FILE)
    df_all = load_predictions(PRED_FILE)

if df_all.empty:
    st.stop()

if df_all.empty:
    st.warning("Нет данных — запустите prepare_map_data.py")
    st.stop()

st.sidebar.header("Фильтры")

min_prob = st.sidebar.slider(
    "Минимальная вероятность пожара, %", 0, 100, 0, step=5
) / 100

view_mode = st.sidebar.radio("Режим", ["Один день", "Неделя"])

available_dates = sorted(df_all["date_only"].unique())
selected_date   = st.sidebar.selectbox(
    "Дата", available_dates, index=len(available_dates) - 1
)

if view_mode == "Один день":
    view = df_all[df_all["date_only"] == selected_date].copy()
    subtitle = f" {selected_date}"
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
    subtitle = f" {selected_date} — {end_date} (макс. за неделю)"

view = view[view["fire_probability"] >= min_prob].copy()

if view.empty:
    st.warning("Нет данных для выбранных фильтров")
    st.stop()

view["color"]        = view["fire_probability"].apply(risk_color)
view["risk_level"]   = view["fire_probability"].apply(risk_label)
view["prob_percent"] = (view["fire_probability"] * 100).round(1)

st.subheader(subtitle)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Точек на карте",       f"{len(view):,}")
c2.metric("Средняя вероятность",  f"{view['fire_probability'].mean()*100:.1f}%")
c3.metric("Максимальная вероятность", f"{view['fire_probability'].max()*100:.1f}%")
c4.metric("Фактических пожаров",  f"{int(view['fire'].sum()):,}")


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

# Точки прогноза
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
        "<b>Уровень риска:</b> {risk_level}<br/>"
        "<b>Координаты:</b> {lat}, {lon}"
    ),
    "style": {"background": "white", "padding": "8px", "borderRadius": "4px"}
}

st.pydeck_chart(
    pdk.Deck(layers=layers, initial_view_state=view_state,
             tooltip=tooltip, map_style="light"),
    use_container_width=True,
)


st.sidebar.markdown("---\n### Уровни риска")
for color, label in [
    ([80,  180, 80],  "Очень низкий  (<20%)"),
    ([180, 200, 80],  "Низкий        (20–40%)"),
    ([230, 180, 60],  "Средний       (40–60%)"),
    ([230, 100, 50],  "Высокий       (60–80%)"),
    ([220, 40,  40],  "Очень высокий (>80%)"),
]:
    st.sidebar.markdown(
        f"<div style='display:flex;align-items:center;margin:4px 0'>"
        f"<span style='width:14px;height:14px;border-radius:50%;"
        f"background:rgb({color[0]},{color[1]},{color[2]});"
        f"display:inline-block;margin-right:8px'></span>{label}</div>",
        unsafe_allow_html=True,
    )
