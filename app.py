import os
import math
from datetime import datetime
import base64
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import keras
import folium
from streamlit_folium import st_folium
from folium.plugins import BeautifyIcon
import plotly.graph_objects as go

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="EV Charging — Smart Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------
# Paths 
# -------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
MODELS_DIR = os.path.join(APP_DIR, "models")
ASSETS_DIR = os.path.join(APP_DIR, "assets")

BACKGROUND_IMG_LOCAL = os.path.join(ASSETS_DIR, "Background.png")
LOGO_IMG_LOCAL = os.path.join(ASSETS_DIR, "Logo.png")

BACKGROUND_IMG_FALLBACK = os.path.join("C:", os.sep, "EV_Charging_Streamlit", "assets", "Background.png")
LOGO_IMG_FALLBACK = os.path.join("C:", os.sep, "EV_Charging_Streamlit", "assets", "Logo.png")

UPLOADED_IMG_PATH = "/mnt/data/5c14069d-0924-450f-9098-89c23e6bc9d8.png"

def _exists(p):
    return (p is not None) and os.path.exists(p)

BACKGROUND_IMG_PATH = (
    BACKGROUND_IMG_LOCAL
    if _exists(BACKGROUND_IMG_LOCAL)
    else (
        BACKGROUND_IMG_FALLBACK
        if _exists(BACKGROUND_IMG_FALLBACK)
        else (UPLOADED_IMG_PATH if _exists(UPLOADED_IMG_PATH) else None)
    )
)
LOGO_IMG_PATH = (
    LOGO_IMG_LOCAL
    if _exists(LOGO_IMG_LOCAL)
    else (
        LOGO_IMG_FALLBACK
        if _exists(LOGO_IMG_FALLBACK)
        else (UPLOADED_IMG_PATH if _exists(UPLOADED_IMG_PATH) else None)
    )
)

def _img_to_data_uri(path):
    if not path:
        return None
    if str(path).startswith("data:"):
        return path
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception:
        return None
    b64 = base64.b64encode(data).decode("utf-8")
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
    return f"data:image/{ext};base64,{b64}"

BACKGROUND_IMG = _img_to_data_uri(BACKGROUND_IMG_PATH) or BACKGROUND_IMG_PATH or ""
LOGO_IMG = _img_to_data_uri(LOGO_IMG_PATH) or LOGO_IMG_PATH or ""

# -------------------------
# Styling: dark theme, background
# -------------------------
page_css = f"""
<style>

[data-testid="stAppViewContainer"] {{
  background: linear-gradient(rgba(10, 10, 15, 0.7), rgba(10, 10, 15, 0.7)), url("{BACKGROUND_IMG}");
  background-size: cover;
  background-attachment: fixed;
  background-position: center;
  background-repeat: no-repeat;
  min-height: 100vh;
}}

body, 
[data-testid="stAppViewContainer"] {{
    color: #e0e0e0;
    font-family: 'Outfit', 'Inter', sans-serif;
}}

h1, h2, h3, h4, h5, h6 {{
    color: #ffeb3b !important;
    font-weight: 600;
}}

.small-muted {{
    color: #a0a0a0 !important;
    font-size: 0.85rem;
}}

/* Streamlit elements */
.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stSlider label,
.stDateInput label,
.stTimeInput label {{
    color: #ffeb3b !important;
}}

[data-testid="stMetricValue"], 
[data-testid="stMetricDelta"] {{
    color: #ffeb3b !important;
}}

[data-testid="stHeader"] {{ 
    background: rgba(0,0,0,0); 
}}

.glass {{
  background: rgba(20,20,25,0.75);
  border-radius: 12px;
  padding: 16px;
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
}}

.header-row {{ 
    display:flex; 
    align-items:center; 
    gap:20px; 
    margin-bottom:12px; 
}}

.header-logo {{
  width:100px;
  height:100px;
  object-fit:contain;
  border-radius:12px;
  padding:8px;
  background: rgba(255,255,255,0.07);
}}

.map-legend {{
  position: fixed; 
  bottom: 60px; 
  left: 28px; 
  z-index: 9999;
  border-radius: 10px; 
  padding: 8px 12px; 
  font-size: 13px;
  background: rgba(10,10,15,0.9); 
  color:#ffeb3b !important; 
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 6px 18px rgba(0,0,0,0.6);
}}

.element-container .folium-map, 
.stFolium iframe, 
.folium-map iframe, 
.leaflet-container, 
.leaflet-container iframe {{
  width: 100% !important;
  height: 100% !important;
  min-height: 520px;
}}

/* Leaflet Popups Fix: Make text readable on white popup background */
.leaflet-popup-content,
.leaflet-popup-content * {{
    color: #333333 !important;
}}

/* Recommendation cards */
.rec-card {{
    background: rgba(20, 20, 25, 0.75);
    padding: 14px 18px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 4px 14px rgba(0,0,0,0.5);
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
}}
.rec-card h2, .rec-card h3, .rec-card h4 {{
    color: #ffeb3b !important;
    font-weight: 700;
}}

</style>
"""

st.markdown(page_css, unsafe_allow_html=True)

logo_html = f'<img class="header-logo" src="{LOGO_IMG}" alt="logo">' if LOGO_IMG else ''
st.markdown(
    f"""
    <div class="header-row">
      {logo_html}
      <div>
        <h2 style="margin:0;color:#fff">EV Charging Smart Predictor</h2>
        <div class="small-muted">Occupancy, demand & optimized recommendations</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Utilities
# -------------------------
def ensure_file(path):
    if not os.path.exists(path):
        st.error(f"Required file not found: {path}")
        st.stop()
    return path

def csv_path(fname):
    return ensure_file(os.path.join(DATA_DIR, fname))

def model_path(fname):
    return ensure_file(os.path.join(MODELS_DIR, fname))

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def clip01(arr):
    return np.clip(arr, 0.0, 1.0)

# -------------------------
# Cached loaders
# -------------------------
@st.cache_data
def load_stations():
    path = csv_path("ev_stations.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    if 'latitude' in df.columns and 'lat' not in df.columns:
        df = df.rename(columns={'latitude':'lat'})
    if 'longitude' in df.columns and 'lon' not in df.columns:
        df = df.rename(columns={'longitude':'lon'})
    if 'num_chargers' in df.columns:
        df['num_chargers'] = pd.to_numeric(df['num_chargers'], errors='coerce').fillna(0).astype(int)
    df['base_price'] = pd.to_numeric(df.get('base_price', pd.Series([8.0]*len(df))),
                                     errors='coerce').fillna(8.0)
    if 'station_id' not in df.columns:
        st.error("ev_stations.csv must contain 'station_id' column.")
        st.stop()
    if 'lat' not in df.columns or 'lon' not in df.columns:
        st.error("ev_stations.csv must contain latitude/longitude columns (lat/lon or latitude/longitude).")
        st.stop()
    return df

@st.cache_data
def load_history():
    path = csv_path("train_processed.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.columns = df.columns.str.lower()
    if 'utilization' in df.columns:
        df['utilization'] = pd.to_numeric(df['utilization'], errors='coerce').fillna(0.0)
    return df

@st.cache_data
def load_history_unscaled():
    df = load_history().copy()
    scaler_path = model_path("scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        scaled_cols = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else []
        if "utilization" in scaled_cols:
            util_idx = scaled_cols.index("utilization")
            mean_val = scaler.mean_[util_idx]
            scale_val = scaler.scale_[util_idx]
            df['utilization'] = df['utilization'] * scale_val + mean_val
    if 'utilization' in df.columns:
        df['utilization'] = pd.to_numeric(df['utilization'], errors='coerce').fillna(0.0).clip(0.0, 1.0)
    return df

@st.cache_resource
def load_model_artifacts():
    mpath = model_path("tcn_model_final.keras")
    scaler_path = model_path("scaler.pkl")
    seq_path = model_path("seq_feature_cols.pkl")
    le_path = model_path("station_label_encoder.pkl")

    model = keras.models.load_model(mpath)

    scaler = joblib.load(scaler_path)
    seq_cols = joblib.load(seq_path)
    le = joblib.load(le_path)

    scaled_cols = list(scaler.feature_names_in_) if hasattr(
        scaler, "feature_names_in_"
    ) else []

    return model, scaler, seq_cols, le, scaled_cols

# -------------------------
# Forecast helpers
# -------------------------
def scale_partial_sequence(seq, scaler, seq_feature_cols, scaled_cols):
    idxs = [seq_feature_cols.index(c) for c in scaled_cols if c in seq_feature_cols]
    if len(idxs) == 0:
        return seq
    N, F = seq.shape[0], len(idxs)
    sub = seq[:, idxs].reshape(-1, F)
    sub_scaled = scaler.transform(sub)
    seq_out = seq.copy()
    seq_out[:, idxs] = sub_scaled.reshape(N, F)
    return seq_out

def forecast_sequence(
    model, full_df, station_id, ts, seq_len,
    seq_feature_cols, scaler, scaled_cols, le_station
):
    g = full_df[full_df["station_id"] == station_id].sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(ts)
    idx = g[g["timestamp"] < ts].shape[0] - 1
    if idx < seq_len - 1:
        raise ValueError("Not enough history for chosen timestamp.")
    seq_window = g.loc[idx - seq_len + 1: idx, seq_feature_cols].values.astype(np.float32)
    seq_window = scale_partial_sequence(seq_window, scaler, seq_feature_cols, scaled_cols)
    seq_window = seq_window.reshape(1, seq_len, len(seq_feature_cols))
    s_enc = le_station.transform([station_id])[0]
    pred = model.predict([seq_window, np.array([s_enc])], verbose=0)
    return clip01(pred.flatten())

def compute_congestion(full_df, station_id, ts, window=3):
    g = full_df[full_df["station_id"] == station_id].sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(ts)
    idx = g[g["timestamp"] < ts].shape[0] - 1
    if idx < 0:
        return 0.0
    start = max(0, idx - window + 1)
    vals = g.loc[start:idx, "utilization"].values
    if len(vals) == 0:
        return 0.0
    mean_val = float(np.nanmean(vals))
    return float(max(0.0, min(1.0, mean_val)))

# -------------------------
# Shared scoring function
# -------------------------
def score_station(
    congestion, dist_km, max_dist, base_price, max_price,
    w_cong, w_dist, w_price
):
    congestion = float(max(0.0, min(1.0, float(congestion))))
    max_dist = float(max(1.0, float(max_dist)))
    max_price = float(max(1.0, float(max_price)))
    dist_norm = min(1.0, float(dist_km) / max_dist)
    price_norm = min(1.0, float(base_price) / max_price)
    score = (w_cong * congestion) + (w_dist * dist_norm) + (w_price * price_norm)
    return float(max(0.0, score))

# -------------------------
# Load data & model
# -------------------------
with st.spinner("Loading data & models..."):
    stations_df = load_stations()
    full_history = load_history()
    full_history_unscaled = load_history_unscaled()
    model, scaler, seq_feature_cols, le_station, scaled_cols = load_model_artifacts()

TYPE_COLORS = {
    "residential":"#2ECC71",
    "office":"#F39C12",
    "mall":"#9B59B6",
    "highway":"#E74C3C"
}
DEFAULT_COLOR = "#AAAAAA"

if "user_loc" not in st.session_state:
    st.session_state["user_loc"] = None
if "pred_cache" not in st.session_state:
    st.session_state["pred_cache"] = {}
if "last_params" not in st.session_state:
    st.session_state["last_params"] = None

# -------------------------
# Tabs
# -------------------------
tab_map, tab_pred, tab_recs, tab_stats = st.tabs(
    ["🗺 Map", "🔮 Prediction", "⭐ Recommendations", "📊 Stats"]
)

# -------------------------
# MAP TAB
# -------------------------
with tab_map:
    # Sidebar Location Selector
    with st.sidebar:
        st.markdown("### 📍 Set Your Location")
        
        # Dropdown to choose from existing stations as user's starting point
        st.markdown("Choose a station to set your location near it:")
        selected_station_for_loc = st.selectbox(
            "Select reference station", 
            ["-- Choose Station --"] + stations_df["station_id"].tolist(),
            key="ref_station"
        )
        if selected_station_for_loc != "-- Choose Station --":
            ref_row = stations_df[stations_df["station_id"] == selected_station_for_loc].iloc[0]
            st.session_state["user_loc"] = (ref_row.lat, ref_row.lon)
            st.success(f"📌 Location set near {selected_station_for_loc}!")
            
        # Or enter coordinates manually
        st.markdown("---")
        st.markdown("Or set coordinates manually:")
        center_lat = stations_df['lat'].mean()
        center_lon = stations_df['lon'].mean()
        manual_lat = st.number_input("Latitude", value=center_lat, format="%.5f", key="man_lat")
        manual_lon = st.number_input("Longitude", value=center_lon, format="%.5f", key="man_lon")
        if st.button("Set Custom Location", key="set_cust_loc"):
            st.session_state["user_loc"] = (manual_lat, manual_lon)
            st.success("📌 Custom location set!")

    st.markdown(
        '<div class="glass"><h3 style="margin:4px">Select your location (click on the map)</h3></div>',
        unsafe_allow_html=True
    )

    center = [stations_df['lat'].mean(), stations_df['lon'].mean()]
    m = folium.Map(location=center, zoom_start=12, control_scale=True, tiles="CartoDB.DarkMatter", scrollWheelZoom=False)

    # station markers
    for _, r in stations_df.iterrows():
        color = TYPE_COLORS.get(str(r.get("station_type","")).lower(), DEFAULT_COLOR)
        popup_html = (
            f"<b>{r.station_id}</b>"
            f"<br>Type: {r.get('station_type','N/A')}"
            f"<br>#Chargers: {int(r.get('num_chargers',0))}"
            f"<br>Price: ₹{r.get('base_price','N/A')}"
        )
        folium.CircleMarker(
            location=(r.lat, r.lon),
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(m)

    # BLACK user marker + green nearest station
    if st.session_state["user_loc"]:
        u_lat, u_lon = st.session_state["user_loc"]
        user_icon = BeautifyIcon(
            icon="user",
            icon_shape="marker",
            border_color="deepskyblue",
            background_color="deepskyblue",
            text_color="white",
            inner_icon_style="font-size:12px;"
        )
        folium.Marker(
            location=(u_lat, u_lon),
            icon=user_icon,
            popup="You (selected)"
        ).add_to(m)

        stations_df["dist_km"] = stations_df.apply(
            lambda r: haversine_km(u_lat, u_lon, r.lat, r.lon), axis=1
        )
        nearest = stations_df.loc[stations_df["dist_km"].idxmin()]
        folium.CircleMarker(
            location=(nearest.lat, nearest.lon),
            radius=10,
            color="green",
            fill=True,
            fill_color="green",
            popup=f"Nearest: {nearest.station_id}",
        ).add_to(m)

    legend_html = """
    <div style="
         position: absolute; 
         bottom: 20px; left: 20px; 
         z-index: 9999;
         width: 180px;
         padding: 10px 14px;
         font-size: 14px;
         background: rgba(0,0,0,0.55);
         border: 1px solid rgba(255,255,255,0.15);
         border-radius: 10px;
         color: #ffeb3b;
         backdrop-filter: blur(6px);
    ">
    <b style="font-size:15px;">Station Types</b><br><br>
    <span style="color:#2ECC71;">●</span> Residential<br>
    <span style="color:#F39C12;">●</span> Office<br>
    <span style="color:#9B59B6;">●</span> Mall<br>
    <span style="color:#E74C3C;">●</span> Highway<br>
    <span style="color:green;">●</span> Nearest<br>
    <span style="color:deepskyblue;">●</span> You (selected)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    map_out = st_folium(m, width=None, height=560, returned_objects=["last_clicked"])

    if map_out and map_out.get("last_clicked"):
        lat = map_out["last_clicked"]["lat"]
        lng = map_out["last_clicked"]["lng"]
        new_loc = (lat, lng)
        if st.session_state.get("user_loc") != new_loc:
            st.session_state["user_loc"] = new_loc
            st.rerun()

    if st.session_state["user_loc"]:
        u_lat, u_lon = st.session_state["user_loc"]
        stations_df["dist_km"] = stations_df.apply(
            lambda r: haversine_km(u_lat, u_lon, r.lat, r.lon), axis=1
        )
        nearest = stations_df.loc[stations_df["dist_km"].idxmin()]

        st.markdown('<div class="glass" style="margin-top:12px">', unsafe_allow_html=True)
        st.markdown(
            f"**Nearest station:** 🟢 **{nearest.station_id}** — "
            f"{nearest.get('station_type','').title()} — "
            f"**{nearest.dist_km:.2f} km**"
        )
        st.markdown(
            f"Chargers: {int(nearest.get('num_chargers',0))} • "
            f"Base price: ₹{nearest.get('base_price','N/A')}"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # quick recs (unchanged logic)
        rec_radius = 15.0
        max_price = stations_df['base_price'].max() if 'base_price' in stations_df.columns else 10.0
        max_dist = (
            rec_radius
            if rec_radius > 0
            else (stations_df['dist_km'].max() if 'dist_km' in stations_df.columns else 10.0)
        )
        quick_recs = []
        q_w_cong, q_w_dist, q_w_price = 1.0, 0.5, 0.3

        for t in sorted(stations_df["station_type"].unique()):
            cand = stations_df[stations_df["station_type"] == t].copy()
            cand["dist"] = cand.apply(
                lambda r: haversine_km(u_lat, u_lon, r.lat, r.lon), axis=1
            )
            if rec_radius > 0:
                cand = cand[cand["dist"] <= rec_radius]
            rows = []
            for _, r in cand.iterrows():
                cong = compute_congestion(full_history_unscaled, r.station_id, datetime.now(), window=3)
                score = score_station(
                    cong, r["dist"], max_dist,
                    r.get("base_price", 8.0),
                    max_price,
                    q_w_cong, q_w_dist, q_w_price
                )
                rows.append(
                    (r.station_id, cong, r["dist"], float(r.get("base_price", 8.0)), score)
                )
            if len(rows) == 0:
                continue
            best = sorted(rows, key=lambda x: x[4])[0]
            quick_recs.append(
                {
                    "type": t,
                    "station": best[0],
                    "congestion": best[1],
                    "dist_km": best[2],
                    "price": best[3],
                    "score": best[4],
                }
            )

        if len(quick_recs) > 0:
            st.subheader("Quick recommendations (nearby, radius 5 km)")
            cols = st.columns(len(quick_recs))
            for c, rec in zip(cols, quick_recs):
                with c:
                    st.markdown(
                        '<div class="glass" style="text-align:left">',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"### {rec['type'].title()}")
                    st.markdown(f"**{rec['station']}**")
                    st.markdown(
                        f"<div class='small-muted'>Distance: {rec['dist_km']:.2f} km</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='small-muted'>Expected congestion: {rec['congestion']:.2f}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='small-muted'>Price: ₹{rec['price']:.2f}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='small-muted'>Score: {rec['score']:.3f}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No nearby stations within 5 km. Use Recommendations tab for global view.")
    else:
        st.info("Click on the map to choose your location. Recommendations will appear here.")

# -------------------------
# PREDICTION TAB (FULLY FIXED)
# -------------------------
with tab_pred:
    st.markdown(
        '<div class="glass" style="padding:12px">'
        '<h3 style="margin:6px">Prediction</h3>'
        '<div class="small-muted">Set timestamp and horizon, then click Run Forecast.</div></div>',
        unsafe_allow_html=True,
    )

    # Auto-detect model expected input shape (seq_len, n_features)
    try:
        model_input_shape = model.inputs[0].shape
        MODEL_SEQ_LEN = int(model_input_shape[1]) 
        MODEL_N_FEAT = int(model_input_shape[2])
    except:
        MODEL_SEQ_LEN = 72
        MODEL_N_FEAT = len(seq_feature_cols)

    st.caption(f"Model expects: sequence={MODEL_SEQ_LEN}, features={MODEL_N_FEAT}")

    # auto-select nearest station if user picked location
    if st.session_state.get("user_loc"):
        u_lat, u_lon = st.session_state["user_loc"]
        stations_df["dist_to_user_km"] = stations_df.apply(
            lambda r: haversine_km(u_lat, u_lon, r.lat, r.lon), axis=1
        )
        default_station = stations_df.loc[
            stations_df["dist_to_user_km"].idxmin()
        ]["station_id"]
    else:
        default_station = stations_df["station_id"].iloc[0]

    station_list = stations_df["station_id"].tolist()
    default_index = stations_df.index[stations_df["station_id"] == default_station].tolist()[0]
    sel_station = st.selectbox("Station for prediction", station_list, index=default_index)

    FIXED_SEQ_LEN = MODEL_SEQ_LEN
    FIXED_AVG_KWH = 12.5
    FIXED_PRICE_MULTIPLIER = 1.0

    with st.form(key="predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            date_sel = st.date_input("Forecast date", datetime.now().date())
            time_sel = st.time_input("Forecast time", datetime.now().time())
        with c2:
            horizon = st.number_input(
                "Horizon (hours to predict)", 1, 24, 24, 1
            )
        run_forecast = st.form_submit_button("Run Forecast")

    if run_forecast:
        ts = datetime.combine(date_sel, time_sel)

        # ---- PROPER history extraction ----
        g = full_history[full_history["station_id"] == sel_station].sort_values("timestamp").reset_index(drop=True)

        if g.empty:
            st.error("No historical data found for this station.")
            st.stop()

        # If user timestamp > latest history → fallback
        if ts > g["timestamp"].max():
            st.warning("Selected timestamp is beyond history. Using latest available timestamp.")
            ts = g["timestamp"].max()

        # Find nearest earlier OR equal timestamp
        idx = g[g["timestamp"] <= ts].shape[0] - 1

        # If still no match → use closest timestamp
        if idx < 0:
            idx = (g["timestamp"] - ts).abs().idxmin()
            ts = g.loc[idx, "timestamp"]

        # Ensure we can take a 72-hour window
        if idx < FIXED_SEQ_LEN - 1:
            st.error(
                f"Not enough data before this timestamp. Need {FIXED_SEQ_LEN} hours, "
                f"but only {idx+1} available."
            )
            st.stop()

        # ---- extract correct 72×33 window ----
        seq_window = g.iloc[idx - FIXED_SEQ_LEN + 1 : idx + 1][seq_feature_cols].values.astype(np.float32)

        # ---- scale only numeric features ----
        seq_window = scale_partial_sequence(
            seq_window, scaler, seq_feature_cols, scaled_cols
        )

        # reshape for model
        seq_window = seq_window.reshape(1, FIXED_SEQ_LEN, MODEL_N_FEAT)

        # embedding
        try:
            s_enc = le_station.transform([sel_station])[0]
        except:
            st.error("Station ID encoding mismatch. Check station_label_encoder.pkl.")
            st.stop()

        # ---- RUN MODEL ----
        pred_raw = model.predict([seq_window, np.array([s_enc])], verbose=0).flatten()

        # horizon slice
        preds = pred_raw[:horizon]

        # ---- INVERSE SCALE PREDICTIONS ----
        if "utilization" in scaled_cols:
            util_idx = scaled_cols.index("utilization")
            mean_val = scaler.mean_[util_idx]
            scale_val = scaler.scale_[util_idx]
            preds = preds * scale_val + mean_val

        # DO NOT CLIP TO ZERO — instead soft floor at 0.02
        preds = np.clip(preds, 0.02, 1.0)

        # normalize if model output has drift
        preds = preds.astype(float)

        # ---- compute demand ----
        row = stations_df[stations_df["station_id"] == sel_station].iloc[0]
        num_chargers = int(row.get("num_chargers", 0))
        base_price = float(row.get("base_price", 8.0))
        price_per_kwh = base_price * FIXED_PRICE_MULTIPLIER

        demand_kwh = preds * num_chargers * FIXED_AVG_KWH
        cost_hr = demand_kwh * price_per_kwh

        # ---- UI ----
        hours = list(range(1, len(preds) + 1))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=hours, y=preds, name="Utilization"))
        fig.add_trace(go.Scatter(x=hours, y=demand_kwh, yaxis="y2", name="Demand (kWh)"))
        fig.update_layout(
            xaxis_title="Hour ahead",
            yaxis_title="Utilization",
            yaxis=dict(range=[0,1]),
            height=420,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#eee",
            yaxis2=dict(title="Demand (kWh)", overlaying="y", side="right")
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="glass" style="padding:10px;margin-top:8px">', unsafe_allow_html=True)
        st.metric("Next-hour predicted utilization", f"{preds[0]:.3f}")
        st.metric("Next-hour available chargers", int(round(num_chargers * (1 - preds[0]))))
        st.markdown(f"<div class='small-muted'>Estimated next-hour demand: {demand_kwh[0]:.2f} kWh</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>Estimated next-hour cost: ₹{cost_hr[0]:.2f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# RECOMMENDATIONS TAB (patched)
# -------------------------
with tab_recs:
    st.markdown(
        '<div class="glass" style="padding:12px">'
        '<h3 style="margin:6px">Full Recommendations</h3></div>',
        unsafe_allow_html=True,
    )

    rec_radius = st.number_input(
        "Recommendation radius (km, 0=global)", value=5.0, step=1.0
    )
    user_coords = st.session_state.get("user_loc", None)
    if user_coords:
        u_lat, u_lon = user_coords
        stations_df["dist_to_user_km"] = stations_df.apply(
            lambda r: haversine_km(u_lat, u_lon, r.lat, r.lon), axis=1
        )
    else:
        # if no user coords set, use a large value so global will show when rec_radius=0
        stations_df["dist_to_user_km"] = 0.0

    st.markdown("### Weights (lower score = better)")
    st.markdown(
        "<div class='small-muted'>Traffic load weight: <b>1.2 (fixed)</b></div>",
        unsafe_allow_html=True,
    )
    w_cong2 = 1.2
    w_dist2 = st.slider("Distance weight", 0.0, 2.0, 0.5, key="w_dist2")
    w_price2 = st.slider("Price weight", 0.0, 2.0, 0.3, key="w_price2")

    rows = []
    max_price = (
        stations_df["base_price"].max()
        if "base_price" in stations_df.columns
        else 10.0
    )
    max_dist = (
        rec_radius
        if rec_radius > 0
        else (
            stations_df["dist_to_user_km"].max()
            if "dist_to_user_km" in stations_df.columns
            else 10.0
        )
    )

    for _, r in stations_df.iterrows():
        if rec_radius > 0 and user_coords and r["dist_to_user_km"] > rec_radius:
            continue
        congestion = compute_congestion(
            full_history_unscaled, r.station_id, datetime.now(), window=3
        )
        score = score_station(
            congestion,
            r.get("dist_to_user_km", 0.0),
            max_dist,
            r.get("base_price", 8.0),
            max_price,
            w_cong2,
            w_dist2,
            w_price2,
        )
        rows.append(
            {
                "station_id": r.station_id,
                "station_type": r.get("station_type", ""),
                "congestion": round(congestion, 3),
                "dist_km": round(r.get("dist_to_user_km", 0.0), 3),
                "price": float(r.get("base_price", 8.0)),
                "num_chargers": int(r.get("num_chargers", 0)),
                "score": round(score, 4),
            }
        )

    if len(rows) == 0:
        st.info("No stations matched (maybe radius too small).")
    else:
        rec_df = pd.DataFrame(rows)
        for t in rec_df["station_type"].unique():
            st.subheader(f"Top {t.title()} stations")
            df_t = rec_df[rec_df["station_type"] == t].sort_values("score").reset_index(
                drop=True
            )
            st.table(df_t.head(15))


# -------------------------
# STATS TAB
# -------------------------
with tab_stats:
    st.markdown(
        '<div class="glass" style="padding:12px">'
        '<h3 style="margin:6px">Station statistics & vacancy visuals</h3></div>',
        unsafe_allow_html=True,
    )

    station_util = (
        full_history_unscaled.groupby("station_id")["utilization"]
        .mean()
        .rename("mean_util")
        .reset_index()
    )
    station_summary = stations_df.merge(station_util, on="station_id", how="left").fillna(
        0.0
    )
    station_summary["mean_util"] = station_summary["mean_util"].clip(0.0, 1.0)
    station_summary["vacancy_pct"] = (1.0 - station_summary["mean_util"]) * 100.0

    # 1) Top vacancy bar chart
    top_n = 25
    ss_sorted = station_summary.sort_values("vacancy_pct", ascending=False).head(top_n)
    fig1 = go.Figure()
    fig1.add_trace(
        go.Bar(
            x=ss_sorted["station_id"],
            y=ss_sorted["vacancy_pct"],
            name="Vacancy %",
            marker_color="#74b9ff",
        )
    )
    fig1.update_layout(
        title=f"Top {top_n} Stations by Vacancy %",
        xaxis_title="Station ID",
        yaxis_title="Vacancy (%)",
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#eee",
    )
    st.plotly_chart(fig1, use_container_width=True)

    # 2) Vacancy % vs chargers — jitter + labels
    jitter_x = station_summary["num_chargers"] + np.random.uniform(
        -0.3, 0.3, size=len(station_summary)
    )
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=jitter_x,
            y=station_summary["vacancy_pct"],
            mode="markers+text",
            text=station_summary["station_id"],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=8, color="#74b9ff"),
        )
    )
    fig2.update_layout(
        title="Station spots (#chargers) vs Vacancy %",
        xaxis_title="Number of chargers (jittered)",
        yaxis_title="Vacancy (%)",
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#eee",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 3) Avg vacancy and avg chargers by station type
    agg = (
        station_summary.groupby("station_type")
        .agg(
            avg_vacancy_pct=("vacancy_pct", "mean"),
            avg_chargers=("num_chargers", "mean"),
            count=("station_id", "count"),
        )
        .reset_index()
    )

    fig3 = go.Figure()

    # Filled bar for Vacancy
    fig3.add_trace(
        go.Bar(
            x=agg["station_type"],
            y=agg["avg_vacancy_pct"],
            name="Avg Vacancy %",
            marker=dict(
                color="#74b9ff",
                pattern=dict(shape="")  # solid
            ),
        )
    )

    # Striped bar for Avg chargers (y2 axis)
    fig3.add_trace(
        go.Bar(
            x=agg["station_type"],
            y=agg["avg_chargers"],
            name="Avg #Chargers",
            yaxis="y2",
            marker=dict(
                color="rgba(0,0,0,0)",
                line=dict(color="#ffa502", width=2),
                pattern=dict(shape="/", fgcolor="#ffa502"),
            ),
        )
    )

    fig3.update_layout(
        title="Avg Vacancy % and Avg #Chargers by Station Type",
        xaxis_title="Station type",
        yaxis=dict(title="Avg Vacancy (%)"),
        yaxis2=dict(
            title="Avg #Chargers",
            overlaying="y",
            side="right",
        ),
        barmode="overlay",
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#eee",
    )
    st.plotly_chart(fig3, use_container_width=True)

st.caption("EV-Lution , Let's Evify Together")
