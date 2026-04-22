import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from datetime import date, timedelta

st.set_page_config(page_title="Airbnb Host Intelligence · Madrid",
                   page_icon="🏠", layout="wide")

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, p, span, div, h1, h2, h3, h4, h5, label, input, button, textarea {
    font-family: 'Inter', sans-serif;
}

/* Remove Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #F7F7F7; }
.block-container { padding: 0 2rem 4rem 2rem !important; max-width: 1200px; }

/* ── Header banner ── */
.navbar {
    background: linear-gradient(135deg, #FF5A5F 0%, #c0392b 100%);
    border-radius: 24px;
    padding: 40px 48px;
    margin-bottom: 36px;
    display: flex;
    align-items: center;
    gap: 24px;
    box-shadow: 0 8px 32px rgba(255, 90, 95, 0.35);
}
.navbar-logo {
    color: white;
    font-size: 64px;
    font-weight: 800;
    letter-spacing: -3px;
    font-family: 'Inter', sans-serif;
    line-height: 1;
}
.navbar-sub {
    color: rgba(255,255,255,0.9);
    font-size: 20px;
    font-weight: 500;
    border-left: 2px solid rgba(255,255,255,0.4);
    padding-left: 24px;
    margin-left: 4px;
    letter-spacing: 0.2px;
}

/* ── Search ── */
.stTextInput input {
    border-radius: 40px !important;
    border: 1.5px solid #DDDDDD !important;
    padding: 14px 24px !important;
    font-size: 16px !important;
    background: white !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08) !important;
    transition: all 0.2s;
}
.stTextInput input:focus-visible {
    border-color: #FF5A5F !important;
    box-shadow: 0 4px 20px rgba(255,90,95,0.15) !important;
    outline: none !important;
}

/* ── Hero image overlay ── */
.hero-wrap {
    position: relative;
    border-radius: 24px;
    overflow: hidden;
    margin-bottom: 8px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
}
.hero-img {
    width: 100%;
    height: 320px;
    object-fit: cover;
    display: block;
}
.hero-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(to bottom, rgba(0,0,0,0.05) 0%, rgba(0,0,0,0.72) 100%);
}
.hero-content {
    position: absolute;
    bottom: 28px;
    left: 32px;
    right: 32px;
    color: white;
}
.hero-title {
    font-size: 28px;
    font-weight: 800;
    margin: 0 0 10px 0;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    line-height: 1.2;
}
.hero-pill {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 13px;
    font-weight: 500;
    margin-right: 6px;
    margin-bottom: 4px;
}

/* ── Stat strip below hero ── */
.stat-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 28px;
}
.stat-box {
    background: white;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    border: 1px solid #EEEEEE;
}
.stat-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #AAAAAA;
    margin: 0 0 6px 0;
}
.stat-value {
    font-size: 22px;
    font-weight: 800;
    color: #222222;
    margin: 0;
}
.stat-sub {
    font-size: 12px;
    color: #AAAAAA;
    margin: 4px 0 0 0;
}
.stat-value-red  { color: #FF5A5F; }
.stat-value-teal { color: #00A699; }

/* ── Section heading ── */
.sec-head {
    font-size: 20px;
    font-weight: 700;
    color: #222222;
    margin: 0 0 18px 0;
}

/* ── Metric card grid ── */
.card-grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.card-grid-4 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.mcard {
    background: white;
    border-radius: 16px;
    padding: 22px 24px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.06);
    border: 1px solid #EEEEEE;
}
.mcard-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #AAAAAA;
    margin: 0 0 8px 0;
}
.mcard-value {
    font-size: 28px;
    font-weight: 800;
    color: #222222;
    margin: 0 0 6px 0;
    line-height: 1.1;
}
.mcard-delta-pos { font-size: 13px; color: #00A699; font-weight: 500; }
.mcard-delta-neg { font-size: 13px; color: #FF5A5F; font-weight: 500; }
.mcard-delta-neu { font-size: 13px; color: #717171; font-weight: 500; }

/* ── Cluster badge ── */
.cluster-banner {
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 24px;
    font-size: 14px;
    font-weight: 500;
    border-left: 5px solid;
    background: white;
}

/* ── Benchmark bar ── */
.bench-wrap {
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.06);
    border: 1px solid #EEEEEE;
    margin-bottom: 24px;
}
.bench-track {
    background: #F0F0F0;
    border-radius: 10px;
    height: 10px;
    position: relative;
    margin: 20px 0 8px 0;
}
.bench-fill {
    background: linear-gradient(90deg, #00A699, #FF5A5F);
    border-radius: 10px;
    height: 100%;
    position: absolute;
    left: 0;
}
.bench-marker {
    position: absolute;
    top: -5px;
    width: 20px;
    height: 20px;
    background: #FF5A5F;
    border: 3px solid white;
    border-radius: 50%;
    box-shadow: 0 2px 6px rgba(255,90,95,0.5);
    transform: translateX(-50%);
}

/* ── Recommendation cards ── */
.rec-card {
    background: white;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 10px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    border: 1px solid #EEEEEE;
    border-left: 4px solid;
}

/* ── Sentiment score ring (CSS only) ── */
.sent-panel {
    background: white;
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.06);
    border: 1px solid #EEEEEE;
    text-align: center;
    margin-bottom: 20px;
}
.sent-score {
    font-size: 56px;
    font-weight: 800;
    margin: 0;
    line-height: 1;
}
.sent-label {
    font-size: 14px;
    font-weight: 500;
    margin: 8px 0 4px 0;
}
.sent-sub {
    font-size: 13px;
    color: #AAAAAA;
    margin: 0;
}

/* ── Complaint cards ── */
.complaint-card {
    background: white;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 14px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.06);
    border: 1px solid #EEEEEE;
}
.complaint-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}
.complaint-emoji { font-size: 20px; }
.complaint-name {
    font-size: 16px;
    font-weight: 700;
    color: #222222;
    flex: 1;
}
.complaint-count {
    background: #FFF0F0;
    color: #FF5A5F;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
}
.complaint-quote {
    background: #FAFAFA;
    border-left: 3px solid #DDDDDD;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    font-style: italic;
    color: #484848;
    font-size: 14px;
    margin-bottom: 14px;
    line-height: 1.6;
}
.complaint-fix {
    background: #FFF8F0;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #484848;
    border-left: 3px solid #F39C12;
}
.fix-label {
    font-weight: 700;
    color: #F39C12;
    margin-right: 6px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid #EEEEEE;
    background: transparent;
    margin-bottom: 4px;
}
.stTabs [data-baseweb="tab"] {
    padding: 14px 32px;
    font-size: 15px;
    font-weight: 500;
    color: #717171;
    border-bottom: 3px solid transparent;
    margin-bottom: -2px;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #FF5A5F !important;
    border-bottom: 3px solid #FF5A5F !important;
    font-weight: 700;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 28px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: white !important;
    border-right: 1px solid #EEEEEE;
}


/* ── Chart container ── */
.chart-wrap {
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.06);
    border: 1px solid #EEEEEE;
    margin-bottom: 24px;
}

/* ── Revenue card ── */
.rev-card {
    background: linear-gradient(135deg, #FF5A5F 0%, #e04045 100%);
    border-radius: 20px;
    padding: 28px 32px;
    color: white;
    box-shadow: 0 8px 24px rgba(255,90,95,0.35);
    margin-bottom: 24px;
}
.rev-title { font-size: 13px; font-weight: 600; text-transform: uppercase;
             letter-spacing: 0.8px; opacity: 0.8; margin: 0 0 8px 0; }
.rev-amount { font-size: 42px; font-weight: 800; margin: 0 0 4px 0; line-height: 1; }
.rev-sub { font-size: 14px; opacity: 0.85; margin: 0; }

hr { border: none; border-top: 1px solid #EEEEEE; margin: 28px 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    with open("rf_regressor.pkl", "rb") as f:  reg    = pickle.load(f)
    with open("rf_classifier.pkl", "rb") as f:  clf    = pickle.load(f)
    with open("kmeans.pkl", "rb") as f:          km     = pickle.load(f)
    with open("cluster_scaler.pkl", "rb") as f:  scaler = pickle.load(f)
    with open("meta.pkl", "rb") as f:            meta   = pickle.load(f)
    occ_rf   = joblib.load("occupancy_model.pkl")
    occ_cols = joblib.load("occ_feature_cols.pkl")
    return reg, clf, km, scaler, meta, occ_rf, occ_cols

@st.cache_data
def load_data():
    listings  = pd.read_csv("listings.csv")
    sentiment = pd.read_csv("listing_sentiment.csv")
    bench     = pd.read_csv("district_benchmark.csv")
    month_mult = pd.read_csv("month_multipliers.csv").set_index("month")
    dow_mult   = pd.read_csv("dow_multipliers.csv").set_index("day_of_week")
    cluster_occ = listings.groupby("cluster")["estimated_occupancy_l365d"].median().round(0).astype(int).to_dict()
    district_medians = listings.groupby("district_name")["price"].median()
    return listings, sentiment, bench, month_mult, dow_mult, cluster_occ, district_medians

reg, clf, km, scaler, meta, occ_rf, occ_cols = load_models()
listings, sentiment, bench, month_mult, dow_mult, cluster_occ, district_medians = load_data()

CLUSTER_NAMES  = {0: "High Performer", 1: "Dormant", 2: "Premium", 3: "Peripheral"}
CLUSTER_COLORS = {0: "#00A699", 1: "#FF5A5F", 2: "#8B5CF6", 3: "#F59E0B"}
CLUSTER_DESC   = {
    0: "Central, well-priced, consistently booked — the sweet spot of the Madrid market.",
    1: "Central location but underperforming on bookings — strong upside if friction is reduced.",
    2: "Large or premium listing commanding top rates — guests expect a flawless experience.",
    3: "Further from the centre with competitive pricing — volume drives revenue here.",
}
ASPECT_EMOJI = {"noise":"🔊","cleanliness":"🧹","maintenance":"🔧",
                "wifi":"📶","accuracy":"📸","communication":"💬","comfort":"🛏️"}
ASPECT_FIX = {
    "noise":         "Be upfront in your listing description about noise levels — mention if it faces a busy street or neighbours are close. Better window sealing can help, but managing guest expectations upfront prevents the complaint entirely.",
    "cleanliness":   "Review your cleaning checklist and consider a professional service between stays. Bathrooms and kitchen surfaces are what guests notice and photograph first.",
    "maintenance":   "Walk through the property before each stay and fix reported issues promptly. Broken fixtures and worn fittings are the most common cause of 4-star reviews — they signal neglect even when everything else is fine.",
    "wifi":          "Test your speed at speedtest.net and check your router placement. If speeds are inconsistent, say so honestly in your listing — guests who need fast WiFi will self-select out rather than leave a bad review.",
    "accuracy":      "Update your photos and description to match the current state of the property. Any gap between what guests see in the listing and what they find on arrival is the fastest way to lose trust.",
    "communication": "Set up automated messages for check-in instructions and common questions. Most communication complaints come from silence before arrival — a simple day-before message resolves the majority of them.",
    "comfort":       "Check the mattress quality, temperature control, and blackout options. Comfort complaints are seasonal — a fan in summer and extra blankets in winter cover most cases without major investment.",
}
DOW_NAMES   = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}
MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# ── Helpers ────────────────────────────────────────────────────────────────────
def build_features(district, room_type, accommodates, bedrooms, bathrooms,
                   distance, availability, min_nights, reviews, rating, loc_score, superhost, instant):
    cluster_input  = np.array([[0, accommodates, bedrooms, bathrooms, distance, 0, 0, rating, availability]])
    cluster        = km.predict(scaler.transform(cluster_input))[0]
    row = {"accommodates":accommodates,"bedrooms":bedrooms,"bathrooms":bathrooms,
           "distance_to_center_km":distance,"minimum_nights":min_nights,
           "availability_365":availability,"number_of_reviews":reviews,
           "review_scores_rating":rating,"review_scores_location":loc_score,
           "host_is_superhost":int(superhost),"instant_bookable":int(instant),"cluster":cluster}
    for col in meta["feature_cols"]:
        if col.startswith("room_type_"):     row[col] = int(room_type == col.replace("room_type_",""))
        elif col.startswith("district_name_"): row[col] = int(district == col.replace("district_name_",""))
    return pd.DataFrame([row])[meta["feature_cols"]], cluster

def dynamic_price(base_price, target_date):
    m, dow = target_date.month, target_date.weekday()
    mm, dm = month_mult.loc[m,"multiplier"], dow_mult.loc[dow,"multiplier"]
    mp, dp = month_mult.loc[m,"pct_change"], dow_mult.loc[dow,"pct_change"]
    return base_price*mm*dm, mm, dm, mp, dp, mm*dm

def predict_occupancy(district, room_type, accommodates, bedrooms, bathrooms,
                      distance, min_nights, reviews, rating, loc_score, superhost, instant, cluster, suggested_price):
    dist_median = district_medians.get(district, district_medians.median())
    price_ratio = suggested_price / dist_median if dist_median > 0 else 1.0
    row = {"accommodates":accommodates,"bedrooms":bedrooms,"bathrooms":bathrooms,
           "distance_to_center_km":distance,"minimum_nights":min_nights,
           "number_of_reviews":reviews,"review_scores_rating":rating,
           "review_scores_location":loc_score,"host_is_superhost":int(superhost),
           "instant_bookable":int(instant),"cluster":cluster,"price_ratio":price_ratio}
    for col in occ_cols:
        if col.startswith("room_type_"):     row[col] = int(room_type == col.replace("room_type_",""))
        elif col.startswith("district_name_"): row[col] = int(district == col.replace("district_name_",""))
    return int(round(occ_rf.predict(pd.DataFrame([row])[occ_cols])[0]))

def mcard(label, value, delta="", delta_type="neu"):
    delta_html = f'<p class="mcard-delta-{delta_type}">{delta}</p>' if delta else ""
    return f"""<div class="mcard">
        <p class="mcard-label">{label}</p>
        <p class="mcard-value">{value}</p>
        {delta_html}
    </div>"""

# ══════════════════════════════════════════════════════════════════════════════
# NAV BAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="navbar">
    <span class="navbar-logo">airbnb</span>
    <span class="navbar-sub">Host Intelligence · Madrid</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH
# ══════════════════════════════════════════════════════════════════════════════
search_query = st.text_input("", placeholder="🔍  Search by listing ID or name — e.g. 5836616")

matched = sent_row = None
prefill = {}
actual_occupancy = None

if search_query.strip():
    query  = search_query.strip()
    result = listings[listings["id"] == int(query)] if query.isdigit() \
             else listings[listings["name"].str.contains(query, case=False, na=False)]
    if result.empty:
        st.warning("No listing found. Fill in your details manually in the sidebar.")
    else:
        matched          = result.iloc[0]
        actual_occupancy = int(matched["estimated_occupancy_l365d"])
        sent_match       = sentiment[sentiment["listing_id"] == int(matched["id"])]
        if not sent_match.empty:
            sent_row = sent_match.iloc[0]

        # ── Hero image with overlay ──
        if pd.notna(matched.get("picture_url")):
            st.markdown(f"""
            <div class="hero-wrap">
                <img class="hero-img" src="{matched['picture_url']}">
                <div class="hero-overlay"></div>
                <div class="hero-content">
                    <p class="hero-title">{matched['name']}</p>
                    <span class="hero-pill">📍 {matched['district_name']}</span>
                    <span class="hero-pill">{matched['room_type']}</span>
                    <span class="hero-pill">👥 {int(matched['accommodates'])} guests</span>
                    <span class="hero-pill">🛏 {matched['bedrooms']:.0f} bed · 🚿 {matched['bathrooms']:.1f} bath</span>
                    {'<span class="hero-pill"><a href="' + matched['listing_url'] + '" style="color:white;text-decoration:none">View on Airbnb ↗</a></span>' if pd.notna(matched.get('listing_url')) else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#FF5A5F,#e04045);border-radius:24px;
                        padding:40px 32px;margin-bottom:8px;color:white;">
                <p class="hero-title">{matched['name']}</p>
                <span class="hero-pill">📍 {matched['district_name']}</span>
                <span class="hero-pill">{matched['room_type']}</span>
                <span class="hero-pill">👥 {int(matched['accommodates'])} guests</span>
            </div>
            """, unsafe_allow_html=True)

        # ── Stat strip ──
        st.markdown(f"""
        <div class="stat-strip">
            <div class="stat-box">
                <p class="stat-label">Current price</p>
                <p class="stat-value">€{matched['price']:.0f}</p>
                <p class="stat-sub">per night</p>
            </div>
            <div class="stat-box">
                <p class="stat-label">Overall rating</p>
                <p class="stat-value">⭐ {matched['review_scores_rating']:.2f}</p>
                <p class="stat-sub">{int(matched['number_of_reviews'])} reviews</p>
            </div>
            <div class="stat-box">
                <p class="stat-label">Est. occupancy</p>
                <p class="stat-value">{int(matched['estimated_occupancy_l365d'])}</p>
                <p class="stat-sub">nights / year</p>
            </div>
            <div class="stat-box">
                <p class="stat-label">Current revenue</p>
                <p class="stat-value stat-value-red">€{int(matched['price'] * matched['estimated_occupancy_l365d']):,}</p>
                <p class="stat-sub">annual estimate</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        prefill = {
            "district":     matched["district_name"],
            "room_type":    matched["room_type"],
            "accommodates": int(matched["accommodates"]),
            "bedrooms":     int(matched["bedrooms"]) if pd.notna(matched["bedrooms"]) else 1,
            "bathrooms":    float(matched["bathrooms"]) if pd.notna(matched["bathrooms"]) else 1.0,
            "distance":     float(matched["distance_to_center_km"]),
            "availability": int(matched["availability_365"]),
            "min_nights":   int(matched["minimum_nights"]),
            "reviews":      int(matched["number_of_reviews"]),
            "rating":       float(matched["review_scores_rating"]),
            "loc_score":    float(matched["review_scores_location"]) if pd.notna(matched.get("review_scores_location")) else 4.5,
            "superhost":    bool(matched["host_is_superhost"]),
            "instant":      bool(matched["instant_bookable"]),
        }

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## 📋 Listing details")
if matched is not None:
    st.sidebar.success("Pre-filled from your listing.")

district     = st.sidebar.selectbox("District",  meta["districts"],
                index=meta["districts"].index(prefill["district"]) if prefill and prefill["district"] in meta["districts"] else 0)
room_type    = st.sidebar.selectbox("Room type", meta["room_types"],
                index=meta["room_types"].index(prefill["room_type"]) if prefill and prefill["room_type"] in meta["room_types"] else 0)
accommodates = st.sidebar.slider("Guests",                   1,    16,   prefill.get("accommodates", 2))
bedrooms     = st.sidebar.slider("Bedrooms",                 0,    10,   prefill.get("bedrooms", 1))
bathrooms    = st.sidebar.slider("Bathrooms",                0.0,  5.0,  prefill.get("bathrooms", 1.0), 0.5)
distance     = st.sidebar.slider("Distance to centre (km)",  0.0,  15.0, prefill.get("distance", 2.0), 0.1)
availability = st.sidebar.slider("Availability (days/year)", 0,    365,  prefill.get("availability", 180))
min_nights   = st.sidebar.slider("Minimum nights",           1,    30,   prefill.get("min_nights", 2))
reviews      = st.sidebar.number_input("Number of reviews",  0,    5000, prefill.get("reviews", 20))
rating       = st.sidebar.slider("Overall rating",           1.0,  5.0,  prefill.get("rating", 4.5), 0.1)
loc_score    = st.sidebar.slider("Location score",           1.0,  5.0,  prefill.get("loc_score", 4.5), 0.1)
superhost    = st.sidebar.checkbox("Superhost",        value=prefill.get("superhost", False))
instant      = st.sidebar.checkbox("Instant bookable", value=prefill.get("instant", False))

current_price = float(matched["price"]) if matched is not None else None
if current_price is None:
    current_price    = st.sidebar.number_input("Your current nightly price (€)", 10, 2000, 100)
    actual_occupancy = None

# ── Run models ──────────────────────────────────────────────────────────────
X_input, cluster = build_features(district, room_type, accommodates, bedrooms, bathrooms,
                                   distance, availability, min_nights, reviews, rating,
                                   loc_score, superhost, instant)
price_pred    = np.exp(reg.predict(X_input)[0])
demand_pred   = clf.predict(X_input)[0]
demand_proba  = clf.predict_proba(X_input)[0][1]
target_occ    = predict_occupancy(district, room_type, accommodates, bedrooms, bathrooms,
                                   distance, min_nights, reviews, rating, loc_score,
                                   superhost, instant, cluster, price_pred)

b = bench[(bench["district_name"] == district) & (bench["room_type"] == room_type)]
b = b.iloc[0] if not b.empty else None

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📊  Overview", "💬  Guest Reviews", "💰  Pricing & Revenue"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    price_delta = price_pred - current_price

    # Model metrics
    st.markdown('<p class="sec-head">Model analysis</p>', unsafe_allow_html=True)
    d_type = "pos" if price_delta >= 0 else "neg"
    d_text = f"{'↑' if price_delta>=0 else '↓'} €{abs(price_delta):.0f} vs your price"
    dp_type = "pos" if demand_proba >= 0.5 else "neg"

    st.markdown(f"""
    <div class="card-grid-3">
        {mcard("Fair-market price", f"€{price_pred:.0f}/night", d_text, d_type)}
        {mcard("High-demand probability", f"{demand_proba:.0%}", "High demand" if demand_pred==1 else "Low demand", "pos" if demand_pred==1 else "neg")}
        {mcard("Market segment", CLUSTER_NAMES[cluster], CLUSTER_DESC[cluster][:60]+"…", "neu")}
    </div>
    """, unsafe_allow_html=True)

    c = CLUSTER_COLORS[cluster]
    st.markdown(f"""
    <div class="cluster-banner" style="border-color:{c};color:#484848;">
        <strong style="color:{c}">{CLUSTER_NAMES[cluster]}</strong> &nbsp;—&nbsp; {CLUSTER_DESC[cluster]}
    </div>
    """, unsafe_allow_html=True)

    # District benchmark
    if b is not None:
        st.markdown('<p class="sec-head">District benchmark</p>', unsafe_allow_html=True)

        pct25, pct75, med = b["pct25_price"], b["pct75_price"], b["median_price"]
        price_range = pct75 - pct25
        marker_pct  = min(100, max(0, (price_pred - pct25) / price_range * 100)) if price_range > 0 else 50

        st.markdown(f"""
        <div class="bench-wrap">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <span style="font-size:14px;color:#717171;">25th pct &nbsp;<strong style="color:#222">€{pct25:.0f}</strong></span>
                <span style="font-size:14px;color:#717171;">Median &nbsp;<strong style="color:#222">€{med:.0f}</strong></span>
                <span style="font-size:14px;color:#717171;">75th pct &nbsp;<strong style="color:#222">€{pct75:.0f}</strong></span>
            </div>
            <div class="bench-track">
                <div class="bench-fill" style="width:{marker_pct}%"></div>
                <div class="bench-marker" style="left:{marker_pct}%"></div>
            </div>
            <p style="font-size:13px;color:#717171;margin:6px 0 0 0;text-align:center;">
                Your model price <strong style="color:#FF5A5F">€{price_pred:.0f}</strong>
                &nbsp;·&nbsp; {int(b['listing_count'])} listings in this segment
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Recommendations
    st.markdown('<p class="sec-head">Actionable recommendations</p>', unsafe_allow_html=True)

    recs = []
    if b is not None:
        if price_pred < b["pct25_price"]:
            recs.append(("⚠️ Underpriced", "#F59E0B",
                f"Your model price (€{price_pred:.0f}) is below the 25th percentile for {room_type}s in {district} (€{b['pct25_price']:.0f}). You're likely leaving money on the table."))
        elif price_pred > b["pct75_price"]:
            recs.append(("ℹ️ Above market rate", "#3B82F6",
                f"Your model price (€{price_pred:.0f}) sits above the 75th percentile (€{b['pct75_price']:.0f}). Test a lower price for 30 days."))
        else:
            recs.append(("✅ Well-priced", "#00A699",
                f"Your model price (€{price_pred:.0f}) is within the normal range for {room_type}s in {district} (€{b['pct25_price']:.0f} – €{b['pct75_price']:.0f})."))

    if demand_pred == 0 and demand_proba < 0.4:
        recs.append(("📉 Low demand signal", "#EF4444",
            "Low probability of sustained demand. Consider reducing minimum nights, increasing availability, or improving review scores."))
    if not instant:
        recs.append(("⚡ Enable Instant Book", "#F59E0B",
            "Listings with Instant Book receive significantly more views. Removing the approval step reduces friction and boosts bookings."))
    if not superhost:
        recs.append(("🏅 Work toward Superhost", "#8B5CF6",
            "Superhosts appear higher in search results. Focus on response rate, cancellation rate, and review scores to qualify."))
    if cluster == 1:
        recs.append(("😴 Dormant listing alert", "#FF5A5F",
            "Central location, but bookings are well below potential. Usually caused by high minimum nights, low availability, or above-market pricing."))

    rec_html = ""
    for title, color, text in recs:
        rec_html += f"""
        <div class="rec-card" style="border-left-color:{color}">
            <strong>{title}</strong>
            <p style="color:#484848;font-size:14px;margin:6px 0 0 0;">{text}</p>
        </div>"""
    st.markdown(rec_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — GUEST REVIEWS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if sent_row is None:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#AAAAAA;">
            <p style="font-size:48px;margin:0">💬</p>
            <p style="font-size:18px;font-weight:600;color:#484848;margin:12px 0 8px">No listing selected</p>
            <p style="font-size:14px;margin:0">Search for a specific listing above to see guest sentiment analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        sc    = float(sent_row["avg_sentiment"])
        emoji = "😍" if sc >= 0.7 else "🙂" if sc >= 0.4 else "😐"
        label = "Very Positive" if sc >= 0.7 else "Positive" if sc >= 0.4 else "Mixed"
        color = "#00A699" if sc >= 0.7 else "#FF5A5F" if sc < 0.4 else "#F59E0B"

        sc1, sc2 = st.columns([1, 2])
        with sc1:
            st.markdown(f"""
            <div class="sent-panel">
                <p class="sent-score" style="color:{color}">{sc:.2f}</p>
                <p class="sent-label" style="color:{color}">{emoji} {label}</p>
                <p class="sent-sub">out of 1.00 · {int(sent_row['review_count'])} reviews analysed</p>
            </div>
            """, unsafe_allow_html=True)

        with sc2:
            raw_complaints = sent_row.get("complaint_counts", None)
            complaints, snippets = {}, {}
            if pd.notna(raw_complaints) and str(raw_complaints) not in ["nan", "{}", ""]:
                try: complaints = json.loads(str(raw_complaints))
                except: pass
                try: snippets = json.loads(str(sent_row.get("complaint_snippets", "{}")))
                except: pass

            if complaints:
                total = sum(complaints.values())
                st.markdown(f'<p class="sec-head">Complaint breakdown — {total} mentions across {len(complaints)} categories</p>', unsafe_allow_html=True)
                sorted_c = sorted(complaints.items(), key=lambda x: x[1], reverse=True)
                bars_html = ""
                for asp, cnt in sorted_c:
                    pct = cnt / max(complaints.values()) * 100
                    bars_html += f"""
                    <div style="margin-bottom:10px;">
                        <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px;">
                            <span>{ASPECT_EMOJI.get(asp,'•')} {asp.capitalize()}</span>
                            <span style="color:#FF5A5F;font-weight:600">{cnt}</span>
                        </div>
                        <div style="background:#F0F0F0;border-radius:6px;height:8px;">
                            <div style="background:#FF5A5F;width:{pct}%;border-radius:6px;height:8px;"></div>
                        </div>
                    </div>"""
                st.markdown(f'<div style="background:white;border-radius:16px;padding:24px;box-shadow:0 1px 8px rgba(0,0,0,0.06);border:1px solid #EEEEEE;">{bars_html}</div>', unsafe_allow_html=True)
            else:
                st.success("✅ No recurring complaints detected — guests are consistently satisfied.")

        # Complaint detail cards
        if complaints:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="sec-head">What guests said — and what to do about it</p>', unsafe_allow_html=True)
            for asp, cnt in sorted(complaints.items(), key=lambda x: x[1], reverse=True):
                snippet = snippets.get(asp, "")
                fix     = ASPECT_FIX.get(asp, "")
                quote_html = f'<div class="complaint-quote">"{snippet}…"</div>' if snippet else ""
                st.markdown(f"""
                <div class="complaint-card">
                    <div class="complaint-header">
                        <span class="complaint-emoji">{ASPECT_EMOJI.get(asp,'•')}</span>
                        <span class="complaint-name">{asp.capitalize()}</span>
                        <span class="complaint-count">{cnt} mention{'s' if cnt>1 else ''}</span>
                    </div>
                    {quote_html}
                    <div class="complaint-fix">
                        <span class="fix-label">💡 Suggested fix:</span>{fix}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — PRICING & REVENUE
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="sec-head">Dynamic pricing planner</p>', unsafe_allow_html=True)
    st.caption("Night-by-night suggested prices based on real Madrid market seasonal and weekly patterns.")

    today = date.today()
    col_d1, col_d2 = st.columns(2)
    start_d = col_d1.date_input("From", value=today + timedelta(days=1),
                                 min_value=today, max_value=today + timedelta(days=365))
    end_d   = col_d2.date_input("To",   value=today + timedelta(days=8),
                                 min_value=today + timedelta(days=1), max_value=today + timedelta(days=365))

    if end_d <= start_d:
        st.error("End date must be after start date.")
        st.stop()

    num_nights = min((end_d - start_d).days, 90)
    if (end_d - start_d).days > 90:
        st.warning("Showing first 90 nights for clarity.")

    dates, rows = [start_d + timedelta(days=i) for i in range(num_nights)], []
    for d_ in dates:
        suggested, mm, dm, mp, dp, combined = dynamic_price(price_pred, d_)
        rows.append({"Date":d_.strftime("%d %b %Y"),"Day":DOW_NAMES[d_.weekday()],
                     "Month factor":f"{mp:+.1f}%","Day factor":f"{dp:+.1f}%",
                     "Combined":f"{(combined-1)*100:+.1f}%","Suggested (€)":round(suggested),
                     "Your price (€)":round(current_price),"Δ per night":round(suggested-current_price),
                     "_suggested":suggested})

    price_df      = pd.DataFrame(rows)
    suggested_arr = price_df["_suggested"].values

    # Chart
    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    x = np.arange(len(dates))
    colors = ["#FF5A5F" if s > current_price*1.05 else "#00A699" if s < current_price*0.95 else "#CCCCCC"
              for s in suggested_arr]
    ax.bar(x, suggested_arr, color=colors, alpha=0.9, width=0.65)
    ax.axhline(current_price, color="#484848", linewidth=1.5, linestyle="--",
               label=f"Your price  €{current_price:.0f}")
    ax.axhline(price_pred, color="#FF5A5F", linewidth=1.2, linestyle=":",
               label=f"Model base  €{price_pred:.0f}")
    if len(dates) <= 31:
        ax.set_xticks(x)
        ax.set_xticklabels([d_.strftime("%d %b") for d_ in dates], rotation=45, ha="right", fontsize=8)
    else:
        step = max(1, len(dates)//15)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([dates[i].strftime("%d %b") for i in x[::step]], rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"€{v:.0f}"))
    ax.tick_params(colors="#717171", labelsize=9)
    for sp in ax.spines.values(): sp.set_color("#EEEEEE")
    ax.set_ylabel("€ per night", color="#717171", fontsize=10)
    ax.legend(fontsize=9, framealpha=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Period summary
    current_occ_rate   = (actual_occupancy if actual_occupancy else cluster_occ.get(cluster,100)) / 365
    target_occ_rate    = target_occ / 365
    expected_current   = current_price * current_occ_rate * num_nights
    expected_suggested = float(np.mean(suggested_arr)) * target_occ_rate * num_nights
    expected_uplift    = expected_suggested - expected_current

    st.markdown(f"""
    <div class="card-grid-4">
        {mcard("Nights selected", str(num_nights), "", "neu")}
        {mcard("Expected · your price", f"€{expected_current:,.0f}", f"{current_occ_rate*100:.0f}% booking rate", "neu")}
        {mcard("Expected · suggested", f"€{expected_suggested:,.0f}", f"{target_occ_rate*100:.0f}% booking rate", "pos")}
        {mcard("Uplift", f"€{expected_uplift:+,.0f}", f"{expected_uplift/expected_current*100:+.1f}%" if expected_current>0 else "", "pos" if expected_uplift>=0 else "neg")}
    </div>
    """, unsafe_allow_html=True)

    with st.expander("See full night-by-night breakdown"):
        display_df = price_df[["Date","Day","Month factor","Day factor","Combined",
                                "Suggested (€)","Your price (€)","Δ per night"]].copy()
        display_df["Δ per night"] = display_df["Δ per night"].apply(lambda v: f"+€{v}" if v>=0 else f"-€{abs(v)}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Annual projection
    st.markdown("<br>", unsafe_allow_html=True)
    current_occ    = actual_occupancy if actual_occupancy else cluster_occ.get(cluster, 100)
    current_annual = current_price * current_occ
    pot_annual     = price_pred * target_occ
    annual_uplift  = pot_annual - current_annual

    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f"""
        <div style="background:white;border-radius:20px;padding:28px 32px;
                    box-shadow:0 1px 8px rgba(0,0,0,0.06);border:1px solid #EEEEEE;">
            <p class="rev-title" style="color:#AAAAAA;">Current annual revenue</p>
            <p style="font-size:42px;font-weight:800;color:#222;margin:0 0 4px 0;">€{current_annual:,.0f}</p>
            <p style="font-size:14px;color:#717171;margin:0;">{current_occ} nights × €{current_price:.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="rev-card">
            <p class="rev-title">Potential annual revenue</p>
            <p class="rev-amount">€{pot_annual:,.0f}</p>
            <p class="rev-sub">{target_occ} nights × €{price_pred:.0f} &nbsp;·&nbsp;
               <strong>+€{annual_uplift:,.0f} uplift ({annual_uplift/current_annual*100:+.1f}%)</strong></p>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        f"Potential revenue uses the RF occupancy model: at a fair-market price of €{price_pred:.0f}/night, "
        f"listings with your profile are predicted to achieve **{target_occ} nights/year** "
        f"(your listing currently books {current_occ} nights/year)."
    )

    with st.expander("How is the price calculated?"):
        st.markdown(
            f"**Base price** (Random Forest model): **€{price_pred:.0f}**  \n"
            "Two SARIMA-derived multipliers are applied:\n"
            "- **Month factor** — seasonal demand (September +33%, February −15%)\n"
            "- **Day-of-week factor** — weekly patterns (Saturday +2.6%, Tuesday −1.4%)\n\n"
            f"Example: September Saturday → **€{price_pred*month_mult.loc[9,'multiplier']*dow_mult.loc[5,'multiplier']:.0f}** "
            f"(+{(month_mult.loc[9,'multiplier']*dow_mult.loc[5,'multiplier']-1)*100:.1f}% above base)"
        )
