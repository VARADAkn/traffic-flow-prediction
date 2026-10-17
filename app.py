"""
Traffic Flow Prediction - Streamlit web app
Linear Regression (traffic volume) + Decision Tree Classifier (Low / Moderate / High)
Dataset: Metro Interstate Traffic Volume (UCI Machine Learning Repository)

Run with:  streamlit run app.py
"""
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

import traffic_core as tc

st.set_page_config(page_title="Traffic Flow Prediction", page_icon="🚦", layout="wide")

COLORS = {"Low": "#2e9e5b", "Moderate": "#e8a317", "High": "#d64545"}
REQUIRED_COLUMNS = ["date_time", "traffic_volume", "temp", "rain_1h", "snow_1h",
                    "clouds_all", "weather_main", "weather_description", "holiday"]


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def load_dataset():
    path = tc.find_local_dataset()
    if path is None:
        try:
            path = tc.download_dataset()
        except Exception:
            return None
    return pd.read_csv(path) if path else None


@st.cache_resource(show_spinner="Training both models on the dataset (first run only)...")
def get_bundle(_raw, key):
    return tc.train_models(tc.prepare(_raw))


@st.cache_data(ttl=1800, show_spinner=False)
def cached_forecast(forecast_days):
    return tc.fetch_forecast(forecast_days)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def level_badge(level):
    return (f"<div style='padding:14px 16px;border-radius:10px;background:{COLORS[level]};"
            f"color:white;text-align:center'>"
            f"<div style='font-size:0.8rem;opacity:0.9'>Predicted traffic level</div>"
            f"<div style='font-size:1.9rem;font-weight:700'>{level}</div></div>")


def traffic_chart(df, height=320, marker_time=None):
    base = alt.Chart(df).encode(
        x=alt.X("time:T", title="Time"),
        y=alt.Y("volume:Q", title="Vehicles / hour"))
    line = base.mark_line(color="#9aa0a6")
    points = base.mark_circle(size=70).encode(
        color=alt.Color("level:N", title="Traffic level",
                        scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
                        legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip("time:T", title="Time", format="%a %d %b, %H:%M"),
                 alt.Tooltip("volume:Q", title="Volume"),
                 alt.Tooltip("level:N", title="Level"),
                 alt.Tooltip("weather:N", title="Weather"),
                 alt.Tooltip("temp_c:Q", title="Temp °C", format=".1f")])
    chart = line + points
    if marker_time is not None:
        rule = alt.Chart(pd.DataFrame({"time": [marker_time]})).mark_rule(
            color="#555", strokeDash=[4, 4]).encode(x="time:T")
        chart = chart + rule
    return chart.properties(height=height, width="container")


# --------------------------------------------------------------------------
# Load data + models
# --------------------------------------------------------------------------
st.title("🚦 Traffic Flow Prediction")
st.caption("Linear Regression predicts the traffic volume · Decision Tree Classification "
           "predicts the traffic level (Low / Moderate / High) · "
           "Trained on the Metro Interstate Traffic Volume dataset (I-94, Minneapolis-St. Paul)")

raw = load_dataset()
if raw is None:
    st.warning("The dataset could not be downloaded automatically. Download "
               "**Metro_Interstate_Traffic_Volume.csv.gz** (or .csv) from the "
               "[UCI repository](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) "
               "and upload it below, or place it in a `data/` folder next to `app.py`.")
    uploaded = st.file_uploader("Upload the dataset", type=["csv", "gz"])
    if uploaded is None:
        st.stop()
    raw = pd.read_csv(uploaded, compression="gzip" if uploaded.name.endswith(".gz") else None)

missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
if missing:
    st.error(f"The dataset is missing required columns: {', '.join(missing)}")
    st.stop()

bundle = get_bundle(raw, (len(raw), str(raw["date_time"].iloc[0]), str(raw["date_time"].iloc[-1])))

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Dataset & models")
    st.write(f"**Rows after cleaning:** {bundle['n_rows']:,}")
    st.write(f"**Training:** {bundle['train_range'][0]:%b %Y} to {bundle['train_range'][1]:%b %Y} "
             f"({bundle['n_train']:,} rows)")
    st.write(f"**Testing:** {bundle['test_range'][0]:%b %Y} to {bundle['test_range'][1]:%b %Y} "
             f"({bundle['n_test']:,} rows)")
    st.markdown("**Traffic levels** (from training data)")
    st.markdown(f"- 🟢 **Low:** under {bundle['t1']:,.0f} vehicles/h\n"
                f"- 🟠 **Moderate:** {bundle['t1']:,.0f} to {bundle['t2']:,.0f}\n"
                f"- 🔴 **High:** {bundle['t2']:,.0f} or more")
    st.caption("Predictions come from patterns in 2012-2018 data for one road segment.")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predict a moment", "📅 Forecast ahead",
                                  "📊 Model performance", "ℹ️ About"])

# --------------------------------------------------------------------------
# Tab 1: predict one moment from user input
# --------------------------------------------------------------------------
with tab1:
    left, right = st.columns([1, 1.4], gap="large")

    with left:
        st.subheader("Enter the conditions")
        date_in = st.date_input("Date", value=tc.tomorrow_local().date())
        hour_in = st.slider("Hour of day (0-23)", 0, 23, 8)
        temp_c = st.slider("Temperature (°C)", -35.0, 40.0, 10.0, 0.5)
        mains = bundle["mains"]
        main = st.selectbox("Weather", mains, index=mains.index("Clouds") if "Clouds" in mains else 0)
        desc = st.selectbox("Weather detail", bundle["descs_by_main"][main])
        c1, c2 = st.columns(2)
        rain = c1.number_input("Rain, last hour (mm)", 0.0, 100.0, 0.0, 0.5)
        snow = c2.number_input("Snow, last hour (mm)", 0.0, 100.0, 0.0, 0.5)
        clouds = st.slider("Cloud cover (%)", 0, 100, 40)
        holiday = st.selectbox("Holiday", ["No Holiday"] + bundle["holidays"])

    when = pd.Timestamp(date_in) + pd.Timedelta(hours=hour_in)
    res = tc.predict_frame(bundle, [when], temp_c, rain, snow, clouds, main, desc, holiday).iloc[0]
    day_times = pd.date_range(pd.Timestamp(date_in), periods=24, freq="h")
    profile = tc.predict_frame(bundle, day_times, temp_c, rain, snow, clouds, main, desc, holiday)

    with right:
        st.subheader("Prediction")
        m1, m2 = st.columns(2)
        m1.metric(f"Traffic volume, {when:%a %d %b %Y, %H:%M}", f"{int(res['volume']):,} vehicles/h")
        m2.markdown(level_badge(res["level"]), unsafe_allow_html=True)
        st.caption("Volume: Linear Regression · Level: Decision Tree Classifier")
        st.markdown("**The same day, hour by hour, with these weather conditions**")
        st.altair_chart(traffic_chart(profile, 300, marker_time=when))

# --------------------------------------------------------------------------
# Tab 2: forecast the coming days
# --------------------------------------------------------------------------
with tab2:
    st.subheader("Traffic forecast for the coming days")
    st.write("The models need weather as input, so for future dates the weather comes from a free "
             "forecast service (Open-Meteo) for the dataset's road location, or from typical "
             "weather for that time of year.")
    c1, c2 = st.columns(2)
    source = c1.radio("Weather source",
                      ["Live weather forecast (up to 15 days)",
                       "Typical weather for that time of year (up to 30 days)"])
    live = source.startswith("Live")
    days = c2.slider("Days ahead", 1, 15 if live else 30, 10)

    if st.button("Predict traffic", type="primary"):
        try:
            with st.spinner("Fetching weather and predicting..."):
                if live:
                    fc_new = tc.forecast_to_predictions(bundle, cached_forecast(days + 1), days)
                else:
                    fc_new = tc.typical_to_predictions(bundle, days)
            st.session_state["fc"] = fc_new
            st.session_state["fc_source"] = "live weather forecast" if live else "typical weather"
        except Exception as exc:
            st.error(f"Could not get the weather forecast ({exc}). "
                     "Try again, or switch to 'Typical weather'.")

    fc = st.session_state.get("fc")
    if fc is not None:
        summary = tc.daily_summary(fc)
        busiest = summary.loc[summary["Avg volume"].idxmax()]
        quietest = summary.loc[summary["Avg volume"].idxmin()]
        k1, k2, k3 = st.columns(3)
        k1.metric("Days predicted", len(summary))
        k2.metric("Busiest day", f"{busiest['Day']} {busiest['Date'][5:]}",
                  f"{busiest['Avg volume']:,} avg vehicles/h", delta_color="off")
        k3.metric("Quietest day", f"{quietest['Day']} {quietest['Date'][5:]}",
                  f"{quietest['Avg volume']:,} avg vehicles/h", delta_color="off")

        st.markdown(f"**Daily summary** (weather source: {st.session_state.get('fc_source')})")
        st.dataframe(summary, hide_index=True)

        st.markdown("**Hourly prediction for the whole period**")
        st.altair_chart(traffic_chart(fc, 320))

        st.markdown("**Look at one day**")
        options = [str(d) for d in sorted(fc["date"].unique())]
        chosen = st.selectbox("Day", options)
        one_day = fc[fc["date"].astype(str) == chosen]
        st.altair_chart(traffic_chart(one_day, 280))

        if st.session_state.get("fc_source") == "live weather forecast":
            st.info("Forecasts are less certain the further ahead they look (especially rain and snow "
                    "after about 5 days), and public holidays are not taken into account.")
        else:
            st.info("This uses typical weather for each month and hour, so it shows the usual "
                    "traffic pattern rather than a weather-specific forecast. "
                    "Public holidays are not taken into account.")

        st.download_button("Download hourly predictions (CSV)",
                           fc.drop(columns=["date"]).to_csv(index=False),
                           file_name="traffic_predictions.csv", mime="text/csv")

# --------------------------------------------------------------------------
# Tab 3: model performance on the chronological test set
# --------------------------------------------------------------------------
with tab3:
    st.subheader("How good are the models?")
    st.caption(f"Evaluated on the later 20% of the data "
               f"({bundle['test_range'][0]:%b %Y} to {bundle['test_range'][1]:%b %Y}) "
               "that the models never saw during training.")

    st.markdown("### Linear Regression (traffic volume)")
    mt = bundle["metrics"]
    a, b, c = st.columns(3)
    a.metric("MAE", f"{mt['MAE']:.1f}", help="Average absolute error in vehicles/hour (lower is better)")
    b.metric("RMSE", f"{mt['RMSE']:.1f}", help="Like MAE but penalises large errors more (lower is better)")
    c.metric("R²", f"{mt['R2']:.3f}", help="Share of variation explained (closer to 1 is better)")

    n = 24 * 7
    comp = pd.DataFrame({"Actual": bundle["y_test"].iloc[:n].values,
                         "Predicted": bundle["y_pred"].iloc[:n].values},
                        index=bundle["test_times"].iloc[:n])
    st.markdown("**Actual vs predicted, first week of the test period**")
    st.line_chart(comp)

    st.markdown("### Decision Tree Classification (traffic level)")
    cc1, cc2 = st.columns([1, 1])
    with cc1:
        st.metric("Accuracy", f"{bundle['accuracy']:.3f}")
        rep = pd.DataFrame(bundle["report"]).T.loc[tc.LEVELS + ["weighted avg"],
                                                   ["precision", "recall", "f1-score", "support"]]
        rep["support"] = rep["support"].astype(int)
        st.dataframe(rep.round(3))
    with cc2:
        fig, ax = plt.subplots(figsize=(4.6, 4))
        ConfusionMatrixDisplay(bundle["cm"], display_labels=tc.LEVELS).plot(
            cmap="Blues", ax=ax, values_format="d", colorbar=False)
        ax.set_title("Confusion matrix")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Most important features (Decision Tree)**")
    st.bar_chart(bundle["importance"].head(10))

# --------------------------------------------------------------------------
# Tab 4: about
# --------------------------------------------------------------------------
with tab4:
    st.subheader("About this project")
    st.markdown("""
**Traffic Flow Prediction Using Linear Regression and Decision Tree Classification**

Traffic congestion causes longer travel times, higher fuel use and more air pollution. Predicting
traffic helps with traffic control, city planning and better use of existing roads.

**Method**
1. Dataset: Metro Interstate Traffic Volume (UCI), hourly westbound I-94 traffic with weather and date-time.
2. Cleaning: duplicates removed, invalid 0 K temperatures and an impossible rainfall value treated as missing,
   the holiday value "None" replaced by "No Holiday".
3. Feature engineering: hour, month, day of week, weekend and holiday flags, plus sine/cosine encoding of hour and month.
4. Chronological split: earlier 80% for training, later 20% for testing.
5. Preprocessing learned from the training data only: median imputation (numeric), most-frequent imputation
   and one-hot encoding (categorical).
6. **Linear Regression** predicts the traffic volume; evaluated with MAE, RMSE and R².
7. **Decision Tree Classifier** predicts Low / Moderate / High, with thresholds taken from the training data;
   evaluated with accuracy, precision, recall, F1-score and a confusion matrix.

**Limitations**
- The data covers one road segment from 2012 to 2018, so predictions do not apply to other roads or to
  today's traffic if patterns have changed.
- Predictions for future dates depend on the weather forecast, which becomes less accurate further ahead.
- Linear Regression cannot follow the two daily rush-hour peaks well; the Decision Tree level is often more reliable.
- No lag features (previous-hour traffic) and no special events or accidents are included.
""")