"""
Core logic for the Traffic Flow Prediction app (no Streamlit code in here).

Pipeline follows the project methodology:
  - cleaning + feature engineering
  - chronological 80/20 split
  - preprocessing fitted on the training data only
  - Linear Regression (traffic volume) + Decision Tree Classifier (Low/Moderate/High)
  - thresholds for the traffic level come from the training data only
"""
import glob
import io
import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import requests

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, classification_report, confusion_matrix)

UCI_URL = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"

# Location of the dataset (I-94 near Minneapolis - St. Paul) and its local time zone
LAT, LON = 44.95, -93.10
TZ = "America/Chicago"

LEVELS = ["Low", "Moderate", "High"]
FEATURES_NUM = ["temp", "rain_1h", "snow_1h", "clouds_all",
                "hour", "month", "day_of_week", "is_weekend", "is_holiday",
                "hour_sin", "hour_cos", "month_sin", "month_cos"]
FEATURES_CAT = ["holiday", "weather_main", "weather_description"]


# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------
def find_local_dataset(folders=("data", ".")):
    for folder in folders:
        hits = sorted(glob.glob(os.path.join(folder, "*raffic*.csv*")))
        if hits:
            return hits[0]
    return None


def download_dataset(dest="data"):
    raw = urllib.request.urlopen(UCI_URL, timeout=60).read()
    zipfile.ZipFile(io.BytesIO(raw)).extractall(dest)
    return find_local_dataset((dest,))


# --------------------------------------------------------------------------
# Cleaning + feature engineering
# --------------------------------------------------------------------------
def time_features(times):
    t = pd.Series(pd.to_datetime(times)).reset_index(drop=True)
    return pd.DataFrame({
        "hour": t.dt.hour,
        "month": t.dt.month,
        "day_of_week": t.dt.dayofweek,
        "is_weekend": (t.dt.dayofweek >= 5).astype(int),
        "hour_sin": np.sin(2 * np.pi * t.dt.hour / 24),
        "hour_cos": np.cos(2 * np.pi * t.dt.hour / 24),
        "month_sin": np.sin(2 * np.pi * t.dt.month / 12),
        "month_cos": np.cos(2 * np.pi * t.dt.month / 12),
    })


def prepare(raw):
    """Clean the raw UCI dataframe and add the engineered features."""
    d = raw.copy()
    d["date_time"] = pd.to_datetime(d["date_time"], errors="coerce")
    d = d.dropna(subset=["date_time", "traffic_volume"])
    d = d.drop_duplicates(subset="date_time", keep="first")
    d = d.sort_values("date_time").reset_index(drop=True)

    d.loc[d["temp"] <= 0, "temp"] = np.nan          # invalid 0 K readings
    d.loc[d["rain_1h"] > 500, "rain_1h"] = np.nan   # impossible hourly rainfall
    d["holiday"] = d["holiday"].replace("None", "No Holiday").fillna("No Holiday")
    for c in ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = pd.concat([d, time_features(d["date_time"])], axis=1)
    d["is_holiday"] = (d["holiday"] != "No Holiday").astype(int)
    return d


def build_features(times, temp_c, rain, snow, clouds, main, desc, holiday="No Holiday"):
    """Build model-ready rows. Scalars are broadcast to all timestamps."""
    t = pd.Series(pd.to_datetime(times)).reset_index(drop=True)
    n = len(t)

    def col(x):
        return pd.Series(np.broadcast_to(np.asarray(x), (n,)).copy())

    X = time_features(t)
    X["temp"] = col(temp_c).astype(float) + 273.15      # dataset uses Kelvin
    X["rain_1h"] = col(rain).astype(float)
    X["snow_1h"] = col(snow).astype(float)
    X["clouds_all"] = col(clouds).astype(float)
    X["holiday"] = col(holiday)
    X["is_holiday"] = (X["holiday"] != "No Holiday").astype(int)
    X["weather_main"] = col(main)
    X["weather_description"] = col(desc)
    return X[FEATURES_NUM + FEATURES_CAT]


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_models(d):
    """Train both models on the earlier 80% of the data; evaluate on the later 20%."""
    X = d[FEATURES_NUM + FEATURES_CAT]
    y = d["traffic_volume"]
    split = int(len(d) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    def make_preprocessor():
        return ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), FEATURES_NUM),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             FEATURES_CAT),
        ])

    # Linear Regression (traffic volume)
    lr = Pipeline([("prep", make_preprocessor()), ("model", LinearRegression())])
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
    }

    # Traffic levels: thresholds from the TRAINING data only
    t1, t2 = [float(v) for v in y_train.quantile([1 / 3, 2 / 3])]

    def to_level(v):
        return np.where(v < t1, "Low", np.where(v < t2, "Moderate", "High"))

    yc_train, yc_test = to_level(y_train.values), to_level(y_test.values)

    # Decision Tree Classifier
    dt = Pipeline([("prep", make_preprocessor()),
                   ("model", DecisionTreeClassifier(max_depth=10, min_samples_leaf=5,
                                                    random_state=42))])
    dt.fit(X_train, yc_train)
    yc_pred = dt.predict(X_test)

    report = classification_report(yc_test, yc_pred, labels=LEVELS, output_dict=True,
                                   zero_division=0)
    cm = confusion_matrix(yc_test, yc_pred, labels=LEVELS)
    names = dt.named_steps["prep"].get_feature_names_out()
    importance = pd.Series(dt.named_steps["model"].feature_importances_, index=names)
    importance = importance.sort_values(ascending=False)

    # Lookups used by the app inputs
    common_desc = d.groupby("weather_main")["weather_description"] \
                   .agg(lambda s: s.value_counts().index[0]).to_dict()
    descs_by_main = {m: g["weather_description"].value_counts().index.tolist()
                     for m, g in d.groupby("weather_main")}

    # "Typical weather" for each (month, hour), from training data only
    tr = d.iloc[:split]
    typical = tr.groupby(["month", "hour"]).agg(
        temp_c=("temp", "median"), rain=("rain_1h", "mean"), snow=("snow_1h", "mean"),
        clouds=("clouds_all", "median"),
        main=("weather_main", lambda s: s.value_counts().index[0]))
    typical["temp_c"] = typical["temp_c"] - 273.15

    return {
        "lr": lr, "dt": dt, "t1": t1, "t2": t2,
        "metrics": metrics,
        "accuracy": float(accuracy_score(yc_test, yc_pred)),
        "report": report, "cm": cm,
        "importance": importance,
        "test_times": d["date_time"].iloc[split:].reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "y_pred": pd.Series(y_pred),
        "n_rows": len(d), "n_train": split, "n_test": len(d) - split,
        "train_range": (d["date_time"].iloc[0], d["date_time"].iloc[split - 1]),
        "test_range": (d["date_time"].iloc[split], d["date_time"].iloc[-1]),
        "common_desc": common_desc,
        "descs_by_main": descs_by_main,
        "mains": sorted(common_desc),
        "holidays": sorted(h for h in d["holiday"].unique() if h != "No Holiday"),
        "typical": typical,
    }


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def predict_frame(bundle, times, temp_c, rain, snow, clouds, main, desc=None,
                  holiday="No Holiday"):
    """Predict volume and level for each timestamp. Returns a tidy dataframe."""
    t = pd.Series(pd.to_datetime(times)).reset_index(drop=True)
    n = len(t)
    main_s = pd.Series(np.broadcast_to(np.asarray(main), (n,)).copy())
    main_s = main_s.where(main_s.isin(bundle["mains"]), "Clouds")
    if desc is None:
        desc = main_s.map(bundle["common_desc"])
    X = build_features(t, temp_c, rain, snow, clouds, main_s, desc, holiday)
    volume = np.maximum(0, bundle["lr"].predict(X)).round(0).astype(int)
    level = bundle["dt"].predict(X)
    return pd.DataFrame({
        "time": t, "date": t.dt.date,
        "temp_c": X["temp"] - 273.15, "rain_mm": X["rain_1h"], "snow_cm": X["snow_1h"],
        "clouds": X["clouds_all"], "weather": main_s,
        "volume": volume, "level": level,
    })


def typical_weather(bundle, times):
    """Typical weather per hour for the same month/hour in the training data."""
    t = pd.Series(pd.to_datetime(times)).reset_index(drop=True)
    typ = bundle["typical"]
    key = pd.MultiIndex.from_arrays([t.dt.month, t.dt.hour])
    w = typ.reindex(key).reset_index(drop=True)
    num = ["temp_c", "rain", "snow", "clouds"]
    w[num] = w[num].fillna(typ[num].median())
    w["main"] = w["main"].fillna("Clouds")
    return w


def daily_summary(fc):
    rows = []
    for day, g in fc.groupby("date"):
        peak = g.loc[g["volume"].idxmax()]
        rows.append({
            "Date": str(day),
            "Day": pd.Timestamp(day).day_name(),
            "Main weather": g["weather"].value_counts().index[0],
            "Avg temp (°C)": round(g["temp_c"].mean(), 1),
            "Rain (mm)": round(g["rain_mm"].sum(), 1),
            "Snow (cm)": round(g["snow_cm"].sum(), 1),
            "Avg volume": int(round(g["volume"].mean())),
            "Peak volume": int(peak["volume"]),
            "Peak hour": peak["time"].strftime("%H:%M"),
            "High-traffic hours": int((g["level"] == "High").sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Weather forecast (Open-Meteo, free, no API key, up to 16 days)
# --------------------------------------------------------------------------
def tomorrow_local():
    return pd.Timestamp.now(tz=TZ).normalize().tz_localize(None) + pd.Timedelta(days=1)


def fetch_forecast(forecast_days):
    """Hourly forecast for the dataset's location. `forecast_days` must be 1..16."""
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": LAT, "longitude": LON,
        "hourly": "temperature_2m,precipitation,snowfall,cloud_cover,weather_code",
        "forecast_days": int(forecast_days), "timezone": TZ}, timeout=30)
    r.raise_for_status()
    w = pd.DataFrame(r.json()["hourly"])
    w["time"] = pd.to_datetime(w["time"])
    return w


def wmo_to_main(code):
    """Convert an Open-Meteo (WMO) weather code to the dataset's weather_main."""
    if code == 0:
        return "Clear"
    if code in (1, 2, 3):
        return "Clouds"
    if code in (45, 48):
        return "Fog"
    if code in (51, 53, 55, 56, 57):
        return "Drizzle"
    if code in (61, 63, 65, 66, 67, 80, 81, 82):
        return "Rain"
    if code in (71, 73, 75, 77, 85, 86):
        return "Snow"
    if code in (95, 96, 99):
        return "Thunderstorm"
    return "Clouds"


def forecast_to_predictions(bundle, w, days):
    """Predict traffic for each day from tomorrow to `days` days ahead."""
    start = tomorrow_local()
    end = start + pd.Timedelta(days=days)
    w = w[(w["time"] >= start) & (w["time"] < end)].reset_index(drop=True)
    if w.empty:
        raise ValueError("The forecast contains no hours for the requested dates.")
    main = w["weather_code"].apply(wmo_to_main)
    return predict_frame(bundle, w["time"], w["temperature_2m"], w["precipitation"],
                         w["snowfall"], w["cloud_cover"], main)


def typical_to_predictions(bundle, days):
    """Predict using typical (historical) weather for that month and hour."""
    times = pd.date_range(tomorrow_local(), periods=days * 24, freq="h")
    w = typical_weather(bundle, times)
    return predict_frame(bundle, times, w["temp_c"], w["rain"], w["snow"],
                         w["clouds"], w["main"])