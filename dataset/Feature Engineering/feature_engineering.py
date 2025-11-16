'''# COMP9417 Group Project — Feature Engineering

 Author: Mohammed Amaan Sarwar 

 What this script does:
 1) Loads the pre-processed AirQualityUCI dataset (expects a merged DateTime)

 2) Replaces sentinel -200 values with NaN 

 3) Sorts chronologically and builds:
    - Calendar & cyclical encodings (hour/day-of-week/month)
    - Temporal lags: 1h, 6h, 12h, 24h (for key pollutants + meteo + sensors)
    - Rolling stats (6h, 24h mean/std)
    - Interaction & residual features (ratios; sensor minus ground-truth)

 4) Creates forecasting targets for horizons H in {1,6,12,24}:
    - Regression targets: <POLLUTANT>_t+H  (shift(-H))
    - CO classification bins: CO_bin_t+H in {low, mid, high}

 5) Optionally drops rows with any NaNs created by lags/rolls/targets

 6) Saves a clean features.CSV for modelling.

 Usage:
   python -m src.feature_engineering \
       --input AirQualityUCI_processed.csv \
       --output AirQualityUCI_features.csv \
       --drop-na-final

 Note:
 - We avoid any leakage by using only the past data to build features
   (lags/rollings) and shifting targets forward for horizons.

 - If your processed file does not have "DateTime", the script will
   try to build it from "Date" + "Time" columns.

'''

from __future__ import annotations

import argparse
from typing import List

import numpy as np
import pandas as pd


POLLUTANTS = [
    "CO(GT)",
    "NMHC(GT)",
    "C6H6(GT)",
    "NOx(GT)",
    "NO2(GT)"
]

SENSORS = [
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)"
]

METEOROLOGICAL_VAR = ["TEMP", "REL_HUMIDITY", "ABS_HUMIDITY"]

HORIZONS = [1, 6, 12, 24]  


CO_LOW = 1.5
CO_HIGH = 2.5


def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "DateTime" in df.columns:
        dt = pd.to_datetime(df["DateTime"], errors="coerce", infer_datetime_format=True)
        df = df.copy()
        df["DateTime"] = dt
        return df

    if {"Date", "Time"}.issubset(set(df.columns)):
        dt = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),
                            errors="coerce", infer_datetime_format=True)
        df = df.copy()
        df["DateTime"] = dt
        return df

    raise ValueError("No 'DateTime' column found and couldn't build from 'Date' + 'Time'.")


def replace_sentinel_neg200(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].replace(-200, np.nan)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["DateTime"]

    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek  # Monday=0
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    return df


def safe_cols(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    present = [c for c in candidates if c in df.columns]
    return present


def add_lag_features(df: pd.DataFrame, base_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in base_cols:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag6"] = df[col].shift(6)
        df[f"{col}_lag12"] = df[col].shift(12)
        df[f"{col}_lag24"] = df[col].shift(24)
    return df


def add_rolling_features(df: pd.DataFrame, base_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in base_cols:
        df[f"{col}_roll_mean_6h"] = df[col].rolling(window=6, min_periods=3).mean()
        df[f"{col}_roll_std_6h"] = df[col].rolling(window=6, min_periods=3).std()

        df[f"{col}_roll_mean_24h"] = df[col].rolling(window=24, min_periods=12).mean()
        df[f"{col}_roll_std_24h"] = df[col].rolling(window=24, min_periods=12).std()
    return df


def add_interaction_and_residuals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    eps = 1e-6
    if {"NOx(GT)", "NO2(GT)"}.issubset(df.columns):
        df["NOx_NO2_ratio"] = df["NOx(GT)"] / (df["NO2(GT)"] + eps)
    if {"C6H6(GT)", "CO(GT)"}.issubset(df.columns):
        df["C6H6_CO_ratio"] = df["C6H6(GT)"] / (df["CO(GT)"] + eps)

   
    pairs = [
        ("PT08.S1(CO)", "CO(GT)"),
        ("PT08.S2(NMHC)", "NMHC(GT)"),
        ("PT08.S3(NOx)", "NOx(GT)"),
        ("PT08.S4(NO2)", "NO2(GT)"),
    ]
    for sensor, truth in pairs:
        if sensor in df.columns and truth in df.columns:
            df[f"{sensor}_minus_{truth}"] = df[sensor] - df[truth]

    return df


def add_future_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in safe_cols(df, POLLUTANTS):
        for h in HORIZONS:
            df[f"{col}_t+{h}"] = df[col].shift(-h)

    if "CO(GT)" in df.columns:
        for h in HORIZONS:
            future_val = df["CO(GT)"].shift(-h)

            bins = pd.cut(
                future_val,
                bins=[-np.inf, CO_LOW, CO_HIGH, np.inf],
                labels=["low", "mid", "high"]
            )
            df[f"CO_bin_t+{h}"] = bins.astype("category")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df = ensure_datetime(df)
    df = replace_sentinel_neg200(df)
    df = df.sort_values("DateTime").reset_index(drop=True)

    df = add_calendar_features(df)

    cols_for_time_ops = safe_cols(df, POLLUTANTS + METEOROLOGICAL_VAR + SENSORS)

    df = add_lag_features(df, cols_for_time_ops)
    df = add_rolling_features(df, safe_cols(df, POLLUTANTS + METEOROLOGICAL_VAR))  # roll mostly on GT + meteo

    df = add_interaction_and_residuals(df)

    df = add_future_targets(df)

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature engineering for AirQuality time series."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to processed CSV (expects a DateTime column or Date+Time)."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV path for engineered features."
    )
    parser.add_argument(
        "--drop-na-final",
        action="store_true",
        help="If set, drops rows with any NaNs created by lags/rolls/targets."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    features = engineer_features(df)

    if args.drop_na_final:
        non_dt_cols = [c for c in features.columns if c != "DateTime"]
        features = features.dropna(subset=non_dt_cols).reset_index(drop=True)

    features.to_csv(args.output, index=False)

    print("Feature Engineering Complete ")
    print(f"Input file:   {args.input}")
    print(f"Output file:  {args.output}")
    print(f"Rows:         {len(features):,}")
    print(f"Columns:      {features.shape[1]}")
    print("Targets created:")
    for h in HORIZONS:
        reg_targets = [f"{p}_t+{h}" for p in POLLUTANTS if f"{p}_t+{h}" in features.columns]
        cls_target = f"CO_bin_t+{h}" if f"CO_bin_t+{h}" in features.columns else "(missing)"
        print(f"  - t+{h}h: regression={len(reg_targets)} cols, classification={cls_target}")


if __name__ == "__main__":
    main()
