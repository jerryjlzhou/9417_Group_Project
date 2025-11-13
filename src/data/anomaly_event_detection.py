import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

os.makedirs("plots", exist_ok=True)

# 1. Loading the preprocessed air quality dataset after EDA + data cleaning

df = pd.read_csv("AirQualityUCI_processed.csv", parse_dates=["DateTime"])
df = df.sort_values("DateTime").reset_index(drop=True)
df["datetime"] = df["DateTime"]  # shorter alias, easier to type

# Main pollutant columns to check for anomalies 
pollutant_cols = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
print("Analysing residual anomalies for:", pollutant_cols)

# 2. Residual-based anomalies (per pollutant)

window = 24   # 24 hours / one day window for rolling median 
resid_flag_cols = [] # columns to hold residual anomaly flags
resid_thresholds = {}   

for pollutant in pollutant_cols:
    base = pollutant.split("(")[0].strip().replace(" ", "_")

    roll_col = f"{base}_roll_med"
    resid_col = f"{base}_residual"
    flag_col = f"flag_anom_{base}"

    # rolling median to represent the baseline / typical value
    df[roll_col] = (
        df[pollutant]
        .rolling(window=window, center=True, min_periods=window // 2)
        .median()
    )

    
    df[resid_col] = (df[pollutant] - df[roll_col]).abs()

    # Setting threshold at 97.5 percentile of residuals.
    # The reasoning for keeping the threshold value that high is so that only the most extreme deviations are flagged as anomalies
    # which are more likely to represent true anomalous events rather than normal fluctuations in air quality data.
    thr = df[resid_col].quantile(0.975)
    resid_thresholds[pollutant] = thr

    df[flag_col] = (df[resid_col] > thr).astype(int)
    resid_flag_cols.append(flag_col)

    n_anom = int(df[flag_col].sum())
    rate = df[flag_col].mean()

    print("\n---", pollutant, "---")
    print("  residual threshold (97.5%):", round(thr, 4))
    print("  # anomalies:", n_anom, "| rate:", round(rate, 4))

    plt.figure(figsize=(14, 3.5))
    plt.plot(df["datetime"], df[pollutant], label=pollutant, linewidth=0.8)
    plt.scatter(
        df.loc[df[flag_col] == 1, "datetime"],
        df.loc[df[flag_col] == 1, pollutant],
        s=15,
        color="red",
        label="residual anomaly",
    )
    plt.xlabel("Time")
    plt.ylabel(pollutant)
    plt.title(f"Residual-based anomalies for {pollutant}")
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_path = f"plots/residual_anomalies_{base}.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print("  saved plot to:", out_path)

# Flags are updated to indicate if any pollutant had a residual anomaly at that time
df["flag_anom_any_residual"] = (df[resid_flag_cols].sum(axis=1) > 0).astype(int)
print(
    "\nAny-pollutant residual anomaly rate:",
    round(df["flag_anom_any_residual"].mean(), 4),
)

# Isolation Forest (multivariate anomalies) (unsupervised method)

if_features = [
    "CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)",
    "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
    "PT08.S4(NO2)", "PT08.S5(O3)",
    "T", "RH", "AH",
]

df_if = df.dropna(subset=if_features).copy()
print("Rows used for Isolation Forest:", len(df_if))

if_model = IsolationForest(
    contamination=0.02,   # rouhgly 2% anomalies expected
    random_state=42,
)
raw_if_labels = if_model.fit_predict(df_if[if_features])

df_if["iso_label_raw"] = raw_if_labels
df_if["flag_anom_iforest"] = df_if["iso_label_raw"].map({1: 0, -1: 1})

print(
    "\nIsolation Forest anomaly rate:",
    round(df_if["flag_anom_iforest"].mean(), 4),
)

# Plotting IF anomalies against CO(GT) as an example (can be plotted for other pollutants similarly)
plt.figure(figsize=(14, 3.5))
plt.plot(df_if["datetime"], df_if["CO(GT)"], label="CO(GT)", linewidth=0.8)
plt.scatter(
    df_if.loc[df_if["flag_anom_iforest"] == 1, "datetime"],
    df_if.loc[df_if["flag_anom_iforest"] == 1, "CO(GT)"],
    s=15,
    color="orange",
    label="IF anomaly",
)
plt.xlabel("Time")
plt.ylabel("CO(GT)")
plt.title("Isolation Forest anomalies (shown on CO(GT))")
plt.legend(loc="upper right")
plt.tight_layout()

if_fig_path = "plots/if_anomalies_CO.png"
plt.savefig(if_fig_path, dpi=150)
plt.show()
print("Saved IF plot to:", if_fig_path)


combined = pd.merge(
    df[["datetime"] + resid_flag_cols + ["flag_anom_any_residual"]],
    df_if[["datetime", "flag_anom_iforest"]],
    on="datetime",
    how="inner",
)

combined["flag_anom_combined"] = (
    (combined["flag_anom_iforest"] == 1)
    | (combined["flag_anom_any_residual"] == 1)
).astype(int)

print(
    "\nCombined (residual OR IF) anomaly rate:",
    round(combined["flag_anom_combined"].mean(), 4),
)

# Analysis of anomaly patterns with respect to meteorological and temporal factors
data_cols = ["T", "RH", "AH", "Hour", "Day", "Month"]

event_data = combined.merge(
    df[["datetime"] + data_cols],
    on="datetime",
    how="left",
)

event_data["is_weekend"] = (event_data["Day"] >= 5).astype(int)

print("\nMean T/RH/AH for normal vs anomalous (combined flag):")
print(event_data.groupby("flag_anom_combined")[["T", "RH", "AH"]].mean())

# anomaly rate by hour of day
hourly_rate = event_data.groupby("Hour")["flag_anom_combined"].mean()
plt.figure(figsize=(8, 3))
hourly_rate.plot(kind="bar")
plt.title("Combined anomaly rate by hour of day")
plt.xlabel("Hour")
plt.ylabel("Anomaly rate")
plt.tight_layout()

hour_fig = "plots/anomaly_rate_by_hour.png"
plt.savefig(hour_fig, dpi=150)
plt.show()
print("Saved hourly rate plot to:", hour_fig)

# anomaly rate by month
monthly_rate = event_data.groupby("Month")["flag_anom_combined"].mean()
plt.figure(figsize=(8, 3))
monthly_rate.plot(kind="bar")
plt.title("Combined anomaly rate by month")
plt.xlabel("Month")
plt.ylabel("Anomaly rate")
plt.tight_layout()

month_fig = "plots/anomaly_rate_by_month.png"
plt.savefig(month_fig, dpi=150)
plt.show()
print("Saved monthly rate plot to:", month_fig)

print("\nWeekend vs weekday anomaly rate:")
print(event_data.groupby("is_weekend")["flag_anom_combined"].mean())

# Dataset with anomaly flags saved for each pollutant

df_with_flags = df.merge(
    combined[["datetime", "flag_anom_iforest", "flag_anom_combined"]],
    on="datetime",
    how="left",
)

df_with_flags[["flag_anom_iforest", "flag_anom_combined"]] = (
    df_with_flags[["flag_anom_iforest", "flag_anom_combined"]]
    .fillna(0)
    .astype(int)
)

out_csv = "air_quality_with_anomaly_flags.csv"
df_with_flags.to_csv(out_csv, index=False)
print("\nSaved master dataset with anomaly flags to:", out_csv)

event_csv = "anomaly_event_data.csv"
event_data.to_csv(event_csv, index=False)
print("Saved event-level anomaly summary to:", event_csv)
