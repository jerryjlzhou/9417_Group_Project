import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

os.makedirs("baseline_classification_plots", exist_ok=True)

# Loading the preprocessed air quality dataset after feature engineering
df = pd.read_csv("AirQualityUCI_feature_engineering.csv", parse_dates=["DateTime"])
df = df.sort_values("DateTime").reset_index(drop=True)
df["Year"] = df["DateTime"].dt.year

df_2004 = df[df["Year"] == 2004].copy()
df_2005 = df[df["Year"] == 2005].copy()

co_vals_2004 = df_2004["CO(GT)"].dropna()
low_thr = co_vals_2004.quantile(1 / 3)
high_thr = co_vals_2004.quantile(2 / 3)

# Categorizing CO(GT) levels
def co_to_bin(v):
    if pd.isna(v):
        return np.nan
    if v < low_thr:
        return "low"
    elif v < high_thr:
        return "mid"
    else:
        return "high"

df_2004["CO_bin_t"] = df_2004["CO(GT)"].apply(co_to_bin)
df_2005["CO_bin_t"] = df_2005["CO(GT)"].apply(co_to_bin)

labels = ["low", "mid", "high"]
horizons = [1, 6, 12, 24]
baseline_acc = {}

for h in horizons:
    print(f"\nNaive Baseline: t → t+{h} prediction")

    future_col = f"CO(GT)_t+{h}"
    if future_col not in df.columns:
        print(f"Missing column: {future_col}")
        continue

    df_2004[f"CO_bin_t+h"] = df_2004[future_col].apply(co_to_bin)
    df_2005[f"CO_bin_t+h"] = df_2005[future_col].apply(co_to_bin)

    ts = df_2005.dropna(subset=["CO_bin_t", "CO_bin_t+h"]).copy()

    y_true = ts["CO_bin_t+h"]
    y_pred = ts["CO_bin_t"]   # baseline prediction

    acc = accuracy_score(y_true, y_pred)
    baseline_acc[h] = acc

    print("Accuracy:", round(acc, 3))
    print(classification_report(y_true, y_pred, labels=labels))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Baseline Confusion Matrix (t+{h})")
    plt.xticks(range(3), labels)
    plt.yticks(range(3), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    
    cm_path = f"baseline_classification_plots/confusion_matrix_t+{h}.png"
    plt.savefig(cm_path, dpi=150)
    plt.show()
    print("Saved:", cm_path)

# Accuracy vs horizon for the baseline
if baseline_acc:
    plt.figure(figsize=(6, 4))
    hs = sorted(baseline_acc.keys())
    plt.plot(hs, [baseline_acc[h] for h in hs], marker="o")
    plt.xticks(hs, [f"t+{h}" for h in hs])
    plt.ylim(0, 1)
    plt.xlabel("Forecast horizon")
    plt.ylabel("Accuracy")
    plt.title("Baseline accuracy vs horizon")
    plt.grid(alpha=0.3)
    plt.tight_layout()
   
    acc_plot_path = "baseline_classification_plots/baseline_accuracy_vs_horizon.png"
    plt.savefig(acc_plot_path, dpi=150)
    plt.show()
    print("Saved:", acc_plot_path)

# Saving baseline accuracies to CSV so the XGBoost classification script can read them
baseline_df = pd.DataFrame(
    {"horizon": list(baseline_acc.keys()),
     "accuracy": list(baseline_acc.values())}
)
baseline_df.to_csv("baseline_classification_accuracy.csv", index=False)
print("\nSaved baseline accuracies to baseline_classification_accuracy.csv")