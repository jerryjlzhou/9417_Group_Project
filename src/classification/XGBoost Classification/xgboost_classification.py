import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

os.makedirs("xgboost_classification_plots", exist_ok=True)

# Loading the preprocessed air quality dataset after feature engineering
df = pd.read_csv("AirQualityUCI_feature_engineering.csv", parse_dates=["DateTime"])
df = df.sort_values("DateTime").reset_index(drop=True)
df["Year"] = df["DateTime"].dt.year

df_2004 = df[df["Year"] == 2004].copy()
df_2005 = df[df["Year"] == 2005].copy()

if df_2004.empty or df_2005.empty:
    raise ValueError("Data not found.")

# Class distribution for CO(GT) levels
col_gt_vals = df_2004["CO(GT)"].dropna()
low_thr = col_gt_vals.quantile(1 / 3)
high_thr = col_gt_vals.quantile(2 / 3)

def co_categories(v):
    if pd.isna(v):
        return np.nan
    if v < low_thr:
        return "low"
    elif v < high_thr:
        return "mid"
    else:
        return "high"

labels = ["low", "mid", "high"]

df_2004["CO_category_current"] = df_2004["CO(GT)"].apply(co_categories)
df_2005["CO_category_current"] = df_2005["CO(GT)"].apply(co_categories)

train_counts = df_2004["CO_category_current"].value_counts(normalize=True)
test_counts = df_2005["CO_category_current"].value_counts(normalize=True)

# Plotting class distribution comparison to show the distribution of CO(GT) levels into the three categories
x_idx = np.arange(len(labels))
plt.figure(figsize=(6, 4))
plt.bar(x_idx - 0.35 / 2,
        [train_counts.get(lbl, 0) for lbl in labels],
        width=0.35, label="2004 (train)")
plt.bar(x_idx + 0.35 / 2,
        [test_counts.get(lbl, 0) for lbl in labels],
        width=0.35, label="2005 (test)")
plt.xticks(x_idx, labels)
plt.ylabel("Proportion")
plt.title("CO(GT) class distribution based on thresholds")
plt.legend()
plt.tight_layout()

plot_path = "xgboost_classification_plots/CO_class_distribution_2004_vs_2005.png"
plt.savefig(plot_path, dpi=150)
plt.show()
print("Saved:", plot_path)

# Feature selection for modeling (without ground truth values)
gt_cols = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
drop_cols = set()

# Dropping columns that are not needed as features for modeling
for c in ["DateTime", "datetime", "Year"]:
    if c in df.columns:
        drop_cols.add(c)

for c in gt_cols:
    if c in df.columns:
        drop_cols.add(c)

for c in df.columns:
    if "_t+" in c:
        drop_cols.add(c)

for c in df.columns:
    if c.startswith("CO_bin_"):
        drop_cols.add(c)

numeric_cols = [c for c in df.columns if df[c].dtype.kind in "iufb"]
feature_cols = [c for c in numeric_cols if c not in drop_cols]

if not feature_cols:
    raise ValueError("No feature columns selected for modeling.")

# Mapping labels to integers for classification
label_map = {"low": 0, "mid": 1, "high": 2}
inv_label_map = {v: k for k, v in label_map.items()}

# Building the train and test dataset for each time lag horizon (1, 6, 12, 24 hours)
def create_dataset(h):
    tgt_reg = f"CO(GT)_t+{h}"
    if tgt_reg not in df.columns:
        print(f"Skipping t+{h}: column {tgt_reg} not found.")
        return None

    tgt_bin = f"CO_bin_t+{h}"

    train_set = df_2004.copy()
    test_set = df_2005.copy()

    train_set[tgt_bin] = train_set[tgt_reg].apply(co_categories)
    test_set[tgt_bin] = test_set[tgt_reg].apply(co_categories)

    train_set = train_set[train_set[tgt_bin].notna()]
    test_set = test_set[test_set[tgt_bin].notna()]

    if len(train_set) < 10 or len(test_set) < 10:
        print(f"Skipping t+{h} horizon: not enough labelled rows after filtering.")
        return None

    cut = int(0.75 * len(train_set))
    if cut == 0 or cut == len(train_set):
        print(f"Skipping t+{h} horizon: unable to make a reasonable train/val split.")
        return None

    X_train = train_set.iloc[:cut][feature_cols]
    y_train = train_set.iloc[:cut][tgt_bin].map(label_map)

    X_val = train_set.iloc[cut:][feature_cols]
    y_val = train_set.iloc[cut:][tgt_bin].map(label_map)

    X_test = test_set[feature_cols]
    y_test = test_set[tgt_bin].map(label_map)

    return X_train, y_train, X_val, y_val, X_test, y_test

# Train tuned XGBoost model for each lag horizon
horizons = [1, 6, 12, 24]
acc_by_h = {}
model_t1 = None

for h in horizons:
    print(f"\n Lag Horizon t+{h}")
    dataset = create_dataset(h)
    if dataset is None:
        continue

    X_train, y_train, X_val, y_val, X_test, y_test = dataset
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    classes = np.array([0, 1, 2])
    class_w = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight_dict = {cls: w for cls, w in zip(classes, class_w)}
    sample_weight = y_train.map(class_weight_dict)

    model = XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.8,
        gamma=0.1,
        objective="multi:softprob",
        num_class=3,
        reg_lambda=2.0,
        reg_alpha=0.5,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    if h == 1:
        model_t1 = model

    probs = model.predict_proba(X_test)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    acc_by_h[h] = acc

    print("Accuracy:", round(acc, 3))

    y_true_lab = [inv_label_map[int(i)] for i in y_test]
    y_pred_lab = [inv_label_map[int(i)] for i in y_pred]

    print(classification_report(
        y_true_lab, y_pred_lab, labels=labels
    ))

    cm = confusion_matrix(y_true_lab, y_pred_lab, labels=labels)

    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"XGBoost Confusion Matrix for horizon (t+{h})")
    plt.xticks(range(3), labels)
    plt.yticks(range(3), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()

    cm_path = f"xgboost_classification_plots/confusion_matrix_t+{h}.png"
    plt.savefig(cm_path, dpi=150)
    plt.show()
    print("Saved:", cm_path)

# Comparison plot: XGBoost vs. naive baseline (accuracy vs lag horizon)
baseline_file = "baseline_classification_accuracy.csv"

if os.path.exists(baseline_file) and acc_by_h:
    base_df = pd.read_csv(baseline_file)
    base_dict = dict(zip(base_df["horizon"], base_df["accuracy"]))

    common_h = [h for h in horizons if h in acc_by_h and h in base_dict]

    if common_h:
        plt.figure(figsize=(7, 4.5))
        plt.plot(common_h,
                 [base_dict[h] for h in common_h],
                 marker="o", label="Baseline", linewidth=2)
        plt.plot(common_h,
                 [acc_by_h[h] for h in common_h],
                 marker="o", label="Tuned XGBoost", linewidth=2)

        plt.xticks(common_h, [f"t+{h}" for h in common_h])
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.xlabel("Forecast horizon")
        plt.title("CO(GT) classification: XGBoost vs baseline model")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        comp_plot_path = "xgboost_classification_plots/xgboost_vs_baseline_accuracy.png"
        plt.savefig(comp_plot_path, dpi=150)
        plt.show()
        print("Saved comparison plot:", comp_plot_path)
    else:
        print("\nNo overlapping horizons between baseline and XGBoost results.")
else:
    print(
        "\nbaseline_classification_accuracy.csv not found or no XGBoost results. "
        "Run baseline_classification.py first to enable comparison plot."
    )