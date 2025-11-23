import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("AirQualityUCI_feature_engineering.csv")



for h in [1, 6, 12, 24]:
    col = f"CO_bin_t+{h}"
    print(f"\nDistribution for {col} (raw data):\n{data[col].value_counts(dropna=False)}")


horizons = [1, 6, 12, 24]
results = []

for h in horizons:
    print("\n" + "=" * 70)
    print(f"Logistic Regression for horizon: CO_bin_t+{h}")
    print("=" * 70)

    target_col = f"CO_bin_t+{h}"


    data_clean = data.dropna(subset=[target_col]).dropna()     
    data_clean = data_clean.dropna()                     


    # Removing the future targets so that only the past info is used

    future_cols = [
        col for col in data_clean.columns
        if col.startswith("CO_bin_t+") 
        or col.endswith("_t+1")
        or col.endswith("_t+6")
        or col.endswith("_t+12")
        or col.endswith("_t+24")
    ]

    future_cols.append("DateTime")  


    X = data_clean.drop(columns=future_cols, errors="ignore")
    y = data_clean[target_col]

    X = X.select_dtypes(include=[np.number])

    print("Final feature matrix shape:", X.shape)

    #Train - test split 
    data_clean["DateTime"] = pd.to_datetime(data_clean["DateTime"])

    train_data = data_clean[data_clean["DateTime"].dt.year == 2004]
    test_data  = data_clean[data_clean["DateTime"].dt.year == 2005]

    X = X.select_dtypes(include=[np.number])
    print("Final feature-matrix shape:", X.shape)

    X_train = train_data[X.columns].select_dtypes(include=[np.number])
    y_train = train_data[target_col]

    X_test = test_data[X.columns].select_dtypes(include=[np.number])
    y_test = test_data[target_col]


    print("Train dataset:", X_train.shape, " Test dataset:", X_test.shape)


    # Class distribution overview (useful for imbalance checks)

    for H in [1, 6, 12, 24]:
        col = f"CO_bin_t+{H}"
        print(f"\nDistribution for {col}:\n{data[col].value_counts(dropna=False)}")

    print("\nTraining distribution:\n", y_train.value_counts())
    print("\nTesting distribution:\n", y_test.value_counts())


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)


    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=500
    )

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    acc_score = accuracy_score(y_test, predictions)
    results.append({"horizon": h, "accuracy": acc_score})

    print(f"\nAccuracy (t+{h}): {acc_score:.4f}")

    print("\nClassification Report:\n", classification_report(y_test, predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))


    # Confusion Matrix Heatmap
    if h == 1:
        conf_matrix = confusion_matrix(y_test, predictions, labels=["low", "mid", "high"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues',
        xticklabels=['low', 'mid', 'high'],
        yticklabels=['low', 'mid', 'high'])
        plt.title("Confusion Matrix – Logistic Regression (1-hour Forecast)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    # Actual vs Predicted Plots for the first 300 timestamps

        plt.figure(figsize=(14, 4))
        plt.plot(y_test.values[:300], label='Actual', marker='o', alpha=0.6)
        plt.plot(predictions[:300], label='Predicted', marker='x', alpha=0.6)
        plt.legend()
        plt.title("Actual vs Predicted CO Class (first 300 steps)")
        plt.xlabel("Time Index")
        plt.ylabel("Class")
        plt.tight_layout()
        plt.show()

    # Class distribution of training labels

        plt.figure(figsize=(6, 4))
        y_train.value_counts().reindex(["low","mid","high"],fill_value = 0).plot(
            kind='bar',
            color=['green', 'orange', 'red']
        )
        plt.title("Training CO Class Distribution (t+1)")
        plt.xlabel("Class Label")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

results_df = pd.DataFrame(results).sort_values("horizon")
print("\n\nLogistic Regression accuracy by horizon:")
print(results_df)



# Accuracy values from your runs
horizons = [1, 6, 12, 24]
logreg_acc = [0.7958, 0.6860, 0.6239, 0.5691]
baseline_acc = [0.7611, 0.3815, 0.3819, 0.6077]

plt.figure(figsize=(8, 5))
plt.plot(horizons, logreg_acc, marker='o', label='Logistic Regression', linewidth=2)
plt.plot(horizons, baseline_acc, marker='o', label='Naïve Baseline', linewidth=2)

plt.title("Logistic Regression vs Naïve Baseline Accuracy Across Horizons")
plt.xlabel("Forecast Horizon (hours)")
plt.ylabel("Accuracy")
plt.xticks([1, 6, 12, 24])
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
