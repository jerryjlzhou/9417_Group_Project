import matplotlib.pyplot as plt

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