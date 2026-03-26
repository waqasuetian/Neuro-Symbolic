import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===================== DATA =====================
datasets = ["BIDS Rules", "TUH Rules"]

total_segments = [7534, 1020511]
correct_predictions = [5563, 688332]
wrong_predictions = [1971, 332179]

segment_accuracy = [0.738, 0.650]
avg_confidence = [0.271, 0.362]
test_score = [0.200, 0.235]

# Professional color palette
colors = {
    "accuracy": "#1f77b4",  # blue
    "confidence": "#ff7f0e",  # orange
    "test_score": "#2ca02c",  # green
     "wrong": "#4a90e2",         # light blue
    "correct": "#003f7f",           # dark blue
}

# ===================== PLOTS =====================

x = np.arange(len(datasets))  # the label locations

# ---------- 1. Accuracy, Confidence, Test Score ----------
plt.figure(figsize=(10,6))
width = 0.25

plt.bar(x - width, segment_accuracy, width=width, color=colors["accuracy"], label="Segment Accuracy")
plt.bar(x, avg_confidence, width=width, color=colors["confidence"], label="Average Confidence")
plt.bar(x + width, test_score, width=width, color=colors["test_score"], label="Confidence-weighted Test Score")

plt.xticks(x, datasets, fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.ylim(0, 1)
plt.title("Segment Metrics for Hybrid SVM + Rule-based Engine", fontsize=14)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ---------- 2. Correct vs Wrong Predictions ----------
plt.figure(figsize=(10,6))
bar_width = 0.35

plt.bar(x - bar_width/2, correct_predictions, width=bar_width, color=colors["correct"], label="Correct Predictions")
plt.bar(x + bar_width/2, wrong_predictions, width=bar_width, color=colors["wrong"], label="Wrong Predictions")

plt.xticks(x, datasets, fontsize=12)
plt.ylabel("Number of Segments", fontsize=12)
plt.title("Correct vs Wrong Segment Predictions", fontsize=14)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ---------- 3. Pie Charts for Both Datasets ----------
fig, axes = plt.subplots(1, 2, figsize=(12,6))

for i, ax in enumerate(axes):
    ax.pie(
        [correct_predictions[i], wrong_predictions[i]],
        labels=["Correct", "Wrong"],
        autopct="%1.1f%%",
        colors=[colors["correct"], colors["wrong"]],
        startangle=90,
        textprops={'fontsize': 12}
    )
    ax.set_title(f"{datasets[i]} Segment Prediction", fontsize=14)

plt.tight_layout()
plt.show()
