import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi

# ===================== DATA =====================
datasets = ["Small Dataset", "Large Dataset"]

methods = ["SVM", "Rule-Based", "Hybrid"]

# Total segments
total_segments = [
    [3769, 491071],  # SVM
    [3769, 491071],  # Rule-Based
    [3769, 491071]   # Hybrid
]

# Correct / Wrong
correct_segments = [
    [3657, 461027],  # SVM
    [3079, 377962],  # Rule
    [3537, 422634]   # Hybrid
]
wrong_segments = [
    [112, 30044],    # SVM
    [690, 113109],   # Rule
    [232, 68437]     # Hybrid
]

# Accuracy
accuracy = [
    [97.03, 93.88],
    [81.69, 76.97],
    [93.84, 86.06]
]

# Confidence-weighted TestScore
test_score = [
    [0.1870, 0.2191],
    [0.1870, 0.2191],  # same as rule for display, can update
    [0.1876, 0.2182]
]

# Correct/Wrong Ratio
ratios = [
    [3657/112, 461027/30044],
    [3079/690, 377962/113109],
    [3537/232, 422634/68437]
]

# Professional color palette
colors = {
    "SVM": "#1f77b4",        # Blue
    "Rule-Based": "#ff7f0e", # Orange
    "Hybrid": "#2ca02c"      # Green
}

x = np.arange(len(datasets))  # x-axis positions

# ===================== 1. Grouped Bar Plot =====================
plt.figure(figsize=(12,6))
width = 0.25

for i, method in enumerate(methods):
    plt.bar(x + (i-1)*width, accuracy[i], width=width, color=colors[method], label=f"{method} Accuracy")

plt.xticks(x, datasets)
plt.ylabel("Accuracy (%)")
plt.title("SVM vs Rule-Based vs Hybrid Accuracy")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.show()

# ===================== 2. Stacked Bar Plot: Correct vs Wrong =====================
plt.figure(figsize=(12,6))
bar_width = 0.25
for i, method in enumerate(methods):
    plt.bar(x + (i-1)*bar_width, correct_segments[i], width=bar_width, color=colors[method], label=f"{method} Correct")
    plt.bar(x + (i-1)*bar_width, wrong_segments[i], bottom=correct_segments[i], width=bar_width, color="#d62728", label=f"{method} Wrong" if i==0 else "")

plt.xticks(x, datasets)
plt.ylabel("Number of Segments")
plt.title("Correct vs Wrong Segments")
plt.legend()
plt.tight_layout()
plt.show()

# ===================== 3. Pie Charts =====================
for j, dataset in enumerate(datasets):
    plt.figure(figsize=(8,8))
    for i, method in enumerate(methods):
        plt.pie([correct_segments[i][j], wrong_segments[i][j]],
                labels=["Correct", "Wrong"],
                autopct="%1.1f%%",
                colors=[colors[method], "#d62728"],
                startangle=90)
        plt.title(f"{dataset} - {method} Prediction Breakdown")
        plt.show()

# ===================== 4. Radar Plot =====================
metrics = ["Accuracy (%)", "TestScore", "Correct/Wrong Ratio"]
N = len(metrics)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)

for i, method in enumerate(methods):
    for j, dataset in enumerate(datasets):
        vals = [accuracy[i][j], test_score[i][j]*100, ratios[i][j]]  # scale test_score for visibility
        vals_norm = [accuracy[i][j]/100, test_score[i][j]/1, ratios[i][j]/max(ratios[i])]  # normalize for radar
        vals_norm += vals_norm[:1]
        ax.plot(angles, vals_norm, linewidth=2, color=colors[method], label=f"{method} ({dataset})")
        ax.fill(angles, vals_norm, color=colors[method], alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=12)
ax.set_yticklabels([])
plt.title("SVM vs Rule vs Hybrid Performance Radar", fontsize=14, pad=20)
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

# ===================== 5. Bubble Plot =====================
plt.figure(figsize=(10,6))
for i, method in enumerate(methods):
    sizes = [ratio/10 for ratio in ratios[i]]  # bubble proportional to Correct/Wrong ratio
    plt.scatter(accuracy[i], test_score[i], s=sizes, color=colors[method], alpha=0.7, label=method)
    for j, dataset in enumerate(datasets):
        plt.text(accuracy[i][j]+0.5, test_score[i][j]+0.002, f"{dataset}", fontsize=10)

plt.xlabel("Accuracy (%)")
plt.ylabel("Confidence-weighted TestScore")
plt.title("Accuracy vs TestScore (Bubble size = Correct/Wrong Ratio)")
plt.legend()
plt.tight_layout()
plt.show()
