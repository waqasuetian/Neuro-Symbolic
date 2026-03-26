import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi

# ===================== OUTPUT FOLDER =====================
output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)

# ===================== DATA =====================
datasets = ["BIDS", "TUH"]
methods = ["SVM", "Rule-Based", "Hybrid"]

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
    [0.1870, 0.2191],   # SVM (if unavailable, keep same or set np.nan)
    [0.1870, 0.2191],   # Rule-Based
    [0.1876, 0.2182]    # Hybrid
]

# Correct/Wrong Ratio
ratios = [
    [3657/112, 461027/30044],
    [3079/690, 377962/113109],
    [3537/232, 422634/68437]
]

# ===================== STYLE =====================
sns.set_theme(style="whitegrid", font_scale=1.1)

colors = {
    "SVM": "#4C72B0",
    "Rule-Based": "#DD8452",
    "Hybrid": "#55A868"
}

wrong_color = "#C44E52"
x = np.arange(len(datasets))

# ===================== 1. GROUPED BAR PLOT =====================
plt.figure(figsize=(10,5))
width = 0.25

for i, method in enumerate(methods):
    plt.bar(x + (i-1)*width, accuracy[i], width=width,
            color=colors[method], label=method)

plt.xticks(x, datasets)
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison: SVM vs Rule-Based vs Hybrid")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_folder}/accuracy_grouped.png", dpi=300)
plt.close()

# ===================== 2. STACKED BAR PLOT =====================
plt.figure(figsize=(10,5))
bar_width = 0.25

for i, method in enumerate(methods):
    plt.bar(x + (i-1)*bar_width, correct_segments[i],
            width=bar_width, color=colors[method],
            label=f"{method} Correct")
    plt.bar(x + (i-1)*bar_width, wrong_segments[i],
            bottom=correct_segments[i],
            width=bar_width, color=wrong_color,
            label="Wrong" if i == 0 else "")

plt.xticks(x, datasets)
plt.ylabel("Number of Segments")
plt.title("Correct vs Wrong Segments")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_folder}/stacked_correct_wrong.png", dpi=300)
plt.close()

# ===================== 3. MULTI-PIE CHART (ONE FIGURE) =====================
fig, axes = plt.subplots(len(datasets), len(methods),
                         figsize=(12, 8))

for j, dataset in enumerate(datasets):
    for i, method in enumerate(methods):
        ax = axes[j, i]
        ax.pie([correct_segments[i][j], wrong_segments[i][j]],
               labels=["Correct", "Wrong"],
               autopct="%1.1f%%",
               colors=[colors[method], wrong_color],
               startangle=90,
               textprops={'fontsize':9})
        ax.set_title(f"{dataset}\n{method}")

plt.suptitle("Prediction Breakdown (Correct vs Wrong)", fontsize=14)
plt.tight_layout()
plt.savefig(f"{output_folder}/multi_pie_breakdown.png", dpi=300)
plt.close()

# ===================== 4. RADAR PLOT =====================
metrics = ["Accuracy", "TestScore", "Ratio"]
N = len(metrics)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)

for i, method in enumerate(methods):
    # average across datasets for clean radar
    avg_accuracy = np.mean(accuracy[i])
    avg_test = np.mean(test_score[i])
    avg_ratio = np.mean(ratios[i])

    vals = [
        avg_accuracy / 100,
        avg_test / 0.25,
        avg_ratio / max([max(r) for r in ratios])
    ]
    vals += vals[:1]

    ax.plot(angles, vals, linewidth=2,
            color=colors[method], label=method)
    ax.fill(angles, vals, color=colors[method], alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_yticklabels([])
plt.title("Overall Performance Radar")
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.savefig(f"{output_folder}/radar_plot.png", dpi=300)
plt.close()

# ===================== 5. BUBBLE PLOT =====================
plt.figure(figsize=(8,6))

for i, method in enumerate(methods):
    sizes = [r*3 for r in ratios[i]]
    plt.scatter(accuracy[i], test_score[i],
                s=sizes,
                color=colors[method],
                alpha=0.7,
                label=method)

    for j, dataset in enumerate(datasets):
        plt.text(accuracy[i][j]+0.3,
                 test_score[i][j]+0.002,
                 dataset,
                 fontsize=9)

plt.xlabel("Accuracy (%)")
plt.ylabel("TestScore")
plt.title("Accuracy vs TestScore (Bubble = Correct/Wrong Ratio)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_folder}/bubble_plot.png", dpi=300)
plt.close()

print(f"All plots saved in: {output_folder}")
