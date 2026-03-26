import matplotlib.pyplot as plt
import numpy as np
from math import pi

# ===================== DATA =====================
datasets = ["BIDS Rules", "TUH Rules"]

segment_accuracy = [0.738, 0.65]
avg_confidence = [0.271, 0.362]
test_score = [0.20, 0.23]
ratios = [5563/1971, 688332/332179]  # Correct / Wrong Ratio

# Normalize metrics to [0,1] for radar plot
max_vals = [1, 1, 1, max(ratios)]  # Segment accuracy, Avg confidence, TestScore, Correct/Wrong Ratio
values = [
    [segment_accuracy[0]/max_vals[0],
     avg_confidence[0]/max_vals[1],
     test_score[0]/max_vals[2],
     ratios[0]/max_vals[3]],
    [segment_accuracy[1]/max_vals[0],
     avg_confidence[1]/max_vals[1],
     test_score[1]/max_vals[2],
     ratios[1]/max_vals[3]]
]

metrics = ["Segment Accuracy", "Avg Confidence", "TestScore", "Correct/Wrong Ratio"]
N = len(metrics)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # Close the loop

# ===================== PLOT =====================
plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)

# Professional colors
colors = ["#1f77b4", "#ff7f0e"]  # Blue for BIDS, Orange for TUH

for i, dataset in enumerate(datasets):
    vals = values[i] + values[i][:1]  # Repeat first value to close the circle
    ax.plot(angles, vals, linewidth=2, color=colors[i], label=dataset)
    ax.fill(angles, vals, color=colors[i], alpha=0.25)

# Labels and grid
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=12)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=10)
ax.set_ylim(0, 1)

# Add value labels on each point
for i, dataset in enumerate(datasets):
    vals = values[i]
    for j, val in enumerate(vals):
        angle_rad = angles[j]
        ax.text(angle_rad, val + 0.05, f"{round(val,2)}", ha='center', va='bottom', fontsize=9)

plt.title("Hybrid SVM + Rule-based Engine Performance Radar", fontsize=14, pad=20)
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.show()
