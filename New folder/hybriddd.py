import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# ------------------------------
# 1. Prepare Data
# ------------------------------
data = {
    'Dataset': ['BIDS Siena', 'BIDS Siena', 'BIDS Siena', 'TUH', 'TUH', 'TUH'],
    'Model': ['SVM', 'Rule-Based', 'Hybrid', 'SVM', 'Rule-Based', 'Hybrid'],
    'Accuracy': [97.03, 81.69, 93.84, 93.88, 76.97, 86.06],
    'TestScore': [np.nan, 0.1870, 0.1876, np.nan, 0.2191, 0.2182],
    'CorrectWrongRatio': [32.65, 4.46, 15.25, 15.35, 3.34, 6.17]
}

df = pd.DataFrame(data)

# ------------------------------
# 2. Create folder for plots
# ------------------------------
plot_folder = "plots"
os.makedirs(plot_folder, exist_ok=True)

# ------------------------------
# 3. Seaborn style
# ------------------------------
sns.set(style="whitegrid", font_scale=1.2)

# Helper function to add data labels
def add_labels(ax, fmt="{:.2f}"):
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(fmt.format(height),
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10)

# ------------------------------
# 4. 1. Grouped Bar Charts
# ------------------------------
metrics = ['Accuracy', 'TestScore', 'CorrectWrongRatio']

# Accuracy
plt.figure(figsize=(10,5))
ax = sns.barplot(data=df, x='Dataset', y='Accuracy', hue='Model', palette='Set2')
add_labels(ax)
plt.title('Segment-level Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 105)
plt.legend(title='Model')
plt.tight_layout()
plt.savefig(f"{plot_folder}/accuracy_comparison.png")
plt.close()

# TestScore (skip NaN)
plt.figure(figsize=(10,5))
ax = sns.barplot(data=df.dropna(subset=['TestScore']), x='Dataset', y='TestScore', hue='Model', palette='Set1')
add_labels(ax, fmt="{:.3f}")
plt.title('Confidence-weighted TestScore Comparison')
plt.ylabel('TestScore')
plt.ylim(0, 0.25)
plt.legend(title='Model')
plt.tight_layout()
plt.savefig(f"{plot_folder}/testscore_comparison.png")
plt.close()

# Correct:Wrong Ratio (log scale)
plt.figure(figsize=(10,5))
ax = sns.barplot(data=df, x='Dataset', y='CorrectWrongRatio', hue='Model', palette='Paired')
add_labels(ax)
plt.title('Correct:Wrong Ratio Comparison')
plt.ylabel('Correct:Wrong Ratio')
plt.yscale('log')
plt.legend(title='Model')
plt.tight_layout()
plt.savefig(f"{plot_folder}/correct_wrong_ratio.png")
plt.close()

# ------------------------------
# 5. Heatmaps (metrics vs models)
# ------------------------------
for dataset in df['Dataset'].unique():
    sub_df = df[df['Dataset']==dataset].set_index('Model')[metrics]
    plt.figure(figsize=(8,5))
    sns.heatmap(sub_df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(f"Performance Heatmap - {dataset}")
    plt.tight_layout()
    plt.savefig(f"{plot_folder}/heatmap_{dataset.replace(' ','_')}.png")
    plt.close()

# ------------------------------
# 6. Radar Charts
# ------------------------------
for dataset in df['Dataset'].unique():
    sub_df = df[df['Dataset']==dataset].set_index('Model')[metrics]
    categories = list(metrics)
    N = len(categories)

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)

    for i, model in enumerate(sub_df.index):
        values = sub_df.loc[model].fillna(0).values.tolist()  # fill NaN with 0 for radar
        values += values[:1]  # close the radar
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)

    plt.xticks([n / float(N) * 2 * pi for n in range(N)], categories)
    ax.set_title(f"Radar Chart - {dataset}")
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(f"{plot_folder}/radar_{dataset.replace(' ','_')}.png")
    plt.close()

# ------------------------------
# 7. Bubble Plot (Accuracy vs TestScore, bubble size = Correct:Wrong)
# ------------------------------
plt.figure(figsize=(10,6))
bubble_df = df.dropna(subset=['TestScore'])
sizes = bubble_df['CorrectWrongRatio'] * 10  # scale size for visibility

for dataset in bubble_df['Dataset'].unique():
    subset = bubble_df[bubble_df['Dataset']==dataset]
    plt.scatter(subset['Accuracy'], subset['TestScore'], s=subset['CorrectWrongRatio']*20, 
                alpha=0.6, label=dataset)

for idx, row in bubble_df.iterrows():
    plt.text(row['Accuracy']+0.2, row['TestScore']+0.002, row['Model'], fontsize=10)

plt.xlabel('Accuracy (%)')
plt.ylabel('TestScore')
plt.title('Bubble Plot: Accuracy vs TestScore (Bubble=Correct:Wrong Ratio)')
plt.legend(title='Dataset')
plt.tight_layout()
plt.savefig(f"{plot_folder}/bubble_plot.png")
plt.close()

print(f"All plots saved in folder: {plot_folder}")
