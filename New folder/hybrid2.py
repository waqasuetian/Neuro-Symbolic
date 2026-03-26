import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data
data = {
    'Dataset': ['BIDS Siena', 'BIDS Siena', 'BIDS Siena', 'TUH', 'TUH', 'TUH'],
    'Model': ['SVM', 'Rule-Based', 'Hybrid', 'SVM', 'Rule-Based', 'Hybrid'],
    'Accuracy': [97.03, 81.69, 93.84, 93.88, 76.97, 86.06],
    'TestScore': [np.nan, 0.1870, 0.1876, np.nan, 0.2191, 0.2182],
    'CorrectWrongRatio': [32.65, 4.46, 15.25, 15.35, 3.34, 6.17]
}

df = pd.DataFrame(data)

# Set Seaborn style
sns.set(style="whitegrid", font_scale=1.2)

# Function to add value labels
def add_labels(ax):
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}', 
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10)

# 1. Accuracy Comparison
plt.figure(figsize=(10,5))
ax1 = sns.barplot(data=df, x='Dataset', y='Accuracy', hue='Model', palette='Set2')
add_labels(ax1)
plt.title('Segment-level Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 105)
plt.legend(title='Model')
plt.tight_layout()
plt.show()

# 2. Confidence-weighted TestScore Comparison
plt.figure(figsize=(10,5))
ax2 = sns.barplot(data=df.dropna(subset=['TestScore']), x='Dataset', y='TestScore', hue='Model', palette='Set1')
add_labels(ax2)
plt.title('Confidence-weighted TestScore Comparison')
plt.ylabel('TestScore')
plt.ylim(0, 0.25)
plt.legend(title='Model')
plt.tight_layout()
plt.show()

# 3. Correct:Wrong Ratio Comparison (log scale for clarity)
plt.figure(figsize=(10,5))
ax3 = sns.barplot(data=df, x='Dataset', y='CorrectWrongRatio', hue='Model', palette='Paired')
add_labels(ax3)
plt.title('Correct:Wrong Ratio Comparison')
plt.ylabel('Correct:Wrong Ratio')
plt.yscale('log')  # log scale to better visualize large differences
plt.legend(title='Model')
plt.tight_layout()
plt.show()
