"""
plot_svm_analysis_live.py

Professional SVM + PCA analysis directly from EEG pipeline.
Handles binary/multiclass automatically.
Generates:
 - PCA 2D scatter (true vs predicted)
 - PCA explained variance plot
 - Support vectors (highlighted)
 - SVM decision function histogram
 - ROC & Precision-Recall curves per class
 - Optional: t-SNE embedding of 64D features
"""

import os
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from config.settings import Config
from data.data_loader import SimpleEEGLoader
import tensorflow as tf

# ==================== Config ====================
cfg = Config()
OUTPUT_DIR = Path(cfg.paths.output_dir)
PLOT_TSNE = True  # set False to skip t-SNE
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== Load Artifacts ====================
scaler = joblib.load(OUTPUT_DIR / "scaler.joblib")
pca = joblib.load(OUTPUT_DIR / "pca.joblib")
svm = joblib.load(OUTPUT_DIR / "svm.joblib")

# ==================== Load Feature Extractor ====================
feat_path = OUTPUT_DIR / "feature_extractor.keras"
full_model_path = OUTPUT_DIR / "cnn_lstm_trained.keras"

if feat_path.exists():
    feat_extractor = tf.keras.models.load_model(feat_path)
elif full_model_path.exists():
    model = tf.keras.models.load_model(full_model_path)
    try:
        feat_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer("deep_features").output)
    except Exception:
        feat_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
else:
    raise FileNotFoundError("No feature extractor or full model found.")

# ==================== Load Test Data ====================
loader = SimpleEEGLoader(cfg)
X_test, y_test = loader.load_all(cfg.paths.test_data)

print(f"Loaded test data: X_test {X_test.shape}, y_test {y_test.shape}")

# ==================== Extract Features ====================
batch_size = getattr(cfg.model, "batch_size", 32)
X_test_feat = feat_extractor.predict(X_test.astype(np.float32), batch_size=batch_size, verbose=1)

# ==================== Scale + PCA ====================
X_scaled = scaler.transform(X_test_feat)
X_pca = pca.transform(X_scaled)
print("Shapes: Features {}, PCA {}".format(X_scaled.shape, X_pca.shape))

# ==================== SVM Predictions ====================
y_pred = svm.predict(X_pca)
decision_scores = svm.decision_function(X_pca) if hasattr(svm, "decision_function") else None

# ==================== Classes & Binarization ====================
classes = np.unique(y_test)
n_classes = len(classes)

# For ROC/PR, force binary format for both binary & multiclass
y_bin = label_binarize(y_test, classes=classes)
if n_classes == 2 and y_bin.shape[1] == 1:
    y_bin = np.hstack([1 - y_bin, y_bin])  # shape (n_samples, 2)

colors = plt.cm.coolwarm(np.linspace(0, 1, n_classes))

# ==================== PCA 2D Scatter ====================
plt.figure(figsize=(8,6))
for cls, c in zip(classes, colors):
    plt.scatter(X_pca[y_test == cls, 0], X_pca[y_test == cls, 1],
                label=f"Class {cls}", edgecolor='k', alpha=0.7, color=c)

# Highlight support vectors in PCA
if hasattr(svm, "support_"):
    sv_indices = svm.support_
    sv_indices = sv_indices[sv_indices < X_pca.shape[0]]  # safe indexing
    plt.scatter(X_pca[sv_indices, 0], X_pca[sv_indices, 1],
                s=60, facecolors='none', edgecolors='k', linewidths=1.5, label="Support Vectors")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D Scatter with Support Vectors")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_2d_support_vectors.png")
plt.close()
print("Saved PCA 2D scatter with support vectors.")

# ==================== PCA Explained Variance ====================
plt.figure(figsize=(7,5))
var_ratio = pca.explained_variance_ratio_
plt.plot(np.cumsum(var_ratio)*100, marker='o')
plt.xlabel("Number of PCA Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("PCA Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_explained_variance.png")
plt.close()
print("Saved PCA explained variance plot.")

# ==================== SVM Decision Function Histogram ====================
if decision_scores is not None:
    plt.figure(figsize=(7,5))
    if n_classes == 2:
        plt.hist(decision_scores, bins=50, color='skyblue', edgecolor='k', alpha=0.7)
        plt.xlabel("Decision Score")
        plt.ylabel("Number of Samples")
        plt.title("SVM Decision Function Histogram (Binary)")
    else:
        for i in range(n_classes):
            plt.hist(decision_scores[:, i], bins=50, alpha=0.5, label=f"Class {classes[i]}")
        plt.xlabel("Decision Score")
        plt.ylabel("Number of Samples")
        plt.title("SVM Decision Function Histogram (Multiclass)")
        plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "svm_decision_hist.png")
    plt.close()
    print("Saved SVM decision function histogram.")

# ==================== ROC Curves ====================
plt.figure(figsize=(8,6))
for i in range(n_classes):
    if n_classes == 2:
        y_score_cls = decision_scores
    else:
        y_score_cls = decision_scores[:, i]
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score_cls)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"Class {classes[i]} (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves per Class")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curves.png")
plt.close()
print("Saved ROC curves.")

# ==================== Precision-Recall Curves ====================
plt.figure(figsize=(8,6))
for i in range(n_classes):
    if n_classes == 2:
        y_score_cls = decision_scores
    else:
        y_score_cls = decision_scores[:, i]
    precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score_cls)
    plt.plot(recall, precision, lw=2, label=f"Class {classes[i]}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves per Class")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pr_curves.png")
plt.close()
print("Saved Precision-Recall curves.")

# ==================== Optional: t-SNE of 64D Features ====================
if PLOT_TSNE:
    tsne = TSNE(n_components=2, random_state=42, init='pca')
    X_tsne = tsne.fit_transform(X_scaled)
    plt.figure(figsize=(8,6))
    for cls, c in zip(classes, colors):
        plt.scatter(X_tsne[y_test==cls,0], X_tsne[y_test==cls,1],
                    label=f"Class {cls}", alpha=0.7, color=c)
    plt.title("t-SNE Embedding of 64D Features")
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tsne_64d_features.png")
    plt.close()
    print("Saved t-SNE embedding of 64D features.")

print("All plots saved in:", OUTPUT_DIR.resolve())
