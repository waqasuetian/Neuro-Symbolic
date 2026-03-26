import json
import numpy as np
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    f1_score
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import SVC


class ReportGenerator:
    def __init__(self, config):
        self.config = config
        self.class_mapping = config.data.event_mapping
        self.inverse_mapping = config.data.inverse_mapping
        self.report_dir = config.paths.output_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Core Metrics
    # -------------------------
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        y_true = y_true.numpy() if hasattr(y_true, "numpy") else y_true
        y_pred = y_pred.numpy() if hasattr(y_pred, "numpy") else y_pred

        # Only use classes actually present in y_true
        classes_in_data = np.unique(y_true)
        print("Classes in data" , classes_in_data)
        class_names = [self.inverse_mapping[c] for c in classes_in_data]

        cm = confusion_matrix(y_true, y_pred, labels=classes_in_data)
        sensitivity = {}
        specificity = {}
        for idx, cls in enumerate(classes_in_data):
            TP = cm[idx, idx]
            FN = cm[idx, :].sum() - TP
            FP = cm[:, idx].sum() - TP
            TN = cm.sum() - (TP + FN + FP)
            sensitivity[class_names[idx]] = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity[class_names[idx]] = TN / (TN + FP) if (TN + FP) > 0 else 0

        report = classification_report(
            y_true,
            y_pred,
            labels=classes_in_data,
            target_names=class_names,
            zero_division=0,
            output_dict=True
        )
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        metrics = {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "weighted_f1": weighted_f1,
            "macro_f1": macro_f1
        }
        return metrics

    def save_metrics_json(self, metrics: dict, filename: str = "metrics.json"):
        path = self.report_dir / filename
        with path.open("w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {path}")

    # -------------------------
    # Plots
    # -------------------------
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], filename: str = "confusion_matrix.png"):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(self.report_dir / filename)
        plt.close()

    def plot_roc_curves(self, y_true, y_score=None, svm_model=None, X_test=None, classes=None, filename_prefix="roc_curve"):
    

        y_true = np.array(y_true)

        # Auto-select classes present
        if classes is None:
            classes = np.unique(y_true)

        # Get decision scores (SVM)
        if svm_model is not None and X_test is not None:
            if hasattr(svm_model, "decision_function"):
                y_score = svm_model.decision_function(X_test)
            elif hasattr(svm_model, "predict_proba"):
                y_score = svm_model.predict_proba(X_test)

        y_score = np.array(y_score)

        plt.figure()

        # ========== BINARY CASE ==========
        if len(classes) == 2:

            # If y_score is shape (N,1) → convert to (N,)
            if y_score.ndim == 2 and y_score.shape[1] == 1:
                y_score = y_score[:, 0]

            # If 2-column, select column 1 (positive class)
            if y_score.ndim == 2 and y_score.shape[1] == 2:
                y_score = y_score[:, 1]

            # If validation contains only class 0 or only class 1 → skip curves
            if len(np.unique(y_true)) < 2:
                print("ROC skipped: validation set has only one class")
                return

            fpr, tpr, _ = roc_curve(y_true, y_score)
            plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC={auc(fpr, tpr):.2f})")

        # ========== MULTI-CLASS CASE ==========
        else:
            y_true_bin = label_binarize(y_true, classes=classes)

            # Fix binary shape issue
            if y_true_bin.shape[1] == 1 and len(classes) == 2:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

            # Ensure scores match one-hot shape
            if y_score.ndim == 1:
                y_score = np.vstack([1 - y_score, y_score]).T

            for i, cls in enumerate(classes):
                # Skip if no positive samples for class
                if len(np.unique(y_true_bin[:, i])) < 2:
                    print(f"Skipping ROC for class {cls}: only one label present")
                    continue

                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                plt.plot(fpr, tpr, lw=2, label=f"{self.inverse_mapping[cls]} (AUC={auc(fpr, tpr):.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend(loc="lower right")
        plt.savefig(self.report_dir / f"{filename_prefix}.png")
        plt.close()
    def plot_precision_recall_curves(self, y_true, y_pred=None, svm_model=None,
                                    X_test=None, classes=None, filename_prefix="pr_curve"):

        
        y_true = np.array(y_true)

        # Auto-detect classes in validation set
        if classes is None:
            classes = np.unique(y_true)

        class_names = [self.inverse_mapping[c] for c in classes]

        # ---------- GET SCORES ----------
        if svm_model is not None and X_test is not None:
            if hasattr(svm_model, "decision_function"):
                y_score = svm_model.decision_function(X_test)

                # Convert binary SVM output to 2 columns
                if y_score.ndim == 1 and len(classes) == 2:
                    y_score = np.vstack([1 - y_score, y_score]).T

            elif hasattr(svm_model, "predict_proba"):
                y_score = svm_model.predict_proba(X_test)

        else:
            # fallback for non-prob models → use prediction labels
            y_score = label_binarize(y_pred, classes=classes)

            # Fix binary case shape
            if y_score.shape[1] == 1 and len(classes) == 2:
                y_score = np.hstack([1 - y_score, y_score])

        # ---------- TRUE LABEL BINARIZATION ----------
        y_true_bin = label_binarize(y_true, classes=classes)

        # Fix binary case
        if y_true_bin.shape[1] == 1 and len(classes) == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

        # ---------- PLOT ----------
        plt.figure()

        for i, cls in enumerate(classes):
            # If class not present → avoid crash
            if len(np.unique(y_true_bin[:, i])) < 2:
                print(f"Skipping PR for class {cls}: only one label present")
                continue

            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
            plt.plot(recall, precision, lw=2, label=f"{class_names[i]} (AUC={auc(recall, precision):.2f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves")
        plt.legend(loc="lower left")
        plt.savefig(self.report_dir / f"{filename_prefix}.png")
        plt.close()


    # def plot_feature_space(self, features: np.ndarray, labels: np.ndarray, class_names: List[str],
    #                        method: str = 'tsne', filename: str = "feature_space.png"):
    #     reducer = TSNE(n_components=2, random_state=42) if method.lower() == 'tsne' else PCA(n_components=2)
    #     reduced_features = reducer.fit_transform(features)
    #     plt.figure(figsize=(8,6))
    #     unique_labels = np.unique(labels)
    #     for idx, cls in enumerate(unique_labels):
    #         plt.scatter(
    #             reduced_features[labels==cls,0],
    #             reduced_features[labels==cls,1],
    #             label=class_names[idx],
    #             alpha=0.7
    #         )
    #     plt.title(f"{method.upper()} of Features")
    #     plt.xlabel("Component 1")
    #     plt.ylabel("Component 2")
    #     plt.legend()
    #     plt.savefig(self.report_dir / filename)
    #     plt.close()

    # def plot_svm_feature_importance(self, svm_model: SVC, feature_names: Optional[List[str]] = None,
    #                                 top_n: int = 20, filename: Optional[str] = "svm_feature_importance.png"):
    #     if not hasattr(svm_model, 'coef_'):
    #         raise ValueError("Feature importance is only available for linear SVMs.")
    #     feature_importance = np.mean(np.abs(svm_model.coef_), axis=0) if svm_model.coef_.ndim > 1 else np.abs(svm_model.coef_)
    #     indices = np.argsort(feature_importance)[-top_n:][::-1]
    #     if feature_names is None:
    #         feature_names = [f"F{i}" for i in range(len(feature_importance))]

    #     top_features = [feature_names[i] for i in indices]
    #     top_values = feature_importance[indices]

    #     plt.figure(figsize=(10,6))
    #     plt.barh(range(top_n), top_values[::-1], align='center', color='skyblue')
    #     plt.yticks(range(top_n), top_features[::-1])
    #     plt.xlabel("Feature Importance (|weight|)")
    #     plt.title("Top Feature Importances - Linear SVM")
    #     plt.tight_layout()
    #     plt.savefig(self.report_dir / filename)
    #     plt.close()
