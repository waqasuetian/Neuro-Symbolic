"""
evaluate_saved_pipeline.py

Load saved artifacts from the training pipeline and generate:
 - classification report (JSON)
 - confusion matrix (PNG)
 - ROC curve(s) (PNG)
 - Precision-Recall curve(s) (PNG)
 - (optional) feature-space PCA scatter (PNG)
"""

import os
from pathlib import Path
import joblib
import numpy as np
import tensorflow as tf

from config.settings import Config
from data.data_loader import SimpleEEGLoader
from evaluation.reporter import ReportGenerator


# Utility helpers

def load_artifacts(output_dir: Path):
    """Load scaler, pca and svm from output_dir. Raises FileNotFoundError if missing."""
    output_dir = Path(output_dir)
    scaler_path = output_dir / "scaler.joblib"
    pca_path = output_dir / "pca.joblib"
    svm_path = output_dir / "svm.joblib"

    missing = [p for p in (scaler_path, pca_path, svm_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifact(s): {missing}")

    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    svm = joblib.load(svm_path)
    return scaler, pca, svm

def load_feature_extractor_from_saved(output_dir: Path):
    """
    Try to load a saved 'feature_extractor.keras' first.
    If missing but a full CNN model 'cnn_lstm_trained.keras' exists, try to create the extractor.
    """
    feat_path = output_dir / "feature_extractor.keras"
    full_model_path = output_dir / "cnn_lstm_trained.keras"

    if feat_path.exists():
        return tf.keras.models.load_model(feat_path)
    if full_model_path.exists():
        model = tf.keras.models.load_model(full_model_path)
        # try to locate 'deep_features' layer; fallback to second-last if not found
        try:
            return tf.keras.Model(inputs=model.input, outputs=model.get_layer("deep_features").output)
        except Exception:
            # fallback to using the penultimate layer
            if len(model.layers) >= 2:
                return tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
            raise RuntimeError("Could not create feature extractor from saved model.")
    raise FileNotFoundError("No saved feature_extractor.keras or cnn_lstm_trained.keras found.")

def main():
    cfg = Config()
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Using output_dir:", output_dir)

    # 1) Load saved artifacts (scaler, pca, svm)
    try:
        scaler, pca, svm = load_artifacts(output_dir)
    except FileNotFoundError as e:
        print("Error loading artifacts:", e)
        return

    # 2) Load feature extractor (saved)
    try:
        feat_extractor = load_feature_extractor_from_saved(output_dir)
    except Exception as e:
        print("Error loading feature extractor:", e)
        return

    # 3) Load test dataset
    loader = SimpleEEGLoader(cfg)
    X_test, y_test = loader.load_all(cfg.paths.test_data)
    if X_test.size == 0:
        print("No test data found at", cfg.paths.test_data)
        return

    print(f"Loaded test data: X_test {X_test.shape}, y_test {y_test.shape}")

    # 4) Extract deep features (batched)
    batch_size = getattr(cfg.model, "batch_size", 32) if hasattr(cfg, "model") else 32
    print("Extracting features from test set (this may take a while)...")
    X_test_feat = feat_extractor.predict(X_test.astype(np.float32), batch_size=batch_size, verbose=1)

    # 5) Scale + PCA transform
    X_test_scaled = scaler.transform(X_test_feat)
    X_test_pca = pca.transform(X_test_scaled)
    print("Feature -> scaled -> PCA shapes:", X_test_feat.shape, X_test_scaled.shape, X_test_pca.shape)
    # 6) Make predictions: labels + scores for ROC/PR
    y_pred = svm.predict(X_test_pca)

    
    # Prefer probabilities; fallback to decision_function; otherwise produce score-alike from labels
    y_score = None
    if hasattr(svm, "predict_proba"):
        try:
            y_score = svm.predict_proba(X_test_pca)
            print("Using predict_proba from SVM for ROC/PR.")
        except Exception:
            y_score = None
    if y_score is None and hasattr(svm, "decision_function"):
        try:
            dec = svm.decision_function(X_test_pca)
            print("Using decision_function from SVM for ROC/PR.")
            # decision_function can be (n,) for binary or (n, n_classes)
            y_score = dec
        except Exception:
            y_score = None

    # If still None, fallback to label-binarized predictions (coarse)
    if y_score is None:
        print("SVM has no probability/decision function available — using label-binarized predictions as fallback.")
        from sklearn.preprocessing import label_binarize
        classes_present = np.unique(y_test)
        y_score = label_binarize(y_pred, classes=classes_present)
        # If binary: shape may be (n,1) → convert to two-column format
        if y_score.ndim == 2 and y_score.shape[1] == 1:
            y_score = np.hstack([1 - y_score, y_score])

    # 7) Generate reports via your ReportGenerator
    reporter = ReportGenerator(cfg)

    # Compute metrics and save to JSON
    metrics = reporter.compute_metrics(y_test, y_pred)
    reporter.save_metrics_json(metrics)

    # Prepare human-readable class names in order of classes used in metrics
    classes_in_data = np.unique(y_test)
    class_names = [reporter.inverse_mapping[int(c)] for c in classes_in_data]

    # Confusion matrix plot (reporter expects cm array + names)
    cm = np.array(metrics["confusion_matrix"])
    reporter.plot_confusion_matrix(cm, class_names, filename="confusion_matrix.png")
    print("Saved confusion_matrix.png")

    # ROC & PR curves — call reporter with SVM and PCA test features (reporter will call decision_function/predict_proba)
    try:
        # pass svm and PCA transformed features; reporter will call svm.decision_function(X_test) or predict_proba
        reporter.plot_roc_curves(y_true=y_test, y_score=y_score, svm_model=svm, X_test=X_test_pca, classes=list(classes_in_data), filename_prefix="roc_curve")
        print("Saved ROC curve(s).")
    except Exception as e:
        print("ROC plotting failed:", e)

    try:
        reporter.plot_precision_recall_curves(y_true=y_test, y_pred=y_pred, svm_model=svm, X_test=X_test_pca, classes=list(classes_in_data), filename_prefix="pr_curve")
        print("Saved Precision-Recall curve(s).")
    except Exception as e:
        print("PR plotting failed:", e)

    print("All done. Artifacts and reports available at:", output_dir.resolve())

if __name__ == "__main__":
    main()
