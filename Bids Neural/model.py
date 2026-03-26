"""
Hybrid CNN-LSTM + PCA + SVM EEG Seizure Detection Pipeline
- Uses ONLY Focal Loss
- Runs ALL epochs (NO early stopping)
- NO oversampling
- Saves BEST model based on validation loss
"""

import joblib
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from data.data_loader import SimpleEEGLoader
from config.settings import Config


# GLOBAL SETTINGS

SEED = 42
EPOCHS = 40
BATCH_SIZE = 32
tf.random.set_seed(SEED)
np.random.seed(SEED)

# FOCAL LOSS 

def focal_loss(alpha=0.25, gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true_one_hot * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))

    return loss_fn

# CNN-LSTM FEATURE EXTRACTOR

def build_cnn_lstm(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv1D(32, 7, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.LSTM(128, return_sequences=False)(x)

    x = layers.Dense(128, activation="relu", name="deep_features")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="CNN_LSTM_EEG")



# MAIN PIPELINE

def main():
    cfg = Config()
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = SimpleEEGLoader(cfg)

    print("\n[1] Loading data...")
    X, y = loader.load_all(cfg.paths.train_data)
    print(f"Loaded: {X.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    num_classes = len(np.unique(y))

    print("\n[2] Building model...")
    model = build_cnn_lstm(X_train.shape[1:], num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=focal_loss(alpha=0.25, gamma=2.0),
        metrics=["accuracy"]
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=out_dir / "best_cnn_lstm.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    print("\n[3] Training CNN-LSTM (NO early stopping)...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint],
        verbose=2
    )

    print("\n[4] Loading BEST saved model...")
    model = tf.keras.models.load_model(
        out_dir / "best_cnn_lstm.keras",
        custom_objects={"loss": focal_loss(alpha=0.25, gamma=2.0)}
    )

    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("deep_features").output
    )

    print("\n[5] Extracting deep features...")
    X_train_feat = feature_extractor.predict(X_train, batch_size=64)
    X_val_feat = feature_extractor.predict(X_val, batch_size=64)

    print("\n[6] Scaling + PCA...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_val_scaled = scaler.transform(X_val_feat)

    pca = PCA(n_components=64, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    print("\n[7] Training SVM...")
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
    svm.fit(X_train_pca, y_train)

    print("\n[8] Evaluating...")
    y_pred = svm.predict(X_val_pca)


    print("HYBRID PIPELINE RESULTS")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred, digits=4))

    print("\n[9] Saving ...")
    model.save(out_dir / "cnn_lstm_final.keras")
    feature_extractor.save(out_dir / "feature_extractor.keras")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(pca, out_dir / "pca.joblib")
    joblib.dump(svm, out_dir / "svm.joblib")

    print("✅ All saved successfully.")


if __name__ == "__main__":
    main()
