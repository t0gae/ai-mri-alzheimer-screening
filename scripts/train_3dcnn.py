import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model


def build_model(input_shape=(64, 64, 64, 1)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv3D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling3D(2)(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv3D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling3D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs)


if __name__ == "__main__":
    X = np.load("data/processed/combined_scans.npy")
    y = np.load("data/processed/combined_labels.npy")

    if X.ndim == 6:
        X = X.squeeze(-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=4,
        class_weight={0: 1.0, 1: 5.0},
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=15, mode="max", restore_best_weights=True
            )
        ],
    )

    model.save("models/final_model.h5")