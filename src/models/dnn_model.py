"""
dnn_model.py

Deep Neural Network for smart grid fault detection using TensorFlow/Keras.
"""

import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, roc_auc_score

logger = logging.getLogger(__name__)

tf.random.set_seed(42)


def build_dnn(input_dim: int, config: dict) -> keras.Model:
    """Build a fully-connected DNN with dropout regularisation."""
    layers_cfg = config.get("hidden_layers", [128, 64, 32])
    dropout_rate = config.get("dropout_rate", 0.3)
    lr = config.get("learning_rate", 0.001)

    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for units in layers_cfg:
        x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="fault_prob")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def train_dnn(X_train, y_train, X_val, y_val, config: dict) -> keras.Model:
    model = build_dnn(X_train.shape[1], config)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=config.get("patience", 15),
            restore_best_weights=True, mode="max"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.get("epochs", 100),
        batch_size=config.get("batch_size", 32),
        callbacks=callbacks,
        verbose=0,
    )
    best_epoch = np.argmax(history.history["val_auc"]) + 1
    logger.info("DNN best epoch: %d / val_auc: %.4f",
                best_epoch, max(history.history["val_auc"]))
    return model


def evaluate(model: keras.Model, X_test, y_test) -> dict:
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    logger.info("DNN AUC: %.4f", auc)
    logger.info("\n%s", classification_report(y_test, y_pred))
    return {"auc": auc}
