"""
train_transformer.py — Transformer encoder for hourly emissions forecasting.

The script deliberately uses src.data_prep.prepare(), so its windows, gap
handling, chronological splits, scaling, and persistence baseline are exactly
the same as the ANN experiment.

The network predicts the CHANGE since the last observed hour (y_t - y_{t-1})
rather than the absolute level. The final forecast is persistence plus that
change, so an output of zero reproduces the persistence baseline exactly. A
level-predicting model has to rebuild ~1,400 tons from scaled inputs every
hour, and in steady operation it missed by ~29 tons where persistence missed
by ~3; that is why it lost to persistence on MAE.

Run with:
    .venv/bin/python train_transformer.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import data_prep  # noqa: E402
import plots  # noqa: E402

MODEL_NAME = "Transformer"
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
EMBED_DIM = 64
NUM_HEADS = 4
FEED_FORWARD_DIM = 128
NUM_BLOCKS = 2
DROPOUT = 0.15

keras.utils.set_random_seed(SEED)


@keras.utils.register_keras_serializable(package="emissions")
class PositionalEmbedding(layers.Layer):
    """Add a learned position vector to each timestep in a fixed window."""

    def __init__(self, sequence_length: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.position_embedding = layers.Embedding(sequence_length, embed_dim)
        self.input_projection = layers.Dense(embed_dim)

    def build(self, input_shape):
        self.input_projection.build(input_shape)
        self.position_embedding.build((self.sequence_length,))
        super().build(input_shape)

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        return self.input_projection(inputs) + self.position_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "embed_dim": self.embed_dim,
        })
        return config


def transformer_block(inputs, block_number: int):
    """Apply pre-normalised self-attention followed by a feed-forward block."""
    attention_input = layers.LayerNormalization(
        epsilon=1e-6, name=f"transformer_{block_number}_attention_norm"
    )(inputs)
    attention_output = layers.MultiHeadAttention(
        num_heads=NUM_HEADS,
        key_dim=EMBED_DIM // NUM_HEADS,
        dropout=DROPOUT,
        name=f"transformer_{block_number}_self_attention",
    )(attention_input, attention_input)
    attention_output = layers.Dropout(DROPOUT)(attention_output)
    residual = layers.Add()([inputs, attention_output])

    feed_forward_input = layers.LayerNormalization(
        epsilon=1e-6, name=f"transformer_{block_number}_feed_forward_norm"
    )(residual)
    feed_forward = layers.Dense(
        FEED_FORWARD_DIM, activation="gelu",
        name=f"transformer_{block_number}_feed_forward_up",
    )(feed_forward_input)
    feed_forward = layers.Dropout(DROPOUT)(feed_forward)
    feed_forward = layers.Dense(
        EMBED_DIM, name=f"transformer_{block_number}_feed_forward_down"
    )(feed_forward)
    feed_forward = layers.Dropout(DROPOUT)(feed_forward)
    return layers.Add(name=f"transformer_{block_number}_output")(
        [residual, feed_forward]
    )


def build_transformer(lookback: int, n_features: int) -> keras.Model:
    inputs = keras.Input(shape=(lookback, n_features), name="hourly_window")
    encoded = PositionalEmbedding(lookback, EMBED_DIM, name="positional_embedding")(
        inputs
    )
    for block_number in range(1, NUM_BLOCKS + 1):
        encoded = transformer_block(encoded, block_number)

    encoded = layers.LayerNormalization(epsilon=1e-6, name="final_norm")(encoded)
    encoded = layers.GlobalAveragePooling1D(name="temporal_pooling")(encoded)
    encoded = layers.Dense(64, activation="gelu", name="regression_hidden")(encoded)
    encoded = layers.Dropout(DROPOUT)(encoded)
    outputs = layers.Dense(1, name="next_hour_emissions")(encoded)

    model = keras.Model(inputs, outputs, name="Transformer_emissions_forecaster")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def change_targets(data: dict):
    """Return the scaled hour-to-hour change for train/val, plus its scaler.

    The scaler is fitted on the training changes only, as with every other
    scaler in the project.
    """
    change_train = data["y_train"] - data["persistence_train"]
    change_val = data["y_val"] - data["persistence_val"]
    scaler = StandardScaler().fit(change_train.reshape(-1, 1))
    return (
        scaler.transform(change_train.reshape(-1, 1)).ravel(),
        scaler.transform(change_val.reshape(-1, 1)).ravel(),
        scaler,
    )


def predict_level(model, X, persistence, change_scaler) -> np.ndarray:
    """Forecast in real units: last observed value + predicted change."""
    change_scaled = model.predict(X, verbose=0).reshape(-1, 1)
    return persistence + change_scaler.inverse_transform(change_scaled).ravel()


def evaluate(y_true, y_pred, persistence) -> dict:
    def score(pred):
        mae = mean_absolute_error(y_true, pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
        safe = np.maximum(np.abs(y_true), 1e-6)
        mape = float(np.mean(np.abs((y_true - pred) / safe)) * 100)
        return {
            "mae": float(mae),
            "rmse": rmse,
            "mape": mape,
            "r2": float(r2_score(y_true, pred)),
        }

    return {"model": score(y_pred), "persistence": score(persistence)}


def run_for_target(target: str) -> dict:
    print(f"\n{'#' * 70}\n#  {MODEL_NAME} — {target}\n{'#' * 70}")
    data = data_prep.prepare(target)
    y_train_change, y_val_change, change_scaler = change_targets(data)
    model = build_transformer(data["lookback"], data["X_train"].shape[2])
    print(f"\n  Architecture ({model.count_params():,} trainable parameters):")
    model.summary(print_fn=lambda line: print("   ", line))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1
        ),
    ]

    print(f"\n  Training (max {EPOCHS} epochs, batch size {BATCH_SIZE})...")
    started = time.time()
    history = model.fit(
        data["X_train"], y_train_change,
        validation_data=(data["X_val"], y_val_change),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )
    train_seconds = time.time() - started

    prediction = predict_level(
        model, data["X_test"], data["persistence_test"], change_scaler
    )
    metrics = evaluate(data["y_test"], prediction, data["persistence_test"])
    model_metrics = metrics["model"]
    baseline_metrics = metrics["persistence"]
    print(
        f"\n  {MODEL_NAME}: MAE {model_metrics['mae']:,.3f} | "
        f"RMSE {model_metrics['rmse']:,.3f} | R² {model_metrics['r2']:.4f}"
    )
    print(
        f"  Persistence: MAE {baseline_metrics['mae']:,.3f} | "
        f"RMSE {baseline_metrics['rmse']:,.3f} | R² {baseline_metrics['r2']:.4f}"
    )

    plots.fig_training_history(history, target, MODEL_NAME)
    plots.fig_prediction_timeseries(
        data["ts_test"], data["y_test"], prediction, target, MODEL_NAME
    )
    plots.fig_scatter(data["y_test"], prediction, target, MODEL_NAME, model_metrics["r2"])
    plots.fig_residuals(data["y_test"], prediction, target, MODEL_NAME)
    plots.fig_baseline_comparison(metrics, target, MODEL_NAME)
    plots.fig_error_breakdown(
        data["ts_test"], data["y_test"], prediction, target, MODEL_NAME
    )

    slug = "co2" if "CO2" in target else "nox"
    model.save(RESULTS_DIR / f"transformer_{slug}.keras")
    np.savez_compressed(
        RESULTS_DIR / f"transformer_{slug}_predictions.npz",
        ts=data["ts_test"], y_true=data["y_test"], y_pred=prediction,
        persistence=data["persistence_test"],
    )
    return {
        "target": target,
        "model": MODEL_NAME,
        "epochs_run": len(history.history["loss"]),
        "train_seconds": round(train_seconds, 1),
        "parameters": int(model.count_params()),
        "n_train": int(len(data["y_train"])),
        "n_val": int(len(data["y_val"])),
        "n_test": int(len(data["y_test"])),
        "metrics": metrics,
    }


def main() -> None:
    print(
        f"TensorFlow {tf.__version__} | seed {SEED} | "
        f"lookback {data_prep.LOOKBACK} h | horizon {data_prep.HORIZON} h"
    )
    results = [run_for_target(target) for target in data_prep.TARGETS]
    output = RESULTS_DIR / "transformer_metrics.json"
    output.write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 70}\nSUMMARY — {MODEL_NAME}\n{'=' * 70}")
    for result in results:
        metrics = result["metrics"]["model"]
        print(
            f"  {result['target']:26s} MAE {metrics['mae']:9,.1f} "
            f"RMSE {metrics['rmse']:9,.1f}   R² {metrics['r2']:.4f}"
        )
    print(f"\n  Metrics -> results/{output.name}")


if __name__ == "__main__":
    main()