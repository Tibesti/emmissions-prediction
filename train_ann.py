"""
train_ann.py — The Artificial Neural Network (ANN) model.
=========================================================

WHAT THIS SCRIPT DOES, IN ONE PARAGRAPH
---------------------------------------
It reads the last 24 hours of plant and weather readings, and predicts what
CO2 (and separately, NOx) emissions will be in the NEXT hour. It does this with
a "feed-forward" neural network — the simplest kind — which will later be
compared against an LSTM and a Transformer. This script trains one model per
pollutant, scores both against a naive baseline, and saves twelve figures.

HOW TO RUN IT
-------------
    .venv/bin/python train_ann.py

WHAT AN ANN ACTUALLY IS (the 30-second version)
-----------------------------------------------
A neural network is a stack of layers. Each layer multiplies its inputs by a
grid of numbers ("weights"), adds an offset ("bias"), and passes the result
through a simple bending function ("activation"). Stacking several such layers
lets the network approximate curved, non-linear relationships — which is what
we need, because emissions do not rise in a perfectly straight line with load.

Training means: make a guess, measure how wrong it was, nudge every weight
slightly in the direction that would have made it less wrong, repeat a few
million times. The nudging algorithm is called backpropagation, and it is done
for you by TensorFlow.

THE ONE THING THAT MAKES AN *ANN* DIFFERENT FROM AN LSTM
--------------------------------------------------------
An ANN has no concept of order. We hand it a 24-hour x 18-feature window, but
the very first thing it does is FLATTEN that into one long row of 432 numbers.
It does not know that number 5 came one hour before number 23. It has to
rediscover any time structure from scratch, treating each of the 432 slots as
an independent variable.

That is precisely the limitation the LSTM and Transformer are designed to fix,
and it is why this comparison is worth making. The ANN here is your BASELINE
ARCHITECTURE — expect it to work reasonably well and to be beaten later.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Silence TensorFlow's routine startup chatter so the output stays readable.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import data_prep          # noqa: E402
import plots              # noqa: E402

MODEL_NAME = "ANN"
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# REPRODUCIBILITY
# Neural networks start from RANDOM weights. Run the same script twice and you
# get slightly different numbers. Fixing the "seed" makes that randomness
# repeatable, so the figures in your thesis can be regenerated exactly. Always
# state the seed in your methodology.
# ---------------------------------------------------------------------------
SEED = 42
keras.utils.set_random_seed(SEED)

# ---------------------------------------------------------------------------
# HYPERPARAMETERS — the settings that are chosen, not learned
# ---------------------------------------------------------------------------
EPOCHS = 120           # max passes over the training data (early stopping
                       # will almost certainly halt it sooner)
BATCH_SIZE = 128       # how many windows the model looks at before each nudge.
                       # Bigger = faster and smoother, but coarser steps.
LEARNING_RATE = 1e-3   # how big each nudge is. Too big and training thrashes
                       # around the answer; too small and it never gets there.
                       # 0.001 is the standard starting point for Adam.
HIDDEN_UNITS = [256, 128, 64]   # neurons per hidden layer, narrowing as it
                                # goes — a funnel that forces the network to
                                # compress the input into what actually matters
DROPOUT = 0.2          # see build_ann() for what this does


def build_ann(lookback: int, n_features: int) -> keras.Model:
    """Assemble the neural network, layer by layer.

    Every layer is explained below. Read it as a pipeline: numbers go in at the
    top, one prediction comes out at the bottom.
    """
    model = keras.Sequential(name="ANN_emissions_forecaster")

    # --- INPUT ------------------------------------------------------------
    # Declares the shape of one training example: 24 hours x 18 features.
    # Note we describe ONE window here; the batch dimension is implicit.
    model.add(keras.Input(shape=(lookback, n_features)))

    # --- FLATTEN ----------------------------------------------------------
    # THIS is the line that makes it an ANN rather than a sequence model.
    # It reshapes the 24 x 18 grid into a single row of 432 numbers, discarding
    # all information about which hour each number came from. Everything below
    # this line treats those 432 values as 432 unrelated inputs.
    model.add(layers.Flatten())

    # --- HIDDEN LAYERS ----------------------------------------------------
    for i, units in enumerate(HIDDEN_UNITS, start=1):
        # Dense = "fully connected": every input connects to every neuron.
        # A 432 -> 256 Dense layer therefore holds 432*256 + 256 = 110,848
        # learnable numbers. This is where the pattern actually gets stored.
        model.add(layers.Dense(units, name=f"dense_{i}"))

        # BatchNormalization re-centres each layer's output to roughly mean 0,
        # std 1. Without it, values can drift larger and larger as they pass up
        # the stack, which stalls learning. It also lets us train faster.
        model.add(layers.BatchNormalization(name=f"batchnorm_{i}"))

        # ReLU is the "bending" function: it passes positive numbers through
        # unchanged and turns negatives into 0. Without a bending function,
        # stacking Dense layers would be pointless — the whole network would
        # collapse mathematically into one straight line, and could not model
        # the curved load-to-NOx relationship at all.
        model.add(layers.Activation("relu", name=f"relu_{i}"))

        # Dropout randomly switches off 20% of this layer's neurons on each
        # training step. That sounds destructive, but it stops the network
        # leaning on any single neuron, forcing a more robust, spread-out
        # solution. It is our main defence against overfitting. Dropout is
        # active ONLY during training; at prediction time all neurons are used.
        model.add(layers.Dropout(DROPOUT, name=f"dropout_{i}"))

    # --- OUTPUT -----------------------------------------------------------
    # One neuron, no activation function. One neuron because we predict a
    # single number (next hour's emissions). NO activation because this is
    # regression — the output must be free to take any value. Putting a ReLU or
    # sigmoid here is a classic beginner mistake that silently caps what the
    # model can predict.
    model.add(layers.Dense(1, name="output"))

    # --- COMPILE ----------------------------------------------------------
    model.compile(
        # Adam is the standard optimiser: it adapts the step size per weight.
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        # Mean Squared Error squares each mistake before averaging, so large
        # misses are punished disproportionately. Appropriate here: badly
        # missing a high-emission hour matters more than a small everyday slip.
        loss="mse",
        # Tracked for reporting only — it does not steer training.
        metrics=["mae"],
    )
    return model


def evaluate(y_true, y_pred, persistence) -> dict:
    """Score the model, and score the naive baseline it has to beat.

    THE FOUR METRICS
    ----------------
    MAE  (Mean Absolute Error) — average size of a miss, in the real unit.
         The most intuitive number; quote this one in your abstract.
    RMSE (Root Mean Squared Error) — same idea, but squares errors first, so it
         punishes big misses harder. RMSE much larger than MAE means the model
         occasionally goes badly wrong.
    MAPE (Mean Absolute Percentage Error) — error as a percentage, so CO2 and
         NOx become comparable despite different units.
    R2   — the share of the real variation the model explains. 1.0 is perfect,
         0.0 means "no better than always guessing the average".

    THE BASELINE
    ------------
    'Persistence' predicts that the next hour equals the current hour. It uses
    no machine learning whatsoever. On smooth hourly data it is surprisingly
    strong, and a model that fails to beat it has demonstrated nothing.
    """
    def score(pred):
        mae = mean_absolute_error(y_true, pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
        # Guard against dividing by zero on the rare near-zero hour.
        safe = np.maximum(np.abs(y_true), 1e-6)
        mape = float(np.mean(np.abs((y_true - pred) / safe)) * 100)
        return {"mae": float(mae), "rmse": rmse, "mape": mape,
                "r2": float(r2_score(y_true, pred))}

    return {"model": score(y_pred), "persistence": score(persistence)}


def run_for_target(target: str, make_data_figures: bool) -> dict:
    """Train, evaluate and plot one ANN for one pollutant."""
    print(f"\n{'#'*70}\n#  {MODEL_NAME}  —  {target}\n{'#'*70}")

    # === 1. DATA =========================================================
    # All the loading, gap-aware windowing, chronological splitting and
    # scaling happens in src/data_prep.py. Read that file first if any of the
    # steps printed below are unclear.
    data = data_prep.prepare(target)

    # Figures describing the dataset itself are identical for both pollutants,
    # so we only generate them once.
    if make_data_figures:
        print("  Generating dataset figures...")
        plots.fig_target_timeline(data["df"], data_prep.TARGETS)
        plots.fig_block_lengths(data["df"], data["lookback"])
        plots.fig_correlation(
            data["df"],
            data_prep.PLANT_COLS + ["SO2 Mass (lbs)"] + data_prep.WEATHER_COLS
            + ["WD50M"],
        )
        plots.fig_daily_seasonal_profile(data["df"], data_prep.TARGETS)
        plots.fig_target_distribution(data["df"], data_prep.TARGETS)
        plots.fig_split_diagram(data)

    # === 2. BUILD ========================================================
    model = build_ann(data["lookback"], data["X_train"].shape[2])
    print(f"\n  Architecture ({model.count_params():,} trainable parameters):")
    model.summary(print_fn=lambda s: print("   ", s))

    # === 3. TRAIN ========================================================
    # Callbacks are helpers that watch training and intervene automatically.
    callbacks = [
        # EarlyStopping watches the VALIDATION loss. Once it stops improving
        # for 12 straight epochs, training halts and the best weights are
        # restored. This is what prevents overfitting: we stop at the moment
        # the model is best at generalising, not at the moment it is best at
        # memorising.
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True,
            verbose=1,
        ),
        # If progress stalls for 6 epochs, halve the learning rate. Think of it
        # as taking smaller, more careful steps as you close in on the answer.
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1,
        ),
    ]

    print(f"\n  Training (max {EPOCHS} epochs, batch size {BATCH_SIZE})...")
    t0 = time.time()
    history = model.fit(
        data["X_train"], data["y_train_scaled"],
        validation_data=(data["X_val"], data["y_val_scaled"]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        # NEVER set shuffle=False thinking it preserves time order — the
        # windows already carry their own history internally, and shuffling
        # them gives the optimiser a less biased gradient each step. The
        # chronological integrity lives in the SPLIT, not in the batch order.
        shuffle=True,
        verbose=2,
    )
    train_secs = time.time() - t0
    print(f"  Finished in {train_secs:.1f}s "
          f"after {len(history.history['loss'])} epochs")

    # === 4. PREDICT ON THE TEST SET ======================================
    # This is the first and only time the test set is used.
    y_pred_scaled = model.predict(data["X_test"], verbose=0).ravel()

    # Undo the scaling so every number below is in real short tons / lbs.
    # Reporting metrics in scaled units is meaningless — always invert first.
    y_pred = data["y_scaler"].inverse_transform(
        y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = data["y_test"]

    # === 5. SCORE ========================================================
    metrics = evaluate(y_true, y_pred, data["persistence_test"])
    m, b = metrics["model"], metrics["persistence"]
    unit = "short tons" if "CO2" in target else "lbs"

    print(f"\n  {'':22s}{MODEL_NAME:>12s}{'Persistence':>14s}{'Change':>10s}")
    print(f"  {'-'*58}")
    for key, label in [("mae", f"MAE ({unit})"), ("rmse", f"RMSE ({unit})"),
                       ("mape", "MAPE (%)"), ("r2", "R²")]:
        delta = (b[key] - m[key]) / b[key] * 100 if key != "r2" else None
        d = f"{delta:+9.1f}%" if delta is not None else ""
        print(f"  {label:22s}{m[key]:12,.3f}{b[key]:14,.3f}{d:>10s}")

    verdict = ("BEATS" if m["rmse"] < b["rmse"] else "DOES NOT BEAT")
    print(f"\n  Verdict: the {MODEL_NAME} {verdict} the persistence baseline "
          f"on RMSE.")

    # === 6. FIGURES ======================================================
    print("\n  Generating model figures...")
    plots.fig_training_history(history, target, MODEL_NAME)
    plots.fig_prediction_timeseries(data["ts_test"], y_true, y_pred, target,
                                    MODEL_NAME)
    plots.fig_scatter(y_true, y_pred, target, MODEL_NAME, m["r2"])
    plots.fig_residuals(y_true, y_pred, target, MODEL_NAME)
    plots.fig_baseline_comparison(metrics, target, MODEL_NAME)
    plots.fig_error_breakdown(data["ts_test"], y_true, y_pred, target,
                              MODEL_NAME)

    # === 7. SAVE =========================================================
    # The trained model, so you never have to retrain to make a new figure.
    slug = "co2" if "CO2" in target else "nox"
    model.save(RESULTS_DIR / f"ann_{slug}.keras")
    # The predictions, so the three architectures can be compared later
    # without rerunning any of them.
    np.savez_compressed(
        RESULTS_DIR / f"ann_{slug}_predictions.npz",
        ts=data["ts_test"], y_true=y_true, y_pred=y_pred,
        persistence=data["persistence_test"],
    )

    return {
        "target": target,
        "model": MODEL_NAME,
        "epochs_run": len(history.history["loss"]),
        "train_seconds": round(train_secs, 1),
        "parameters": int(model.count_params()),
        "n_train": int(len(data["y_train"])),
        "n_val": int(len(data["y_val"])),
        "n_test": int(len(data["y_test"])),
        "metrics": metrics,
    }


def main() -> None:
    print(f"TensorFlow {tf.__version__} | seed {SEED} | "
          f"lookback {data_prep.LOOKBACK} h | horizon {data_prep.HORIZON} h")

    results = []
    for i, target in enumerate(data_prep.TARGETS):
        results.append(run_for_target(target, make_data_figures=(i == 0)))

    # Save all metrics as JSON so the LSTM and Transformer scripts can build a
    # single comparison table at the end of the project.
    out = RESULTS_DIR / "ann_metrics.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}\nSUMMARY — {MODEL_NAME}\n{'='*70}")
    for r in results:
        m = r["metrics"]["model"]
        print(f"  {r['target']:26s} MAE {m['mae']:9,.1f}   "
              f"RMSE {m['rmse']:9,.1f}   R² {m['r2']:.4f}")
    print(f"\n  Metrics -> results/{out.name}")
    print(f"  Figures -> figures/  ({len(list(plots.FIG_DIR.glob('*.png')))} "
          f"PNG + matching PDF)")


if __name__ == "__main__":
    main()
