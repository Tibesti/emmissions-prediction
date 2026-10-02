"""
compare_models.py — ANN vs Transformer, over several training seeds.

A single training run is partly luck: a different random initialisation can
move the test MAE by a few percent, which is about the size of the gap between
the two architectures. So each model is trained once per seed in SEEDS, and
the comparison reports:

  * mean ± standard deviation of every test metric across seeds,
  * the seed ENSEMBLE (average of the per-seed forecasts) for each model,
  * a Diebold-Mariano test of whether the ensembles' test errors differ.

Both models use exactly the same data, windows, splits, training target (the
hour-to-hour change) and loss, so any difference is down to architecture.

Each finished run is cached in results/comparison_runs/, so an interrupted
comparison resumes where it stopped. Delete that folder to retrain everything.

Run with:
    .venv/bin/python compare_models.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import data_prep  # noqa: E402
import plots  # noqa: E402
import train_ann  # noqa: E402
import train_transformer  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = RESULTS_DIR / "comparison_runs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
MODELS = {"ANN": train_ann, "Transformer": train_transformer}

# Newey-West lags for the Diebold-Mariano variance. Forecast errors on hourly
# data are autocorrelated (a bad hour tends to be followed by another), and
# ignoring that would overstate significance. 24 lags covers one daily cycle.
DM_LAGS = 24


def slug(target: str) -> str:
    return "co2" if "CO2" in target else "nox"


def run_one(module, model_name: str, data: dict, seed: int) -> dict:
    """Train one model with one seed, or load the cached result."""
    path = CACHE_DIR / f"{model_name.lower()}_{slug(data['target'])}_seed{seed}.npz"
    if path.exists():
        cached = np.load(path)
        return {key: cached[key] for key in cached.files}

    print(f"  training {model_name} seed {seed}...", flush=True)
    model, history, change_scaler, seconds = module.train(data, seed=seed, verbose=0)
    run = {
        "val_pred": module.forecast(model, data, change_scaler, "val"),
        "test_pred": module.forecast(model, data, change_scaler, "test"),
        "epochs": np.array(len(history.history["loss"])),
        "seconds": np.array(seconds),
    }
    np.savez_compressed(path, **run)
    return run


def score(y_true, y_pred) -> dict:
    return train_ann.evaluate(y_true, y_pred, y_true)["model"]


def diebold_mariano(y_true, pred_a, pred_b, power: int) -> dict:
    """Test H0: forecasts A and B are equally accurate.

    power=1 compares absolute errors (MAE), power=2 squared errors (MSE).
    A negative statistic means A is more accurate than B.
    """
    d = np.abs(y_true - pred_a) ** power - np.abs(y_true - pred_b) ** power
    n = len(d)
    centred = d - d.mean()
    variance = centred @ centred / n
    for lag in range(1, DM_LAGS + 1):
        weight = 1 - lag / (DM_LAGS + 1)
        variance += 2 * weight * (centred[lag:] @ centred[:-lag]) / n
    statistic = d.mean() / np.sqrt(variance / n)
    return {"statistic": float(statistic),
            "p_value": float(2 * (1 - norm.cdf(abs(statistic))))}


def compare_target(target: str) -> tuple[list[dict], dict]:
    print(f"\n{'#' * 70}\n#  {target}\n{'#' * 70}")
    data = data_prep.prepare(target, verbose=False)
    y_val, y_test = data["y_val"], data["y_test"]

    rows, ensembles = [], {}
    for model_name, module in MODELS.items():
        val_preds, test_preds = [], []
        for seed in SEEDS:
            run = run_one(module, model_name, data, seed)
            val_preds.append(run["val_pred"])
            test_preds.append(run["test_pred"])
            rows.append({
                "target": target, "model": model_name, "seed": seed,
                "epochs": int(run["epochs"]), "seconds": float(run["seconds"]),
                **{f"val_{k}": v for k, v in score(y_val, run["val_pred"]).items()},
                **score(y_test, run["test_pred"]),
            })
        ensembles[model_name] = {
            "val": np.mean(val_preds, axis=0),
            "test": np.mean(test_preds, axis=0),
        }

    persistence = data["persistence_test"]
    summary = {
        "persistence": score(y_test, persistence),
        "persistence_val": score(y_val, data["persistence_val"]),
        "ensemble": {name: score(y_test, e["test"]) for name, e in ensembles.items()},
        "ensemble_val": {name: score(y_val, e["val"]) for name, e in ensembles.items()},
        "diebold_mariano": {},
    }
    pairs = [("ANN", "Transformer"), ("ANN", "Persistence"), ("Transformer", "Persistence")]
    forecasts = {**{n: e["test"] for n, e in ensembles.items()}, "Persistence": persistence}
    for a, b in pairs:
        summary["diebold_mariano"][f"{a} vs {b}"] = {
            "absolute_error": diebold_mariano(y_test, forecasts[a], forecasts[b], 1),
            "squared_error": diebold_mariano(y_test, forecasts[a], forecasts[b], 2),
        }

    np.savez_compressed(
        RESULTS_DIR / f"comparison_{slug(target)}_predictions.npz",
        ts=data["ts_test"], y_true=y_test, persistence=persistence,
        **{f"{name.lower()}_ensemble": e["test"] for name, e in ensembles.items()},
    )
    plots.fig_comparison_timeseries(
        data["ts_test"], y_test, persistence,
        {name: e["test"] for name, e in ensembles.items()}, target,
    )
    summary["_regime"] = {"y_true": y_test, "persistence": persistence,
                          "preds": {n: e["test"] for n, e in ensembles.items()}}
    return rows, summary


def print_summary(runs: pd.DataFrame, summaries: dict) -> pd.DataFrame:
    table = []
    for target, summary in summaries.items():
        for model_name in MODELS:
            subset = runs[(runs["target"] == target) & (runs["model"] == model_name)]
            for metric in ["mae", "rmse", "mape", "r2"]:
                table.append({
                    "target": target, "model": model_name, "metric": metric,
                    "mean": subset[metric].mean(), "std": subset[metric].std(),
                    "best_seed": subset[metric].min() if metric != "r2" else subset[metric].max(),
                    "ensemble": summary["ensemble"][model_name][metric],
                    "persistence": summary["persistence"][metric],
                })
    table = pd.DataFrame(table)

    print(f"\n{'=' * 70}\nSUMMARY — test set, {len(SEEDS)} seeds per model\n{'=' * 70}")
    for target, summary in summaries.items():
        print(f"\n  {target}")
        print(f"  {'':12s}{'MAE mean±std':>18s}{'ens.':>8s}{'RMSE mean±std':>20s}{'ens.':>8s}")
        for model_name in MODELS:
            t = table[(table["target"] == target) & (table["model"] == model_name)]
            mae, rmse = t[t["metric"] == "mae"].iloc[0], t[t["metric"] == "rmse"].iloc[0]
            print(f"  {model_name:12s}{mae['mean']:11,.2f} ± {mae['std']:4.2f}"
                  f"{mae['ensemble']:8,.2f}{rmse['mean']:13,.2f} ± {rmse['std']:4.2f}"
                  f"{rmse['ensemble']:8,.2f}")
        p = summary["persistence"]
        print(f"  {'Persistence':12s}{p['mae']:18,.2f}{'':8s}{p['rmse']:20,.2f}")
        print("  Diebold-Mariano (ensembles; negative = first is more accurate):")
        for pair, tests in summary["diebold_mariano"].items():
            a, s = tests["absolute_error"], tests["squared_error"]
            print(f"    {pair:28s} |error| DM {a['statistic']:+6.2f} (p={a['p_value']:.3g})"
                  f"   error² DM {s['statistic']:+6.2f} (p={s['p_value']:.3g})")
    return table


def main() -> None:
    print(f"Seeds {SEEDS} | models {list(MODELS)} | cache {CACHE_DIR}")
    all_rows, summaries = [], {}
    for target in data_prep.TARGETS:
        rows, summaries[target] = compare_target(target)
        all_rows += rows
    runs = pd.DataFrame(all_rows)

    plots.fig_model_comparison(
        runs, {t: s["persistence"] for t, s in summaries.items()}
    )
    plots.fig_error_by_regime({t: s.pop("_regime") for t, s in summaries.items()})

    table = print_summary(runs, summaries)
    runs.to_csv(RESULTS_DIR / "comparison_runs.csv", index=False)
    table.to_csv(RESULTS_DIR / "comparison_summary.csv", index=False)
    (RESULTS_DIR / "comparison_summary.json").write_text(json.dumps(summaries, indent=2))
    print("\n  Results -> results/comparison_runs.csv, comparison_summary.csv/.json")
    print("  Figures -> figures/13_model_comparison, 14_*_comparison_timeseries, "
          "15_error_by_regime")


if __name__ == "__main__":
    main()
