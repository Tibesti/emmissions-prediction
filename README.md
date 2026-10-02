# Forecasting hourly CO₂ and NOx emissions from a coal-fired power plant

**Author:** Tobiloba Olugbemi · MSc research project

This MSc research project compares neural-network architectures for **one-hour-ahead forecasting of CO₂ and NOx emissions** at Georgia Power's Plant Bowen (EPA ORIS 703, Georgia, USA). It uses hourly Continuous Emissions Monitoring System (CEMS) data and on-site weather.

Two architectures are implemented and compared under identical conditions:

| Model | Idea | Script | Notebook |
|---|---|---|---|
| **ANN** (feed-forward network) | Flattens the 24-hour input window, so it has no notion of time order. This makes it the baseline architecture. | [`train_ann.py`](train_ann.py) | [`notebooks/ann.ipynb`](notebooks/ann.ipynb) |
| **Transformer** (encoder) | Self-attention over the 24 hourly steps, with positional embeddings. | [`train_transformer.py`](train_transformer.py) | [`notebooks/transformer.ipynb`](notebooks/transformer.ipynb) |
| *Comparison* | 5 seeds per model, seed ensembles, Diebold–Mariano tests | [`compare_models.py`](compare_models.py) | [`notebooks/comparison.ipynb`](notebooks/comparison.ipynb) |

An LSTM is planned as a third architecture but is not yet implemented.

---

## Contents

1. [Key results](#1-key-results)
2. [Data](#2-data)
3. [Methodology](#3-methodology)
4. [Model development and selection](#4-model-development-and-selection)
5. [Reproducing the results](#5-reproducing-the-results)
6. [Repository structure](#6-repository-structure)
7. [Limitations and threats to validity](#7-limitations-and-threats-to-validity)
8. [Data sources and acknowledgements](#8-data-sources-and-acknowledgements)

---

## 1. Key results

Test set: 11,594 hours, Dec 2024 – Mar 2026. Each model was trained with 5 seeds. Lower is better for MAE and RMSE.

| | CO₂ MAE (t/h) | CO₂ RMSE (t/h) | NOx MAE (lb/h) | NOx RMSE (lb/h) |
|---|---:|---:|---:|---:|
| **ANN**, mean ± sd of 5 seeds | 53.99 ± 0.23 | **106.07 ± 0.51** | **103.11 ± 0.59** | **249.94 ± 0.70** |
| **Transformer**, mean ± sd of 5 seeds | **53.58 ± 0.62** | 107.44 ± 1.12 | 104.73 ± 0.92 | 253.94 ± 1.73 |
| ANN, 5-seed ensemble | 53.29 | **105.10** | **101.84** | **248.73** |
| Transformer, 5-seed ensemble | **52.51** | 105.82 | 102.78 | 250.57 |
| Persistence baseline | 53.21 | 119.46 | 102.32 | 264.06 |

R² is 0.977–0.978 for CO₂ and 0.952–0.954 for NOx for both models, against 0.972 and 0.948 for persistence.

**Findings**

1. **Both models clearly beat persistence on RMSE.** The reduction is 10–12% for CO₂ and 4–6% for NOx, and the Diebold–Mariano tests give p < 10⁻⁷ for both models and both pollutants. The gain comes from **large ramps**. In the quarter of test hours with the largest hour-to-hour change, the models' MAE is 13–15% (CO₂) and 8–9% (NOx) below persistence (Figure 15).
2. **On MAE, neither model is significantly different from persistence** (DM p = 0.16–0.90). In steady operation persistence is almost exact (MAE 1.4 t/h in the steadiest quarter of hours), and the models give back what they gain on ramps.
3. **ANN vs Transformer: no architecture dominates.**
   - CO₂ MAE: the Transformer is slightly but significantly better (DM p = 0.006).
   - CO₂ RMSE: no significant difference (p = 0.23).
   - NOx: the ANN is slightly better on both metrics, but not significantly at the 5% level (p = 0.06 and 0.08).
   - The differences between architectures (≤ 1.6%) are several times smaller than the effect of the training target (13–19%, §4).
4. **Practical trade-off.** The ANN trains about 10× faster (about 40 s vs about 6 min per run on CPU) and varies less between seeds. The Transformer uses half the parameters (78k vs 148k).
5. **Ramp onsets are not anticipated.** Both models follow sudden ramps about one hour late (Figure 14). One hour ahead, the timing of dispatch changes is not predictable from past plant and weather data alone.

The full tables, significance tests and figures are in [`notebooks/comparison.ipynb`](notebooks/comparison.ipynb), [`results/comparison_summary.csv`](results/comparison_summary.csv) and [`results/comparison_runs.csv`](results/comparison_runs.csv).

---

## 2. Data

### Sources

| File (`bowen-plant-data/`) | Source | Contents |
|---|---|---|
| `hourly-emissions-facility-aggregation-*.csv` | US EPA Clean Air Markets Program Data (CAMPD), facility-level hourly emissions | Gross load (MW), heat input (mmBtu), CO₂ mass (short tons), NOx mass (lbs), SO₂ mass (lbs) |
| `POWER_Point_Hourly_*_034d12N_084d92W_LST.csv` | NASA POWER (MERRA-2 / CERES), point 34.1231 N, 84.9203 W, local standard time | Hourly wind speed and direction at 50 m, specific humidity, surface pressure, 2 m temperature, surface shortwave irradiance |
| `combined-data.csv` | Derived | The two files joined on date and hour. **This is the file the models read.** |

- **Coverage:** 01-01-2017 00:00 to 31-03-2026 23:00. There are 77,896 hourly rows out of 81,048 possible hours.
- **Missing hours:** 3,152 hours across 20 gaps, concentrated in 2019–2022. These are absent rows in the EPA file, consistent with unit outages. [`NOTES.md`](NOTES.md) lists every gap, and [`data-gaps-and-windowing.md`](data-gaps-and-windowing.md) explains how they are handled.
- **Offline hours:** 208 rows with gross load ≤ 10 MW are removed. The study models *emissions given that the plant is operating*.

`raw-data/` (third-party operational spreadsheets) is deliberately **not** in this repository. Nothing in the pipeline uses it.

### Input features (17 per hour)

| Group | Features |
|---|---|
| Plant (from past hours only) | Heat Input, Gross Load, CO₂ Mass, NOx Mass |
| Weather | `WS50M`, `QV2M`, `PS`, `T2M`, `ALLSKY_SFC_SW_DWN` |
| Wind direction, cyclical | `wd_sin`, `wd_cos` (from `WD50M`, so that 359° and 1° are neighbours) |
| Calendar, cyclical | hour-of-day, day-of-week and month-of-year, each encoded as sin/cos |

SO₂ is deliberately excluded. It is driven by coal sulphur content and scrubber performance rather than the combustion dynamics being modelled.

---

## 3. Methodology

All data handling lives in [`src/data_prep.py`](src/data_prep.py), and **every model calls the same `prepare()` function**. As a result, all architectures see identical windows, splits and scaling.

### 3.1 Forecasting task

- **Input:** the previous **24 hours** × 17 features, i.e. rows *t−24 … t−1*.
- **Output:** emissions at hour *t*, one hour ahead.
- One model is trained per pollutant.

**Leakage guard.** Hour *t* contributes nothing to the input. This matters because EPA derives CO₂ from heat input (they correlate at 1.000). If hour *t*'s heat input were an input, the model could read the answer directly.

### 3.2 Gap-aware windowing

The series is split into **continuous blocks** wherever consecutive timestamps are more than one hour apart, and windows are built **within blocks only**. No training example spans an outage. Without this, a 24-row window could silently cover several days and a shutdown–restart cycle.

This step produces 19 blocks and 77,289 windows. 39 rows in 4 blocks are too short to form a window.

### 3.3 Chronological split and scaling

| Split | Windows | Period (forecast hour) | Used for |
|---|---:|---|---|
| Train | 54,102 (70%) | 2017-01-02 → 2023-08-04 | fitting weights and all scalers |
| Validation | 11,593 (15%) | 2023-08-04 → 2024-12-03 | early stopping, learning-rate schedule, **choosing settings** |
| Test | 11,594 (15%) | 2024-12-03 → 2026-03-31 | final evaluation only |

There is no random shuffling across time. Feature and target scalers are fitted on the training split only.

### 3.4 Training target: change since the previous hour

Both models predict the **hour-to-hour change** Δₜ = yₜ − yₜ₋₁ (standardised on the training set). They do not predict the absolute level. The forecast is

  ŷₜ = yₜ₋₁ + Δ̂ₜ

So a model that outputs zero reproduces the persistence forecast exactly, and the network only has to learn departures from it. No extra information is used, because yₜ₋₁ is already the last row of the input window.

[Section 4](#4-model-development-and-selection) explains why: level-predicting models lost to persistence.

### 3.5 Baseline

**Persistence** (ŷₜ = yₜ₋₁) is the reference every model must beat. On smooth hourly emissions it is a very strong baseline, and a model that does not beat it has shown nothing.

### 3.6 Architectures

**ANN** (147,713 parameters)
- Flatten (24 × 17 = 408 inputs).
- Three hidden layers of 256, 128 and 64 units. Each is Dense → BatchNorm → ReLU → Dropout(0.2).
- Dense(1) output.

**Transformer encoder** (78,081 parameters)
- Each hour is linearly projected to 64 dimensions, and a learned positional embedding is added.
- Two pre-norm encoder blocks, each with:
  - multi-head self-attention (4 heads, key dimension 16, dropout 0.15);
  - a GELU feed-forward layer (128 units);
  - residual connections.
- Final LayerNorm.
- Pooling: the **last time step's** encoding is concatenated with the **mean over the window**.
- Dense(64, GELU) → Dropout → Dense(1).

### 3.7 Training (both models)

| Setting | ANN | Transformer |
|---|---|---|
| Optimiser | Adam, lr 1e-3 | Adam, lr 3e-4 |
| Loss | Huber (δ = 1) on the scaled change | Huber (δ = 1) on the scaled change |
| Batch size / max epochs | 128 / 120 | 128 / 100 |
| Early stopping | validation loss, patience 12, best weights restored | same |
| Learning-rate schedule | ×0.5 after 6 epochs without improvement (min 1e-6) | same |

### 3.8 Evaluation

- **Metrics:** MAE, RMSE, MAPE and R², computed on the test set in real units (short tons and lbs).
- **Seeds:** each model is trained with **5 random seeds** (42–46). Results are reported as mean ± standard deviation, and the **seed ensemble** (the average of the 5 forecasts) is scored as well.
- **Significance:** the **Diebold–Mariano test** is applied to the ensembles. It compares absolute errors and squared errors, with Newey–West variance (24 lags) because hourly errors are autocorrelated.
- **Error by regime:** test hours are grouped into quartiles of |yₜ − yₜ₋₁|, from steady operation to the largest ramps.

---

## 4. Model development and selection

This section records how the final configuration was reached, for transparency.

**The first versions predicted the absolute emission level and did not beat persistence on MAE.** Test-set results from those versions:

| Original version (level target, MSE loss) | CO₂ MAE | CO₂ RMSE | NOx MAE | NOx RMSE |
|---|---:|---:|---:|---:|
| ANN | 63.6 | 108.2 | 127.0 | 258.9 |
| Transformer | 65.4 | 113.6 | 120.3 | 263.5 |
| Persistence | 53.2 | 119.5 | 102.3 | 264.1 |

The diagnosis came from splitting test hours at the median hour-to-hour change. In the steadiest half, persistence missed by about 3 t CO₂ per hour, while both models missed by about 29. The models had to reconstruct a level of around 1,400 t from scaled inputs every hour and could not reproduce "no change" precisely. That motivated the change target (§3.4).

**The remaining settings were then chosen on the validation set** (seed 42), never on the test set:

| Candidate (all with change target) | CO₂ val MAE | CO₂ val RMSE | NOx val MAE | NOx val RMSE | Chosen |
|---|---:|---:|---:|---:|:---:|
| ANN, MSE loss | 49.05 | 91.81 | 94.09 | 209.69 | |
| **ANN, Huber loss** | **47.32** | 91.73 | **91.25** | 211.46 | ✓ |
| Transformer, mean pooling, MSE loss | 50.90 | 94.70 | 97.71 | 212.23 | |
| **Transformer, last-step + mean pooling, Huber loss** | **47.00** | 92.23 | 90.84 | 213.95 | ✓ |
| *Persistence* | 47.17 | 104.38 | 90.70 | 222.62 | |

Huber loss lowered validation MAE for the ANN at essentially unchanged RMSE. For the Transformer, last-step pooling with Huber loss beat mean pooling with MSE. The shared change target and shared loss mean the final comparison isolates architecture.

---

## 5. Reproducing the results

### Setup

Python 3.11 is required. TensorFlow runs on CPU, and no GPU is needed.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Run

```bash
.venv/bin/python train_ann.py          # ~2 min:  ANN, seed 42, data + model figures 01–12
.venv/bin/python train_transformer.py  # ~12 min: Transformer, seed 42, figures 07–12
.venv/bin/python compare_models.py     # ~70 min: 5 seeds × 2 models × 2 pollutants, figures 13–15
```

- `compare_models.py` caches each finished run in `results/comparison_runs/`. An interrupted run resumes where it stopped. Delete that folder to retrain from scratch.
- The notebooks in [`notebooks/`](notebooks/) run the same code interactively. Select the `.venv` kernel. They import the model, target and metric code from the scripts rather than copying it, so the scripts and notebooks cannot drift apart.

### Reproducibility notes

- Seeds are fixed with `keras.utils.set_random_seed`.
- TensorFlow on CPU can still differ in the last decimal places between machines and library versions. This is why results are reported over 5 seeds rather than one.
- [`requirements.txt`](requirements.txt) pins the exact library versions used for every reported number.

---

## 6. Repository structure

```
├── bowen-plant-data/          raw EPA + NASA files and the merged combined-data.csv
├── src/
│   ├── data_prep.py           the whole data pipeline (load → features → gaps → windows → split → scale → change target)
│   └── plots.py               every figure (colour-vision-deficiency-safe palette, 300 dpi PNG + vector PDF)
├── train_ann.py               ANN: build, train, evaluate, figures, save
├── train_transformer.py       Transformer: build, train, evaluate, figures, save
├── compare_models.py          multi-seed comparison, ensembles, Diebold–Mariano tests, figures 13–15
├── notebooks/                 ann.ipynb, transformer.ipynb, comparison.ipynb
├── results/                   trained models (.keras), test predictions (.npz), metrics (.json/.csv)
├── figures/                   all figures (PNG for documents, PDF for LaTeX)
├── NOTES.md                   data sources, column reference, full list of gaps
├── data-gaps-and-windowing.md rationale for gap-aware windowing
└── requirements.txt
```

### Figures

| # | Figure | Produced by |
|---|---|---|
| 01–06 | data timeline, block lengths, correlation matrix, daily/seasonal profiles, target distributions, chronological split | `train_ann.py` |
| 07–12 | per model and pollutant: training history, two-week test forecast, predicted vs actual, residuals, comparison with persistence, error by hour and level | `train_ann.py`, `train_transformer.py` |
| 13 | test MAE and RMSE per seed, ANN vs Transformer vs persistence | `compare_models.py` |
| 14 | the most volatile 4 days of the test set: actual vs both ensembles vs persistence | `compare_models.py` |
| 15 | test MAE by quartile of hour-to-hour change (steady → largest ramps) | `compare_models.py` |

### Saved models

The `.keras` models output the **scaled change**, not the emission level. To get a forecast in real units, use `forecast()` from the matching training script with the change scaler from `data_prep.change_targets()`.

---

## 7. Limitations and threats to validity

- **Strong baseline at a one-hour horizon.** Hourly emissions are highly persistent, so the margin over persistence is modest. The models' advantage is concentrated in ramping hours (Figure 15). In steady operation, persistence is close to unbeatable.
- **Ramp onsets.** On large ramps, both models largely follow the actual curve about one hour late (Figure 14). From past data alone they cannot anticipate *when* a dispatch change starts, only partly how it continues. Dispatch schedules or demand forecasts would be needed for that.
- **CO₂ is calculated, not measured.** Under 40 CFR Part 75, CO₂ mass is largely derived from heat input. CO₂ forecasting is therefore close to forecasting fuel input.
- **A single plant and a single test period** (Dec 2024 – Mar 2026). Generalisation to other plants, fuels or operating regimes is untested.
- **Facility-level aggregation.** Bowen's units are summed, so unit-level outages appear as load changes rather than as separate units.
- **Hyperparameters** were chosen on one validation split with one seed. There was no extensive search, and both architectures received comparable tuning effort.

---

## 8. Data sources and acknowledgements

- **Emissions data:** U.S. Environmental Protection Agency, Clean Air Markets Program Data (CAMPD), https://campd.epa.gov/
- **Weather data:** these data were obtained from the NASA Langley Research Center (LaRC) POWER Project, funded through the NASA Earth Science/Applied Science Program, https://power.larc.nasa.gov/
