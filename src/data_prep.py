"""
data_prep.py — Turning the raw CSV into training-ready arrays.
================================================================

WHY THIS FILE EXISTS
--------------------
Three models will be compared in this project: an ANN, an LSTM, and a
Transformer. For that comparison to mean anything, all three MUST be fed
*exactly* the same data, split at exactly the same points in time. If the ANN
saw slightly different rows than the LSTM, you could never tell whether a
difference in scores came from the architecture or from the data.

So every model imports its data from this one file. Nothing here is
model-specific.

READ THIS FILE TOP TO BOTTOM — it follows the order the data actually flows:
    load CSV -> add features -> keep operating hours -> find gaps ->
    cut into windows -> split by time -> scale numbers
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# CONFIGURATION — the knobs you are most likely to change
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "bowen-plant-data" / "combined-data.csv"

# How many past hours the model looks at to make one prediction.
# 24 = "look at the last full day to predict the next hour".
LOOKBACK = 24

# How far ahead we predict. 1 = the very next hour.
HORIZON = 1

# A row is treated as "the plant was running" only if gross load is above this.
# The plant's normal output is ~1400 MW, so anything at or below 10 MW is
# effectively off. See NOTES.md for why we model operating hours only.
OPERATING_LOAD_THRESHOLD = 10.0

# How to divide the timeline into train / validation / test.
# These are FRACTIONS OF TIME, not random samples — see split_chronologically().
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test gets the remaining 0.15

# The two things we are trying to predict. Each gets its own trained model, so
# every script in this project runs twice — once per target.
TARGETS = ["CO2 Mass (short tons)", "NOx Mass (lbs)"]

# Columns that describe what the plant was doing. These are used as INPUTS from
# past hours only (never from the hour we are predicting) — see build_windows().
#
# SO2 Mass is deliberately EXCLUDED as an input (project decision). It is a
# third pollutant driven by coal sulphur content and scrubber performance
# rather than by the combustion dynamics we are modelling, so it is neither a
# predictor we want nor a target we report. It stays in the CSV, unused.
#
# Past CO2 and NOx ARE inputs. That is not leakage: they are the target's own
# recorded history up to hour t-1, which any real forecaster would have on
# hand. This is what makes the task autoregressive, and it is standard for
# emissions forecasting.
PLANT_COLS = [
    "Heat Input (mmBtu)",
    "Gross Load (MW)",
    "CO2 Mass (short tons)",
    "NOx Mass (lbs)",
]

# Weather columns straight from NASA POWER.
# NOTE: WD50M (wind direction) is deliberately NOT in this list. Wind direction
# is an angle, and angles break normal maths: 359 degrees and 1 degree are two
# degrees apart in reality but 358 apart numerically. A neural network fed raw
# degrees would learn nonsense at the north wrap-around. We convert it into two
# ordinary numbers (sine and cosine) in add_engineered_features() instead.
WEATHER_COLS = [
    "WS50M",
    "QV2M",
    "PS",
    "T2M",
    "ALLSKY_SFC_SW_DWN",
]


# ----------------------------------------------------------------------------
# STEP 1 — LOAD THE CSV AND BUILD A REAL TIMESTAMP
# ----------------------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    """Read combined-data.csv and give every row a proper datetime.

    The CSV stores time as two separate columns: a 'Date' string like
    "01-01-2017" and an integer 'Hour' from 0-23. Neither is usable on its own
    for sorting or for measuring the distance between two rows, so we glue them
    into a single pandas Timestamp column called 'ts'.
    """
    df = pd.read_csv(DATA_FILE)

    # format="%d-%m-%Y" tells pandas the date is DAY-MONTH-YEAR. Without this
    # it might read 03-04-2017 as March 4th instead of April 3rd.
    df["ts"] = (
        pd.to_datetime(df["Date"], format="%d-%m-%Y")
        + pd.to_timedelta(df["Hour"], unit="h")
    )

    # Sorting is not optional. Everything downstream — gap detection, the
    # chronological split — assumes the rows run oldest to newest.
    df = df.sort_values("ts").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# STEP 2 — CREATE EXTRA INPUT COLUMNS THE MODEL CAN ACTUALLY USE
# ----------------------------------------------------------------------------

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time and wind-direction features.

    A neural network has no idea what a clock or a calendar is. If you hand it
    the number 23 for 11pm and 0 for midnight, it sees a huge jump, when really
    those two hours are adjacent. The standard fix is to place each cyclical
    value on a circle and hand over its x and y coordinates:

        angle = 2 * pi * (value / period)
        x = cos(angle)   y = sin(angle)

    Now hour 23 and hour 0 sit right next to each other on the circle, which is
    the truth we want the model to see. We do this for hour-of-day, day-of-week
    and month-of-year, and for wind direction.
    """
    df = df.copy()

    def cyclical(values: pd.Series, period: int, name: str) -> None:
        """Write two columns, <name>_sin and <name>_cos, into df."""
        angle = 2.0 * np.pi * values / period
        df[f"{name}_sin"] = np.sin(angle)
        df[f"{name}_cos"] = np.cos(angle)

    # Hour of day: captures the daily demand cycle (low at 4am, peak late
    # afternoon). Period 24 because the pattern repeats every 24 hours.
    cyclical(df["ts"].dt.hour, 24, "hour")

    # Day of week: captures weekday vs weekend electricity demand.
    cyclical(df["ts"].dt.dayofweek, 7, "dow")

    # Month of year: captures summer/winter demand and the fact that NOx
    # formation is temperature-dependent.
    cyclical(df["ts"].dt.month, 12, "month")

    # Wind direction, in degrees, converted to a point on a circle as explained
    # in the docstring above.
    cyclical(df["WD50M"], 360, "wd")

    return df


def feature_columns() -> list[str]:
    """The final, ordered list of input columns fed to every model.

    The order is fixed so that column 3 means the same thing in the ANN, the
    LSTM and the Transformer. Never shuffle this list.
    """
    return (
        PLANT_COLS
        + WEATHER_COLS
        + ["wd_sin", "wd_cos",
           "hour_sin", "hour_cos",
           "dow_sin", "dow_cos",
           "month_sin", "month_cos"]
    )


# ----------------------------------------------------------------------------
# STEP 3 — KEEP ONLY THE HOURS THE PLANT WAS ACTUALLY RUNNING
# ----------------------------------------------------------------------------

def filter_operating_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Drop hours where the plant was offline.

    When gross load is 0, emissions are 0 too. Those rows teach the model
    nothing about combustion — they only teach it "sometimes everything is
    zero", which drags predictions down and inflates the R-squared score for
    the wrong reason. The project models 'emissions GIVEN the plant is
    operating', which is the simpler and more defensible claim.

    This removes ~208 of 77,896 rows (about 0.3%).
    """
    keep = df["Gross Load (MW)"] > OPERATING_LOAD_THRESHOLD
    return df.loc[keep].reset_index(drop=True)


# ----------------------------------------------------------------------------
# STEP 4 — FIND THE GAPS AND CUT THE DATA INTO CONTINUOUS BLOCKS
# ----------------------------------------------------------------------------

def segment_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Label every row with a 'block_id' that increases at each time gap.

    THIS IS THE MOST IMPORTANT FUNCTION IN THE FILE.

    The dataset is missing 3,152 hours across 20 outage gaps (see NOTES.md).
    Those hours are not blank rows — they are simply absent. So two rows that
    sit next to each other in the file can be weeks apart in real time.

    A model never sees timestamps. It sees a stack of 24 rows and is told
    "these are 24 consecutive hours". If a gap hides inside that stack, we are
    lying to the model: it would learn that the plant can drop from 700 MW to 0
    and back in one hour, which never happened.

    The fix: compare each row's timestamp with the previous row's. Wherever the
    difference is more than one hour, we have found the seam between two
    separate 'blocks' of continuous operation. cumsum() then turns that
    True/False seam marker into a running block number.

        diff:    1h  1h  1h  580h  1h  1h
        is_gap:   F   F   F     T   F   F
        cumsum:   0   0   0     1   1   1     <- block_id
    """
    df = df.copy()

    # .diff() gives the time elapsed since the previous row. The first row has
    # no previous row, so it produces NaT (not-a-time), which compares as False
    # — exactly what we want, since row 0 starts block 0.
    is_gap = df["ts"].diff() > pd.Timedelta("1h")
    df["block_id"] = is_gap.cumsum()
    return df


def block_report(df: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    """Print a summary of the blocks — run this before trusting any results.

    If the data turned out to be shattered into hundreds of tiny fragments,
    most of it could not form a single training window and the whole approach
    would need rethinking. This is the sanity check that proves it did not.
    """
    sizes = df.groupby("block_id").size()
    too_short = sizes < lookback + HORIZON
    print("  Continuous blocks      :", len(sizes))
    print("  Rows total             :", int(sizes.sum()))
    print(f"  Block length (hours)   : min={sizes.min()}, "
          f"median={sizes.median():.0f}, max={sizes.max()}")
    print(f"  Blocks too short to use: {int(too_short.sum())} "
          f"({int(sizes[too_short].sum())} rows unusable)")
    return sizes


# ----------------------------------------------------------------------------
# STEP 5 — TURN ROWS INTO SLIDING WINDOWS (THE ACTUAL TRAINING EXAMPLES)
# ----------------------------------------------------------------------------

def build_windows(df: pd.DataFrame, target: str, lookback: int = LOOKBACK):
    """Convert the table into (X, y) pairs, never crossing a block boundary.

    One training example looks like this, for lookback=24:

        X  =  rows t-24 ... t-1   (24 rows x 17 columns of numbers)
        y  =  the target value at row t   (a single number)

    Read that carefully, because it is what protects this project from a fatal
    mistake. The inputs stop at t-1. The hour being predicted, t, contributes
    NOTHING to the inputs.

    Why that matters here specifically: EPA does not measure CO2 directly, it
    calculates it from heat input. In this dataset CO2 and Heat Input correlate
    at 1.000 — perfectly. If hour t's heat input were allowed into X, the model
    could read the answer off its own input sheet and score near-perfectly
    while having learned nothing. That is called data leakage, and it is the
    single most common way a thesis model gets impressive-but-worthless
    results. Stopping the window at t-1 makes it a genuine forecast.

    Returns
    -------
    X   : array, shape (n_samples, lookback, n_features)
    y   : array, shape (n_samples,)
    ts  : array of timestamps, one per sample — the hour being predicted
    last_obs : array — the target's value at t-1, used later as the
               'persistence' baseline to compare the model against
    """
    feats = feature_columns()
    X_list, y_list, ts_list, last_list = [], [], [], []

    # Handle each continuous block completely separately. A window is never
    # allowed to start in one block and end in another.
    for _, block in df.groupby("block_id"):
        if len(block) < lookback + HORIZON:
            continue  # too short to yield even one window — skip it

        Xb = block[feats].to_numpy(dtype="float32")
        yb = block[target].to_numpy(dtype="float32")
        tb = block["ts"].to_numpy()

        # Slide the window forward one hour at a time.
        for i in range(lookback, len(block) - HORIZON + 1):
            t = i + HORIZON - 1          # index of the hour being predicted
            X_list.append(Xb[i - lookback:i])   # the 24 hours BEFORE it
            y_list.append(yb[t])
            ts_list.append(tb[t])
            last_list.append(yb[i - 1])  # most recent known value = baseline

    return (
        np.asarray(X_list, dtype="float32"),
        np.asarray(y_list, dtype="float32"),
        np.asarray(ts_list),
        np.asarray(last_list, dtype="float32"),
    )


# ----------------------------------------------------------------------------
# STEP 6 — SPLIT INTO TRAIN / VALIDATION / TEST *BY TIME*
# ----------------------------------------------------------------------------

def split_chronologically(n: int):
    """Return index ranges for train / val / test, in time order.

    DO NOT use sklearn's train_test_split here. That shuffles rows at random,
    which for time-series data means the model trains on Wednesday and is
    tested on Tuesday — it gets to see the future. Scores come out beautiful
    and completely fake.

    Instead: the oldest 70% trains the model, the next 15% tunes it (early
    stopping), and the newest 15% is touched exactly once, at the very end.
    That mirrors how the model would really be used — trained on the past,
    asked about the future.
    """
    i_train = int(n * TRAIN_FRAC)
    i_val = int(n * (TRAIN_FRAC + VAL_FRAC))
    return slice(0, i_train), slice(i_train, i_val), slice(i_val, n)


# ----------------------------------------------------------------------------
# STEP 7 — SCALE THE NUMBERS
# ----------------------------------------------------------------------------

def scale_features(X_train, X_val, X_test):
    """Put every input column on a comparable scale (mean 0, std 1).

    Why bother: the raw columns have wildly different magnitudes. Heat Input
    runs to ~35,000 while wind speed sits around 4. A neural network updates
    its weights using gradients, and huge-magnitude inputs produce huge
    gradients that swamp the small ones — the model would effectively ignore
    wind speed entirely. Standardising removes that unfairness.

    CRITICAL DETAIL: the scaler is FIT on the training set only, then APPLIED
    to validation and test. If you fit it on everything, the training data
    silently absorbs knowledge of the test set's mean and spread — leakage
    again, subtler but still wrong.

    The arrays are 3-D (samples x hours x features) and StandardScaler wants
    2-D, so we temporarily flatten the first two dimensions, scale, and put the
    shape back.
    """
    n_features = X_train.shape[2]
    scaler = StandardScaler()

    scaler.fit(X_train.reshape(-1, n_features))

    def apply(A):
        return scaler.transform(A.reshape(-1, n_features)).reshape(A.shape)

    return apply(X_train), apply(X_val), apply(X_test), scaler


def scale_target(y_train, y_val, y_test):
    """Standardise the target too, and return the scaler so we can undo it.

    Training is more stable when the target is around 0 rather than around
    1,400. But every metric and every chart must be reported in the REAL unit
    (short tons, lbs), so we keep the scaler and call inverse_transform on the
    predictions before scoring. Never report a metric in scaled units.
    """
    scaler = StandardScaler()
    scaler.fit(y_train.reshape(-1, 1))

    def apply(v):
        return scaler.transform(v.reshape(-1, 1)).ravel()

    return apply(y_train), apply(y_val), apply(y_test), scaler


# ----------------------------------------------------------------------------
# STEP 8 — THE TRAINING TARGET: CHANGE SINCE THE LAST HOUR
# ----------------------------------------------------------------------------

def change_targets(data: dict):
    """Return the scaled hour-to-hour change for train/val, plus its scaler.

    Hourly emissions are highly persistent, so "next hour = this hour" is a
    strong baseline. A model asked to predict the absolute level (~1,400 tons
    of CO2) has to rebuild that number from scaled inputs every hour, and in
    steady operation it missed by ~29 tons where persistence missed by ~3.
    So every model predicts the CHANGE  y_t - y_{t-1}  instead, and the
    forecast is  y_{t-1} + predicted change  (see level_from_change). An
    output of zero reproduces persistence exactly; the network only has to
    learn the departures from it.

    This uses no extra information: y_{t-1} is already the last row of every
    input window. The scaler is fitted on the training changes only.
    """
    change_train = data["y_train"] - data["persistence_train"]
    change_val = data["y_val"] - data["persistence_val"]
    scaler = StandardScaler().fit(change_train.reshape(-1, 1))
    return (
        scaler.transform(change_train.reshape(-1, 1)).ravel(),
        scaler.transform(change_val.reshape(-1, 1)).ravel(),
        scaler,
    )


def level_from_change(change_scaled, persistence, scaler) -> np.ndarray:
    """Turn a model's scaled change prediction into a forecast in real units."""
    change = scaler.inverse_transform(np.reshape(change_scaled, (-1, 1))).ravel()
    return persistence + change


# ----------------------------------------------------------------------------
# THE ONE FUNCTION THE MODEL SCRIPTS ACTUALLY CALL
# ----------------------------------------------------------------------------

def prepare(target: str, lookback: int = LOOKBACK, verbose: bool = True) -> dict:
    """Run steps 1-7 and hand back everything a model needs.

    Call this from train_ann.py, and later from the LSTM and Transformer
    scripts, so all three are guaranteed to train on identical data.
    """
    if verbose:
        print(f"\n{'='*70}\nPREPARING DATA FOR TARGET: {target}\n{'='*70}")

    df = load_raw()
    if verbose:
        print(f"\n[1] Loaded {len(df):,} hourly rows "
              f"({df['ts'].min():%Y-%m-%d} to {df['ts'].max():%Y-%m-%d})")

    df = add_engineered_features(df)
    if verbose:
        print(f"[2] Added cyclical time + wind-direction features "
              f"-> {len(feature_columns())} input columns")

    before = len(df)
    df = filter_operating_hours(df)
    if verbose:
        print(f"[3] Kept operating hours only: {len(df):,} rows "
              f"({before - len(df)} offline rows removed)")

    df = segment_blocks(df)
    if verbose:
        print("[4] Gap-aware segmentation:")
        block_report(df, lookback)

    X, y, ts, last_obs = build_windows(df, target, lookback)
    if verbose:
        print(f"[5] Built {len(X):,} windows of shape "
              f"({lookback} hours x {X.shape[2]} features)")

    tr, va, te = split_chronologically(len(X))
    if verbose:
        print(f"[6] Chronological split — "
              f"train {tr.stop - tr.start:,} | "
              f"val {va.stop - va.start:,} | "
              f"test {te.stop - te.start:,}")
        print(f"    train ends {pd.Timestamp(ts[tr.stop - 1]):%Y-%m-%d}, "
              f"test starts {pd.Timestamp(ts[te.start]):%Y-%m-%d}")

    Xtr_s, Xva_s, Xte_s, x_scaler = scale_features(X[tr], X[va], X[te])
    ytr_s, yva_s, yte_s, y_scaler = scale_target(y[tr], y[va], y[te])
    if verbose:
        print("[7] Scaled features and target (fitted on training data only)\n")

    return {
        "target": target,
        "lookback": lookback,
        "feature_names": feature_columns(),
        # scaled — what the model trains on
        "X_train": Xtr_s, "X_val": Xva_s, "X_test": Xte_s,
        "y_train_scaled": ytr_s, "y_val_scaled": yva_s, "y_test_scaled": yte_s,
        # real units — what we score and plot
        "y_train": y[tr], "y_val": y[va], "y_test": y[te],
        "ts_train": ts[tr], "ts_val": ts[va], "ts_test": ts[te],
        # the target's value at t-1 for every window. On the test set this is
        # the naive baseline every model must beat; on train/val it lets a
        # model learn the hour-to-hour change instead of the absolute level.
        "persistence_train": last_obs[tr],
        "persistence_val": last_obs[va],
        "persistence_test": last_obs[te],
        "y_scaler": y_scaler, "x_scaler": x_scaler,
        "df": df,
    }
