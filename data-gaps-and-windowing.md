# Data Gaps & Gap-Aware Windowing — Plant Bowen CEMS Modeling

**Project:** MSc thesis — comparing ANN, LSTM, and Transformer architectures for forecasting CO₂ and NOx emissions.
**Dataset:** Hourly CEMS data from Plant Bowen (ORIS 703, Georgia) via EPA CAMPD.
**Issue:** The hourly record contains gaps — hours with no rows at all, most likely unit downtime that was never recorded.

---

## 1. Why gaps matter for these models

The models never see timestamps directly. They see **sliding windows of consecutive rows** and implicitly assume row `t-1` occurred exactly one hour before row `t`.

If hours are simply absent from the file:

- **Fabricated dynamics.** A naive 24-row window can silently span several days and a full shutdown–restart cycle, while the model is told it represents 24 consecutive hours. This corrupts training data.
- **Sequence models are hit hardest.** LSTM and Transformer exist specifically to learn temporal patterns, so training them on windows with hidden time-skips undermines their core advantage. The ANN sees the same flattened windows, so it is affected too.
- **False transitions at gap edges.** A unit that went offline at ~700 MW and returned days later ramping up from low load looks, across the gap, like an impossible one-hour jump. These edges inject noise exactly where combustion behavior is least typical — startups are when NOx behaves most erratically.

## 2. What gaps do NOT hurt

- Evaluation metrics, split logic, and the validity of the three-way architecture comparison — **provided windowing is gap-aware**.
- If the gaps genuinely are downtime, the missing hours would have been excluded anyway (offline hours are filtered out / carry no meaningful emissions signal). Absent rows and filtered offline rows are equivalent from the model's perspective.

**Reframe:** the dataset is not one continuous series; it is a **collection of continuous operating blocks**.

## 3. The fix: gap-aware windowing

Never let a training window span a gap. Build windows *within* continuous blocks only.

### Algorithm

1. Sort by timestamp.
2. Compute the time difference between consecutive rows. Wherever it exceeds 1 hour, mark a block boundary.
3. Generate sliding windows within each block only.
4. Drop blocks shorter than `lookback + 1` rows (they cannot form a single window).

### Reference implementation (pandas)

```python
import pandas as pd
import numpy as np

def segment_blocks(df, ts_col="ts", expected_freq="1h"):
    """Assign a block_id that increments at every temporal gap."""
    df = df.sort_values(ts_col).reset_index(drop=True)
    gap = df[ts_col].diff() > pd.Timedelta(expected_freq)
    df["block_id"] = gap.cumsum()
    return df

def make_windows_blockwise(df, feature_cols, target_col, lookback,
                           ts_col="ts"):
    """Sliding windows X[t-lookback:t] -> y[t], never crossing a block boundary."""
    Xs, ys, label_ts = [], [], []
    for _, block in df.groupby("block_id"):
        if len(block) < lookback + 1:
            continue  # block too short to form a window
        X = block[feature_cols].values
        y = block[target_col].values
        t = block[ts_col].values
        for i in range(lookback, len(block)):
            Xs.append(X[i - lookback:i])
            ys.append(y[i])
            label_ts.append(t[i])
    return np.array(Xs), np.array(ys), np.array(label_ts)
```

### Sanity check before training — block-length report

```python
def block_report(df, ts_col="ts"):
    sizes = df.groupby("block_id").size()
    print(f"Blocks: {len(sizes)}")
    print(f"Rows total: {sizes.sum()}")
    print(f"Block length (hours): min={sizes.min()}, "
          f"median={sizes.median():.0f}, max={sizes.max()}")
    print(f"Blocks shorter than lookback+1: "
          f"{(sizes < LOOKBACK + 1).sum()} "
          f"({sizes[sizes < LOOKBACK + 1].sum()} rows unusable)")
```

If the data is fragmented into hundreds of very short blocks, the lost windows become significant — investigate before proceeding. Otherwise the cost is trivial: with a 24-hour lookback and ~40 outages, roughly ~960 windows are lost out of 20,000+.

## 4. Verify the gaps really are downtime

Before accepting gaps as outages, inspect gross load around a few gap edges:

- **Consistent with downtime:** unit ramping *down* before the gap and ramping *up* after it. → Proceed with gap-aware windowing.
- **NOT consistent with downtime:** unit at steady full load immediately before and after the gap. → This indicates a **download problem** (missing months in the CAMPD query, or a unit-selection/filter issue). Fix by re-downloading, not by windowing.

## 5. Related decisions (record these in the methodology chapter)

- **Offline vs. missing:** Rows where the unit is offline (`op_time = 0` / gross load = 0) are not missing data. Chosen approach: model **operating hours only** ("emissions given operation") — simpler and easier to defend than modeling on/off dynamics.
- **True measurement gaps are rare:** Part 75 requires high CEMS availability, and EPA substitute-data procedures fill monitor failures (biased high) before data reaches CAMPD.
- **Chronological splits still apply:** ~70/15/15 by time, scalers fitted on training data only.
- **Multiple units:** Bowen has four units; model a single unit (the one with the most operating hours) rather than summing units with different outage schedules — or state the aggregation choice explicitly.

## 6. One-sentence write-up for the thesis

> "Training windows were constructed within continuous operating periods only, so that no sample spans a monitoring gap or unit outage; days with insufficient continuous data were excluded."

---

*Reference note for Claude Code sessions: apply `segment_blocks` + `make_windows_blockwise` in any data-prep or model-training script for this project before building train/val/test sets.*
