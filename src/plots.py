"""
plots.py — Every figure this project produces.
==============================================

All figures are saved to figures/ as 300 dpi PNG (print-quality for a thesis)
and as PDF (vector, so it stays sharp at any zoom in the final document).

COLOUR CHOICES ARE NOT DECORATION. The three series colours below are from a
palette validated for colour-vision deficiency, so a reader with deuteranopia
or protanopia can still tell the lines apart — and so can anyone who prints
your thesis in greyscale, because the colours also differ in lightness. Every
chart additionally carries a legend or a direct label, so identity is never
communicated by colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render to file, never try to open a window
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# --- the validated palette -------------------------------------------------
C_ACTUAL = "#2a78d6"   # blue    — measured / actual values
C_PRED = "#eb6834"     # orange  — model predictions
C_BASE = "#1baf7a"     # aqua    — persistence baseline
C_ACCENT = "#4a3aa7"   # violet  — secondary highlights
C_WARN = "#e34948"     # red     — errors, gaps, things to notice

INK = "#0b0b0b"        # primary text
INK_2 = "#52514e"      # secondary text (axis labels, captions)
GRID = "#d9d8d4"       # recessive grid — must never compete with the data

# Diverging colormap for correlations: two hues with a NEUTRAL GREY midpoint.
# Never use a rainbow for this — rainbows invent boundaries that aren't in the
# data and are unreadable in greyscale.
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "corr", [C_ACTUAL, "#f2f1ee", C_WARN]
)
# Sequential colormap: ONE hue, light to dark. For magnitude only.
CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "mag", ["#eaf2fc", C_ACTUAL, "#123a66"]
)


def _style() -> None:
    """Global matplotlib defaults, applied once when this module is imported."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,      # grid sits BEHIND the data, never on top
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,
        "font.size": 10,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })
    # Remove the top and right box lines — they add ink without adding meaning.
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


_style()


def _save(fig, name: str) -> Path:
    """Write one figure to PNG (for slides/Word) and PDF (for LaTeX)."""
    png = FIG_DIR / f"{name}.png"
    fig.savefig(png)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"    saved figures/{name}.png")
    return png


def _slug(target: str) -> str:
    """'CO2 Mass (short tons)' -> 'co2' so filenames stay clean."""
    return "co2" if "CO2" in target else "nox"


def _unit(target: str) -> str:
    return "short tons" if "CO2" in target else "lbs"


# ===========================================================================
# PART A — FIGURES THAT DESCRIBE THE DATA
# These belong in your Data / Methodology chapter. They are produced once,
# not once per model.
# ===========================================================================

def fig_target_timeline(df: pd.DataFrame, targets: list[str]) -> None:
    """Full 2017-2026 history of both pollutants, with outage gaps marked.

    WHAT IT SHOWS YOUR EXAMINER: the scale of the series, its seasonality, the
    long-run decline in output, and — critically — exactly where the 20 data
    gaps fall. Because the x-axis is real time, the gaps appear as blank
    stretches rather than being hidden by the line joining across them.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    for ax, target, colour in zip(axes, targets, [C_ACTUAL, C_ACCENT]):
        # Plot each continuous block separately so matplotlib does NOT draw a
        # straight line across a three-week outage. That line would be a
        # fabricated measurement.
        for _, blk in df.groupby("block_id"):
            ax.plot(blk["ts"], blk[target], color=colour, linewidth=0.35)
        ax.set_ylabel(f"{target.split(' Mass')[0]}\n({_unit(target)}/hour)")
        ax.set_title(f"{target} — hourly, operating hours only", loc="left")

    # Shade the gaps in red so they are impossible to miss.
    edges = df.groupby("block_id")["ts"].agg(["min", "max"]).sort_values("min")
    for i in range(1, len(edges)):
        start = edges.iloc[i - 1]["max"]
        end = edges.iloc[i]["min"]
        if (end - start) > pd.Timedelta("24h"):  # only label the big ones
            for ax in axes:
                ax.axvspan(start, end, color=C_WARN, alpha=0.16, linewidth=0)

    axes[0].plot([], [], color=C_WARN, alpha=0.4, linewidth=8,
                 label="data gap (>24 h)")
    axes[0].legend(loc="upper right")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Year")
    fig.suptitle("Plant Bowen hourly emissions, 2017–2026",
                 fontsize=13, fontweight="bold", x=0.09, ha="left")
    fig.tight_layout()
    _save(fig, "01_data_timeline")


def fig_block_lengths(df: pd.DataFrame, lookback: int) -> None:
    """How long each continuous operating block is.

    WHAT IT SHOWS: that the gap-aware windowing costs you almost nothing. If
    the bars piled up on the left (lots of very short blocks) you would be
    throwing away most of your data. They don't — so the method is cheap.
    """
    sizes = df.groupby("block_id").size().sort_values(ascending=False)
    minimum = lookback + 1
    usable = sizes >= minimum

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(sizes)), sizes.values, width=0.75,
           color=np.where(usable, C_ACTUAL, GRID))
    ax.axhline(minimum, color=C_WARN, linestyle="--", linewidth=1.4)
    ax.text(len(sizes) - 0.5, minimum * 2.0,
            f"minimum {minimum} h",
            color=C_WARN, fontsize=9, ha="right")
    ax.set_yscale("log")  # blocks range from a few hours to several years
    ax.set_xticks(range(len(sizes)), range(1, len(sizes) + 1))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Continuous operating block (longest to shortest)")
    ax.set_ylabel("Length (hours, log scale)")
    ax.set_title(f"{int(usable.sum())} of {len(sizes)} blocks are long enough to "
                 f"form a window; the {int((~usable).sum())} grey ones hold only "
                 f"{int(sizes[~usable].sum())} hours", loc="left")
    _save(fig, "02_block_lengths")


def fig_correlation(df: pd.DataFrame, cols: list[str]) -> None:
    """Correlation between every variable pair.

    WHAT IT SHOWS — and this is a key result for your write-up: Heat Input and
    CO2 sit at 1.00. EPA computes CO2 from heat input via a fixed carbon
    factor, so they are the same measurement twice. That is exactly why the
    input window must stop at hour t-1; you can point at this cell to justify
    the decision.
    """
    corr = df[cols].corr()
    labels = [c.replace(" Mass", "").replace(" (short tons)", "")
               .replace(" (lbs)", "").replace(" (mmBtu)", "")
               .replace(" (MW)", "") for c in cols]

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr, cmap=CMAP_DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), labels)
    ax.grid(False)

    # Direct labels: the number in every cell, so the figure is readable
    # without decoding the colour bar — and works in greyscale.
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.55 else INK)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson correlation", color=INK_2)
    cbar.outline.set_visible(False)
    ax.set_title("Correlation between plant variables and weather\n"
                 "(operating hours only)", loc="left")
    _save(fig, "03_correlation_matrix")


def fig_daily_seasonal_profile(df: pd.DataFrame, targets: list[str]) -> None:
    """Average emissions by hour of day and by month.

    WHAT IT SHOWS: that the cyclical hour/month features added in data_prep are
    justified — there IS a real daily and seasonal pattern for the model to
    learn. If these lines were flat, those features would be dead weight.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    hours = df["ts"].dt.hour
    months = df["ts"].dt.month

    for target, colour in zip(targets, [C_ACTUAL, C_ACCENT]):
        # Normalise to a percentage of each series' own mean, so two pollutants
        # with totally different units share ONE y-axis. This is the correct
        # alternative to a dual-axis chart, which is never acceptable.
        name = target.split(" Mass")[0]
        by_h = df.groupby(hours)[target].mean()
        by_m = df.groupby(months)[target].mean()
        axes[0].plot(by_h.index, 100 * by_h / by_h.mean(), color=colour,
                     marker="o", markersize=3.5, label=name)
        axes[1].plot(by_m.index, 100 * by_m / by_m.mean(), color=colour,
                     marker="o", markersize=3.5, label=name)

    for ax, xlabel, title in [
        (axes[0], "Hour of day (local)", "Daily cycle"),
        (axes[1], "Month", "Seasonal cycle"),
    ]:
        ax.axhline(100, color=GRID, linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean emissions\n(% of annual average)")
        ax.set_title(title, loc="left")
        ax.legend()
    axes[1].set_xticks(range(1, 13))
    fig.tight_layout()
    _save(fig, "04_daily_seasonal_profile")


def fig_target_distribution(df: pd.DataFrame, targets: list[str]) -> None:
    """How the two pollutants are distributed.

    WHAT IT SHOWS: whether the target is skewed. A long right tail means rare
    high-emission hours, and those are exactly the hours a model tends to get
    wrong — worth saying so in your discussion.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, target, colour in zip(axes, targets, [C_ACTUAL, C_ACCENT]):
        ax.hist(df[target], bins=70, color=colour, edgecolor="white",
                linewidth=0.3)
        mean, med = df[target].mean(), df[target].median()
        ax.axvline(mean, color=C_WARN, linestyle="--", linewidth=1.4,
                   label=f"mean {mean:,.0f}")
        ax.axvline(med, color=INK_2, linestyle=":", linewidth=1.4,
                   label=f"median {med:,.0f}")
        ax.legend(loc="upper right")
        ax.set_xlabel(f"{target} per hour")
        ax.set_ylabel("Number of hours")
        ax.set_title(target.split(" Mass")[0], loc="left")
    fig.tight_layout()
    _save(fig, "05_target_distributions")


def fig_split_diagram(data: dict) -> None:
    """Where the train / validation / test boundaries fall in time.

    WHAT IT SHOWS: proof that the split is chronological, not random. An
    examiner looking for methodological soundness will want this.
    """
    fig, ax = plt.subplots(figsize=(11, 2.6))
    spans = [
        ("Train (70%)", data["ts_train"], C_ACTUAL),
        ("Validation\n(15%)", data["ts_val"], C_BASE),
        ("Test\n(15%)", data["ts_test"], C_PRED),
    ]
    for i, (label, ts, colour) in enumerate(spans):
        start, end = pd.Timestamp(ts[0]), pd.Timestamp(ts[-1])
        ax.barh(0, end - start, left=start, height=0.42, color=colour,
                edgecolor="white", linewidth=2)
        # Direct label on the bar — no legend needed.
        ax.text(start + (end - start) / 2, 0, f"{label}\n{len(ts):,} windows",
                ha="center", va="center", color="white", fontsize=8.5,
                fontweight="bold")
        ax.text(start, -0.34, f"{start:%b %Y}", fontsize=8, color=INK_2)
    ax.set_ylim(-0.6, 0.4)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("Chronological split — the model is never trained on data "
                 "that comes after its test set", loc="left")
    _save(fig, "06_chronological_split")


# ===========================================================================
# PART B — FIGURES THAT EVALUATE A TRAINED MODEL
# One set per target, produced by train_ann.py after training.
# ===========================================================================

def fig_training_history(history, target: str, model_name: str) -> None:
    """Training and validation loss, epoch by epoch.

    HOW TO READ IT: both curves should fall and then flatten. If the training
    curve keeps dropping while the validation curve turns UP, the model is
    memorising the training set instead of learning the pattern — overfitting.
    The dashed line marks the epoch early stopping chose as best.
    """
    h = history.history
    fig, ax = plt.subplots(figsize=(7.5, 4))
    epochs = range(1, len(h["loss"]) + 1)
    ax.plot(epochs, h["loss"], color=C_ACTUAL, label="Training loss")
    ax.plot(epochs, h["val_loss"], color=C_PRED, label="Validation loss")

    best = int(np.argmin(h["val_loss"])) + 1
    ax.axvline(best, color=INK_2, linestyle="--", linewidth=1.2)
    ax.text(best, max(h["loss"]) * 0.9, f" best epoch: {best}",
            color=INK_2, fontsize=9)

    ax.set_xlabel("Epoch")
    loss = getattr(history.model, "loss", "mse")
    loss_label = "Mean squared error" if loss == "mse" else "Huber loss"
    ax.set_ylabel(f"{loss_label} (scaled units)")
    ax.set_title(f"{model_name} training history — {target}", loc="left")
    ax.legend()
    _save(fig, f"07_{_slug(target)}_{model_name.lower()}_training_history")


def fig_prediction_timeseries(ts, y_true, y_pred, target, model_name,
                              days: int = 14) -> None:
    """Actual vs predicted over a two-week slice of the test set.

    WHAT IT SHOWS: the single most persuasive figure in an emissions
    forecasting write-up. It shows whether the model tracks the real shape of
    the load cycle or merely produces a smoothed average. Look specifically at
    the peaks and troughs — models usually under-shoot both.
    """
    n = days * 24
    fig, ax = plt.subplots(figsize=(11, 4))
    t = pd.to_datetime(ts[:n])
    ax.plot(t, y_true[:n], color=C_ACTUAL, label="Actual (measured)")
    ax.plot(t, y_pred[:n], color=C_PRED, label=f"{model_name} prediction",
            alpha=0.9)
    ax.fill_between(t, y_true[:n], y_pred[:n], color=C_WARN, alpha=0.12,
                    linewidth=0, label="Error")
    ax.set_ylabel(f"{target}\n({_unit(target)}/hour)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title(f"{model_name}: first {days} days of the test set — {target}",
                 loc="left")
    ax.legend(ncol=3)
    _save(fig, f"08_{_slug(target)}_{model_name.lower()}_timeseries")


def fig_scatter(y_true, y_pred, target, model_name, r2: float) -> None:
    """Predicted against actual, with the perfect-prediction line.

    HOW TO READ IT: every dot is one test hour. A perfect model puts all dots
    on the dashed 1:1 line. Dots below the line mean the model under-predicted.
    A fan shape widening to the right means the model is less reliable at high
    emissions — very common, and worth reporting.
    """
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(y_true, y_pred, s=5, color=C_ACTUAL, alpha=0.18,
               edgecolors="none")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color=INK_2, linestyle="--", linewidth=1.3,
            label="Perfect prediction (1:1)")
    ax.set_xlabel(f"Actual ({_unit(target)}/hour)")
    ax.set_ylabel(f"Predicted ({_unit(target)}/hour)")
    ax.set_title(f"{model_name} — {target.split(' Mass')[0]}", loc="left")
    # The headline number goes directly on the chart, not just in the caption.
    ax.text(0.04, 0.94, f"R² = {r2:.4f}", transform=ax.transAxes,
            fontsize=12, fontweight="bold", color=INK, va="top")
    ax.legend(loc="lower right")
    _save(fig, f"09_{_slug(target)}_{model_name.lower()}_scatter")


def fig_residuals(y_true, y_pred, target, model_name) -> None:
    """Two diagnostics of the model's mistakes.

    LEFT — histogram of errors. You want a tall, narrow, symmetric bell centred
    on zero. Off-centre means the model is biased (systematically too high or
    too low), which is a real finding worth reporting.

    RIGHT — errors against predicted value. You want a flat, even band. A cone
    that widens to the right (heteroscedasticity) means errors grow with the
    size of the prediction.
    """
    resid = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))

    axes[0].hist(resid, bins=80, color=C_ACTUAL, edgecolor="white",
                 linewidth=0.3)
    axes[0].axvline(0, color=INK_2, linewidth=1.3)
    axes[0].axvline(resid.mean(), color=C_WARN, linestyle="--", linewidth=1.4)
    axes[0].text(resid.mean(), axes[0].get_ylim()[1] * 0.9,
                 f"  bias = {resid.mean():+,.1f}", color=C_WARN, fontsize=9)
    axes[0].set_xlabel(f"Prediction error ({_unit(target)}/hour)")
    axes[0].set_ylabel("Number of test hours")
    axes[0].set_title("Distribution of errors", loc="left")

    axes[1].scatter(y_pred, resid, s=4, color=C_ACTUAL, alpha=0.16,
                    edgecolors="none")
    axes[1].axhline(0, color=INK_2, linewidth=1.3)
    axes[1].set_xlabel(f"Predicted ({_unit(target)}/hour)")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Errors vs prediction size", loc="left")

    fig.suptitle(f"{model_name} residual diagnostics — {target}",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    _save(fig, f"10_{_slug(target)}_{model_name.lower()}_residuals")


def fig_baseline_comparison(metrics: dict, target: str, model_name: str) -> None:
    """Model against the persistence baseline, on three metrics.

    WHY THIS FIGURE MATTERS MORE THAN ANY OTHER: 'persistence' just guesses
    that the next hour equals the current hour. It requires no machine learning
    at all. If your ANN cannot beat it, the ANN has added nothing — and an
    examiner WILL ask. Reporting this comparison up front is what separates a
    credible result from an impressive-looking one.
    """
    names = ["MAE", "RMSE", "MAPE (%)"]
    model_vals = [metrics["model"]["mae"], metrics["model"]["rmse"],
                  metrics["model"]["mape"]]
    base_vals = [metrics["persistence"]["mae"], metrics["persistence"]["rmse"],
                 metrics["persistence"]["mape"]]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, name, mv, bv in zip(axes, names, model_vals, base_vals):
        bars = ax.bar([model_name, "Persistence"], [mv, bv],
                      color=[C_PRED, C_BASE], width=0.55)
        for bar, val in zip(bars, [mv, bv]):
            ax.text(bar.get_x() + bar.get_width() / 2, val,
                    f"{val:,.1f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=INK)
        improve = 100 * (bv - mv) / bv if bv else 0
        ax.set_title(f"{name}   ({improve:+.1f}% vs baseline)", loc="left",
                     fontsize=10)
        ax.set_ylim(0, max(mv, bv) * 1.25)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Does the model beat doing nothing clever? — {target}",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    _save(fig, f"11_{_slug(target)}_{model_name.lower()}_vs_baseline")


def fig_error_breakdown(ts, y_true, y_pred, target, model_name) -> None:
    """Where the model struggles: by hour of day, and by output level.

    WHAT IT SHOWS: an average error hides the interesting story. This splits it
    up. Typically errors spike during morning ramp-up hours and at the lowest
    load levels, because those are the least steady combustion conditions —
    a genuinely useful finding for your discussion chapter.
    """
    err = np.abs(y_pred - y_true)
    t = pd.to_datetime(ts)
    frame = pd.DataFrame({"hour": t.hour, "actual": y_true, "err": err})

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    by_hour = frame.groupby("hour")["err"].mean()
    axes[0].bar(by_hour.index, by_hour.values, color=C_ACTUAL, width=0.72)
    axes[0].axhline(err.mean(), color=C_WARN, linestyle="--", linewidth=1.3)
    axes[0].text(0, err.mean(), f" overall MAE {err.mean():,.0f}",
                 color=C_WARN, fontsize=9, va="bottom")
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel(f"Mean absolute error\n({_unit(target)}/hour)")
    axes[0].set_title("Error by time of day", loc="left")

    # Bin the actual values into deciles, so each bar holds the same number of
    # hours — otherwise the rare extremes would look artificially noisy.
    frame["bin"] = pd.qcut(frame["actual"], 10, duplicates="drop")
    by_level = frame.groupby("bin", observed=True)["err"].mean()
    centres = [iv.mid for iv in by_level.index]
    axes[1].plot(centres, by_level.values, color=C_ACTUAL, marker="o",
                 markersize=5)
    axes[1].axhline(err.mean(), color=C_WARN, linestyle="--", linewidth=1.3)
    axes[1].set_xlabel(f"Actual emissions level ({_unit(target)}/hour, deciles)")
    axes[1].set_ylabel("Mean absolute error")
    axes[1].set_title("Error by emissions level", loc="left")

    fig.suptitle(f"{model_name} error breakdown — {target}",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    _save(fig, f"12_{_slug(target)}_{model_name.lower()}_error_breakdown")


# ===========================================================================
# PART C — FIGURES THAT COMPARE ARCHITECTURES
# Produced by compare_models.py from several training seeds per model.
# ===========================================================================

# One fixed colour per series across every comparison figure.
MODEL_COLOURS = {"ANN": C_ACCENT, "Transformer": C_PRED, "Persistence": C_BASE}


def fig_model_comparison(runs: pd.DataFrame, persistence: dict) -> None:
    """Test-set MAE and RMSE for every model and seed, against persistence.

    HOW TO READ IT: each small dot is one training run (one random seed); the
    large marker is their mean and the vertical line spans best to worst seed.
    The dashed line is persistence. If one model's whole range sits below the
    other's, the ranking does not depend on luck of initialisation. Dots are
    used instead of bars because the differences are small next to the
    values themselves, and bars must start at zero.
    """
    targets = list(dict.fromkeys(runs["target"]))
    models = list(dict.fromkeys(runs["model"]))
    fig, axes = plt.subplots(len(targets), 2, figsize=(10, 3.6 * len(targets)),
                             squeeze=False)
    rng = np.random.default_rng(0)
    for r, target in enumerate(targets):
        for c, (metric, label) in enumerate([("mae", "MAE"), ("rmse", "RMSE")]):
            ax = axes[r, c]
            for i, model in enumerate(models):
                vals = runs.loc[(runs["target"] == target)
                                & (runs["model"] == model), metric].to_numpy()
                colour = MODEL_COLOURS.get(model, C_ACTUAL)
                ax.vlines(i, vals.min(), vals.max(), color=colour, linewidth=2)
                ax.scatter(i + rng.uniform(-0.08, 0.08, len(vals)), vals, s=16,
                           color=colour, alpha=0.55, edgecolors="none", zorder=3)
                ax.scatter(i, vals.mean(), s=90, color=colour, edgecolors="white",
                           linewidth=1.2, zorder=4)
                ax.text(i + 0.14, vals.mean(), f"{vals.mean():,.1f}",
                        va="center", fontsize=9, fontweight="bold", color=INK)
            base = persistence[target][metric]
            ax.axhline(base, color=MODEL_COLOURS["Persistence"], linestyle="--",
                       linewidth=1.4)
            ax.text(len(models) - 0.5, base, f"persistence {base:,.1f}",
                    ha="right", va="bottom", fontsize=9,
                    color=MODEL_COLOURS["Persistence"])
            ax.set_xticks(range(len(models)), models)
            ax.set_xlim(-0.5, len(models) - 0.5)
            ax.grid(axis="x", visible=False)
            ax.set_ylabel(f"{label} ({_unit(target)}/hour)")
            ax.set_title(f"{target.split(' Mass')[0]} — test {label}", loc="left")
    fig.suptitle(f"ANN vs Transformer across {runs['seed'].nunique()} training seeds",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    _save(fig, "13_model_comparison")


def fig_comparison_timeseries(ts, y_true, persistence, preds: dict, target,
                              hours: int = 96) -> None:
    """Actual vs every model over the most volatile stretch of the test set.

    WHY THIS STRETCH: in steady hours every forecaster, including persistence,
    sits on top of the actual line, so a random window shows nothing. This
    picks the continuous `hours`-long window with the most hour-to-hour
    movement, which is where the architectures actually differ.
    """
    t = pd.to_datetime(ts)
    move = np.abs(y_true - persistence)
    window = np.convolve(move, np.ones(hours), mode="valid")
    continuous = (t[hours - 1:] - t[:len(window)]) == pd.Timedelta(hours=hours - 1)
    window = np.where(continuous, window, -np.inf)
    s = slice(int(np.argmax(window)), int(np.argmax(window)) + hours)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(t[s], y_true[s], color=C_ACTUAL, linewidth=2.2, label="Actual (measured)")
    ax.plot(t[s], persistence[s], color=MODEL_COLOURS["Persistence"],
            linestyle="--", linewidth=1.3, label="Persistence")
    for name, pred in preds.items():
        ax.plot(t[s], pred[s], color=MODEL_COLOURS.get(name, C_WARN),
                linewidth=1.5, label=name)
    ax.set_ylabel(f"{target}\n({_unit(target)}/hour)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.set_title(f"Most volatile {hours // 24} days of the test set — {target}",
                 loc="left")
    ax.legend(ncol=len(preds) + 2)
    _save(fig, f"14_{_slug(target)}_comparison_timeseries")


def fig_error_by_regime(by_target: dict) -> None:
    """MAE per quartile of hour-to-hour change, for every model and persistence.

    HOW TO READ IT: the test hours are sorted by how much emissions moved
    since the previous hour and cut into four equal groups. Persistence's
    error in each group is exactly that movement, so it rises steeply. A
    forecaster earns its keep in the right-hand groups (ramps); in the
    left-hand group (steady operation) persistence is close to unbeatable.

    by_target: {target: {"y_true": ..., "persistence": ..., "preds": {name: ...}}}
    """
    fig, axes = plt.subplots(1, len(by_target), figsize=(5.4 * len(by_target), 3.9),
                             squeeze=False)
    labels = ["Q1\nsteadiest", "Q2", "Q3", "Q4\nbiggest ramps"]
    for ax, (target, d) in zip(axes[0], by_target.items()):
        move = np.abs(d["y_true"] - d["persistence"])
        quartile = pd.qcut(move, 4, labels=False, duplicates="drop")
        series = {**d["preds"], "Persistence": d["persistence"]}
        width = 0.8 / len(series)
        for j, (name, pred) in enumerate(series.items()):
            err = np.abs(d["y_true"] - pred)
            mae = [err[quartile == q].mean() for q in range(4)]
            ax.bar(np.arange(4) + (j - (len(series) - 1) / 2) * width, mae,
                   width=width, color=MODEL_COLOURS.get(name, C_ACTUAL), label=name)
        ax.set_xticks(range(4), labels)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(f"MAE ({_unit(target)}/hour)")
        ax.set_title(target.split(" Mass")[0], loc="left")
    axes[0][0].legend(loc="upper left")
    fig.suptitle("Test error by size of the hour-to-hour change",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    _save(fig, "15_error_by_regime")
