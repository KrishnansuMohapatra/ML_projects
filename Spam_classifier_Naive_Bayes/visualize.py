from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from main import LABELS, build_detail, build_model, load_data, threshold_sweep, top_features

BG = "#0d1117"
GRID = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
HAM = "#58a6ff"
SPAM = "#f778ba"
ACCENT = "#d29922"
GOOD = "#3fb950"

FIGURES_DIR = "figures"
RANDOM_STATE = 42
RAW_ROWS_IN_CSV = 5572
DEFAULT_THRESHOLD = 0.5

CMAP = LinearSegmentedColormap.from_list("nb", [BG, "#1f6feb", "#a371f7", SPAM])
GAUGE = LinearSegmentedColormap.from_list("gauge", [HAM, GOOD, ACCENT, SPAM])


def apply_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "text.color": TEXT,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 11,
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": GRID,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.labelcolor": TEXT,
            "legend.fontsize": 9.5,
            "lines.linewidth": 2.2,
            "font.size": 10,
            "figure.dpi": 110,
            "savefig.dpi": 165,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.2,
        }
    )


def caption(fig: plt.Figure, text: str) -> plt.Figure:
    fig.supxlabel(text, fontsize=8.5, color=MUTED, ha="left", x=0.01)
    return fig


def bar_labels(ax: plt.Axes, bars, fmt: str = "%.3f", size: float = 8.5) -> None:
    ax.bar_label(bars, fmt=fmt, padding=3, fontsize=size, color=MUTED)


def fig_class_distribution(counts: pd.Series) -> plt.Figure:
    values = counts.reindex(LABELS).to_numpy()
    total = int(values.sum())

    fig, ax = plt.subplots(figsize=(6.4, 5.0), layout="constrained")
    wedges, _ = ax.pie(
        values,
        colors=[HAM, SPAM],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.4, "edgecolor": BG, "linewidth": 2.5},
    )
    ax.text(0, 0.16, f"{total:,}", ha="center", va="center", fontsize=34, fontweight="bold")
    ax.text(0, -0.2, "messages", ha="center", va="center", fontsize=10, color=MUTED)
    ax.text(0, -0.44, f"{values[1] / total:.1%} spam", ha="center", va="center", fontsize=11, color=SPAM, fontweight="bold")
    ax.set_title("Class balance")
    ax.grid(False)
    fig.legend(
        wedges,
        [f"{lab}   {int(v):,}   {v / total:.1%}" for lab, v in zip(LABELS, values)],
        loc="outside lower center",
        ncol=2,
    )
    dropped = RAW_ROWS_IN_CSV - total
    return caption(fig, f"after dropping {dropped:,} empty or duplicate rows from spam.csv")


def fig_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    share = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(5.6, 5.0), layout="constrained")
    masked = np.ma.masked_where(cm <= 0, cm.astype(float))
    im = ax.imshow(masked, cmap=CMAP, norm=LogNorm(vmin=0.5, vmax=float(cm.max())))
    ax.grid(False)

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n{share[i, j]:.1%}",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color=BG if share[i, j] > 0.5 else TEXT,
            )
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Confusion matrix")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("messages (log scale)", color=MUTED, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_visible(False)
    note = f"log colour scale keeps the {cm[0, 1]:,}-message cell visible next to {cm[0, 0]:,}"
    return caption(fig, f"{note}; {int(cm[0, 1] + cm[1, 0])} errors in total")


def fig_pr_roc(y_true: np.ndarray, proba: np.ndarray) -> plt.Figure:
    precision, recall, _ = precision_recall_curve(y_true, proba)
    fpr, tpr, _ = roc_curve(y_true, proba)
    ap_score = average_precision_score(y_true, proba)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.7), layout="constrained")

    ax1.plot(recall, precision, color=SPAM, label=f"PR curve   AP = {ap_score:.3f}")
    ax1.axhline(y_true.mean(), color=MUTED, ls=":", lw=1.4, label=f"baseline   {y_true.mean():.3f}")
    ax1.fill_between(recall, precision, y_true.mean(), color=SPAM, alpha=0.12)
    ax1.set(xlabel="recall", ylabel="precision", xlim=(0, 1.02), ylim=(0, 1.03), title="Precision-Recall")
    ax1.legend(loc="lower left")

    ax2.plot(fpr, tpr, color=HAM, label=f"ROC curve   AUC = {auc(fpr, tpr):.3f}")
    ax2.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1.4, label="chance")
    ax2.fill_between(fpr, tpr, color=HAM, alpha=0.12)
    ax2.set(
        xlabel="false positive rate",
        ylabel="true positive rate",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
        title="ROC",
    )
    ax2.legend(loc="lower right")

    fig.suptitle("Discrimination curves", x=0.01, ha="left", weight="bold", fontsize=12.5)
    return caption(fig, "average precision is the honest score here - accuracy hides the 87/13 imbalance")


def fig_performance_bars(report: dict) -> plt.Figure:
    names = LABELS + ["macro avg"]
    metrics = ["precision", "recall", "f1-score"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4), layout="constrained")

    x = np.arange(len(names))
    width = 0.26
    for i, (metric, color) in enumerate(zip(metrics, [HAM, SPAM, ACCENT])):
        values = [report[n][metric] for n in names]
        bars = ax.bar(x + (i - 1) * width, values, width, label=metric, color=color, alpha=0.88)
        bar_labels(ax, bars)

    ax.set_xticks(x, [n.replace(" avg", "\naverage") for n in names])
    ax.set_ylim(0, 1.18)
    ax.set_title("Per-class scores")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=3)
    return caption(fig, "spam recall is the weak axis and it is what pulls macro average under accuracy")


def fig_length_distribution(messages: pd.Series, labels: pd.Series) -> plt.Figure:
    lengths = messages.str.len()
    bins = np.linspace(0, min(500, lengths.max()), 60)

    fig, ax = plt.subplots(figsize=(7.8, 4.5), layout="constrained")
    for label, color in zip(LABELS, [HAM, SPAM]):
        subset = lengths[labels.eq(label)]
        ax.hist(
            subset,
            bins=bins,
            color=color,
            alpha=0.6,
            label=f"{label}   median {subset.median():.0f} chars",
            edgecolor=BG,
            linewidth=0.4,
        )
    ax.set_yscale("log")
    ax.set(xlabel="message length (characters)", ylabel="messages (log scale)", title="Message length by class")
    ax.legend(loc="upper right")
    return caption(fig, "spam skews longer, but the curves overlap heavily through the 100-200 character band")


def fig_confidence_distribution(proba: np.ndarray, y_true: np.ndarray, threshold: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.8, 4.5), layout="constrained")
    bins = np.linspace(0, 1, 51)

    for value, name, color in [(0, "ham", HAM), (1, "spam", SPAM)]:
        ax.hist(
            proba[y_true == value],
            bins=bins,
            color=color,
            alpha=0.6,
            label=name,
            edgecolor=BG,
            linewidth=0.4,
        )
    ax.axvline(threshold, color=ACCENT, ls="--", lw=1.8, label=f"threshold {threshold:.2f}")
    ax.set_yscale("log")
    ax.set(xlabel="predicted p(spam)", ylabel="messages (log scale)", title="Class separation")
    ax.legend(loc="upper center", ncol=4)
    return caption(fig, "the mass at 0 and 1 is confidence; every error sits in the middle band")


def fig_threshold_tradeoff(sweep: pd.DataFrame, threshold: float) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.0, 6.6), sharex=True, layout="constrained", height_ratios=[2, 1]
    )

    for metric, color in [("precision", HAM), ("recall", SPAM), ("f1", GOOD)]:
        ax1.plot(sweep["threshold"], sweep[metric], color=color, label=metric)
        ax1.fill_between(sweep["threshold"], 0, sweep[metric], color=color, alpha=0.07)
    ax1.axvline(threshold, color=ACCENT, ls="--", lw=1.8)
    ax1.set(ylabel="score", ylim=(0, 1.06), title="Decision threshold trade-off")
    ax1.legend(loc="lower left", ncol=3)

    ax2.plot(sweep["threshold"], sweep["false_positives"], color=HAM, label="false positives")
    ax2.plot(sweep["threshold"], sweep["missed_spam"], color=SPAM, label="missed spam")
    ax2.axvline(threshold, color=ACCENT, ls="--", lw=1.8)
    ax2.set(xlabel="threshold", ylabel="errors")
    ax2.legend(loc="upper left", ncol=2)

    return caption(fig, "0.50-0.95 is flat and costs recall for nothing; the useful band is 0.15-0.40")


def fig_top_features(model, n: int = 15) -> plt.Figure:
    table = top_features(model, n=n)
    fig, ax = plt.subplots(figsize=(8.4, 0.26 * len(table) + 1.5), layout="constrained")

    ax.barh(
        table["feature"],
        table["log_ratio"],
        color=[SPAM if v > 0 else HAM for v in table["log_ratio"]],
        alpha=0.9,
        height=0.72,
    )
    ax.axvline(0, color=MUTED, lw=1)
    ax.invert_yaxis()
    ax.set(xlabel="log P(feature | spam) - log P(feature | ham)", title=f"Strongest {n} spam and {n} ham features")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(table["log_ratio"].min() * 1.06, table["log_ratio"].max() * 1.06)
    return caption(fig, "these log-ratios are exactly the weights the model sums at prediction time")


def fig_common_words(model, messages: pd.Series, y_true: np.ndarray, n: int = 12) -> plt.Figure:
    vocabulary = model.named_steps["tfidf"].vocabulary_
    counts = CountVectorizer(vocabulary=vocabulary).fit_transform(messages)
    names = np.asarray(model.named_steps["tfidf"].get_feature_names_out())
    total = int(counts.sum())

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 0.32 * n + 1.7), layout="constrained")
    for ax, value, name, color in zip(axes, [0, 1], LABELS, [HAM, SPAM]):
        row = np.asarray(counts[y_true == value].sum(axis=0)).ravel()
        order = np.argsort(row)[::-1][:n]
        ax.barh(names[order][::-1], row[order][::-1], color=color, alpha=0.85, height=0.7)
        ax.set(xlabel="occurrences", title=f"Most frequent in {name}")
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=8.5)
        ax.set_xlim(0, float(row[order[0]]) * 1.3)

    fig.suptitle("Most frequent vocabulary by class", x=0.01, ha="left", weight="bold", fontsize=12.5)
    return caption(fig, f"counts over the {total:,} tokens in the trained vocabulary")


def fig_probability_gauge(proba: float, threshold: float) -> plt.Figure:
    verdict = "spam" if proba >= threshold else "ham"
    color = SPAM if verdict == "spam" else HAM

    fig, ax = plt.subplots(figsize=(8.4, 1.9), layout="constrained")
    ax.imshow(np.linspace(0, 1, 512).reshape(1, -1), aspect="auto", cmap=GAUGE, vmin=0, vmax=1)
    ax.axvline(threshold, color=TEXT, ls="--", lw=1.6)
    ax.axvline(proba, color=color, lw=3.5, solid_capstyle="round")
    ax.text(
        proba,
        1.24,
        f"{proba:.1%}  {verdict}",
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        color=color,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0.55, 1.4)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(f"p(spam)      dashed marker = threshold {threshold:.2f}")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


def fig_token_contributions(tokens: list[str], contributions: np.ndarray) -> plt.Figure:
    order = np.argsort(contributions)[::-1]
    tokens = [tokens[i] for i in order]
    values = np.asarray(contributions)[order]
    keep = values != 0

    fig, ax = plt.subplots(figsize=(8.0, max(1.4, 0.3 * int(keep.sum()) + 1.3)), layout="constrained")
    ax.barh(
        tokens,
        values,
        color=[SPAM if v > 0 else HAM for v in values],
        alpha=0.9,
        height=0.68,
    )
    ax.axvline(0, color=MUTED, lw=1)
    ax.invert_yaxis()
    ax.set(xlabel="log-probability contribution", title="Which tokens drove the decision")
    ax.grid(axis="y", visible=False)
    limit = float(np.abs(values[keep]).max()) if keep.any() else 1.0
    ax.set_xlim(-limit * 1.35, limit * 1.35)
    return caption(fig, "pink pushed toward spam, blue toward ham; the rest of the message is noise to the model")


def build_all(
    detail: pd.DataFrame,
    model,
    messages: pd.Series,
    y_train: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    out_dir: str | None = FIGURES_DIR,
) -> dict[str, plt.Figure]:
    apply_theme()
    y_true = (detail["actual"] == "spam").astype(int).to_numpy()
    y_pred = (detail["predicted"] == "spam").astype(int).to_numpy()
    proba = detail["p_spam"].to_numpy()
    report = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True, zero_division=0)

    figures = {
        "01_class_distribution": fig_class_distribution(detail["actual"].value_counts()),
        "02_confusion_matrix": fig_confusion_matrix(y_true, y_pred),
        "03_pr_roc": fig_pr_roc(y_true, proba),
        "04_performance_bars": fig_performance_bars(report),
        "05_length_distribution": fig_length_distribution(detail["message"], detail["actual"]),
        "06_confidence_distribution": fig_confidence_distribution(proba, y_true, threshold),
        "07_threshold_tradeoff": fig_threshold_tradeoff(threshold_sweep(detail), threshold),
        "08_top_features": fig_top_features(model, n=15),
        "09_common_words": fig_common_words(model, messages, y_train, n=12),
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(os.path.join(out_dir, f"{name}.png"))
    return figures


def main() -> None:
    from sklearn.model_selection import train_test_split

    from main import tune_alpha

    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["target"],
        test_size=0.2,
        stratify=df["target"],
        random_state=RANDOM_STATE,
    )
    alpha = tune_alpha(X_train, y_train)
    model = build_model().set_params(nb__alpha=alpha)
    model.fit(X_train, y_train)
    detail = build_detail(model, X_test, y_test)

    figures = build_all(detail, model, X_train, y_train.to_numpy())
    for name in figures:
        print(f"wrote {os.path.join(FIGURES_DIR, name + '.png')}")


if __name__ == "__main__":
    main()
