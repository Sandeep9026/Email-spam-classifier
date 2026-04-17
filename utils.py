from __future__ import annotations

import logging
import os
import re
import string
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


LOGGER = logging.getLogger(__name__)
NLTK_RESOURCES = ("stopwords",)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_nltk_resources() -> None:
    for resource in NLTK_RESOURCES:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            LOGGER.warning(
                "NLTK resource '%s' is unavailable. Falling back to sklearn stopwords.",
                resource,
            )


@lru_cache(maxsize=1)
def get_stop_words() -> set[str]:
    ensure_nltk_resources()
    try:
        return set(stopwords.words("english"))
    except LookupError:
        LOGGER.warning("Using sklearn English stopword list because NLTK stopwords are unavailable.")
        return set(ENGLISH_STOP_WORDS)


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

    df = pd.read_csv(path)
    expected_columns = {"label", "text"}
    if not expected_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns {expected_columns}, got {set(df.columns)}"
        )

    df = df.loc[:, ["label", "text"]].dropna().copy()
    df["label"] = df["label"].str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def clean_text(text: str, stop_words: set[str]) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = re.findall(r"\b[a-z]+\b", text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return " ".join(filtered_tokens)


def preprocess_series(texts: Iterable[str]) -> pd.Series:
    stop_words = get_stop_words()
    return pd.Series([clean_text(text, stop_words) for text in texts])


def save_artifact(obj: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    LOGGER.info("Saved artifact to %s", path.resolve())


def load_artifact(path: str | Path) -> object:
    return joblib.load(path)


def save_plot(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved figure to %s", path.resolve())


def plot_class_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df, x="label", hue="label", palette="Set2", legend=False, ax=ax)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    return fig


def plot_confusion_matrix(conf_matrix, labels: list[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return fig


def plot_model_comparison(results_df: pd.DataFrame) -> plt.Figure:
    metrics = ["accuracy", "precision", "recall", "f1"]
    melted = results_df.melt(
        id_vars=["model_name"], value_vars=metrics, var_name="metric", value_name="score"
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=melted, x="metric", y="score", hue="model_name", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.legend(title="Model")
    return fig
