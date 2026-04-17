from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils import resample

from utils import (
    configure_logging,
    load_dataset,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    preprocess_series,
    save_artifact,
    save_plot,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingArtifacts:
    best_model_name: str
    metrics_path: Path
    best_model_path: Path


def balance_training_data(X_train: pd.Series, y_train: pd.Series) -> tuple[pd.Series, pd.Series]:
    train_df = pd.DataFrame({"text": X_train, "target": y_train})
    majority = train_df[train_df["target"] == 0]
    minority = train_df[train_df["target"] == 1]

    if minority.empty or majority.empty:
        LOGGER.warning("Skipping resampling because one class is missing in the training split.")
        return X_train, y_train

    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=42,
    )
    balanced_df = (
        pd.concat([majority, minority_upsampled])
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )
    return balanced_df["text"], balanced_df["target"]


def build_models() -> dict[str, Pipeline]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
    return {
        "naive_bayes": Pipeline(
            steps=[
                ("tfidf", vectorizer),
                ("model", MultinomialNB()),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "svm": Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)),
                (
                    "model",
                    SVC(
                        kernel="linear",
                        probability=True,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model_name: str, pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series) -> dict:
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "avg_spam_probability": float(probabilities.mean()),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["ham", "spam"],
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    return metrics


def train(args: argparse.Namespace) -> TrainingArtifacts:
    configure_logging()
    project_root = Path(__file__).resolve().parent
    data_path = project_root / args.data_path
    models_dir = project_root / "models"
    figures_dir = project_root / "reports" / "figures"
    reports_dir = project_root / "reports"

    df = load_dataset(data_path)
    df["processed_text"] = preprocess_series(df["text"])

    save_plot(plot_class_distribution(df), figures_dir / "class_distribution.png")

    X_train, X_test, y_train, y_test = train_test_split(
        df["processed_text"],
        df["target"],
        test_size=args.test_size,
        random_state=42,
        stratify=df["target"],
    )

    X_train_balanced, y_train_balanced = balance_training_data(X_train, y_train)

    LOGGER.info(
        "Training samples before balancing: %s | after balancing: %s",
        len(X_train),
        len(X_train_balanced),
    )

    results: list[dict] = []
    trained_models: dict[str, Pipeline] = {}

    for model_name, pipeline in build_models().items():
        LOGGER.info("Training model: %s", model_name)
        pipeline.fit(X_train_balanced, y_train_balanced)
        metrics = evaluate_model(model_name, pipeline, X_test, y_test)
        results.append(metrics)
        trained_models[model_name] = pipeline
        conf_fig = plot_confusion_matrix(
            metrics["confusion_matrix"],
            labels=["ham", "spam"],
            title=f"Confusion Matrix - {model_name}",
        )
        save_plot(conf_fig, figures_dir / f"confusion_matrix_{model_name}.png")

    results_df = pd.DataFrame(results).drop(columns=["classification_report", "confusion_matrix"])
    results_df = results_df.sort_values(by=["f1", "precision", "recall"], ascending=False)
    best_model_name = results_df.iloc[0]["model_name"]

    save_plot(plot_model_comparison(results_df), figures_dir / "model_comparison.png")

    metrics_path = reports_dir / "metrics_summary.csv"
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(metrics_path, index=False)

    detailed_metrics_path = reports_dir / "detailed_metrics.json"
    detailed_metrics = {result["model_name"]: result for result in results}
    detailed_metrics_path.write_text(json.dumps(detailed_metrics, indent=2), encoding="utf-8")

    best_model_path = models_dir / "spam_classifier.joblib"
    metadata_path = models_dir / "metadata.json"
    save_artifact(trained_models[best_model_name], best_model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "best_model_name": best_model_name,
                "dataset_path": str(data_path),
                "test_size": args.test_size,
                "total_samples": int(len(df)),
                "class_distribution": df["label"].value_counts().to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    LOGGER.info("Best model selected: %s", best_model_name)
    LOGGER.info("Metrics summary saved to %s", metrics_path.resolve())
    return TrainingArtifacts(
        best_model_name=best_model_name,
        metrics_path=metrics_path,
        best_model_path=best_model_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the spam email classifier.")
    parser.add_argument(
        "--data-path",
        default="data/spam.csv",
        help="Path to the CSV dataset relative to the project root.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for evaluation.",
    )
    return parser


if __name__ == "__main__":
    artifacts = train(build_arg_parser().parse_args())
    print(f"Best model: {artifacts.best_model_name}")
    print(f"Saved model: {artifacts.best_model_path}")
    print(f"Metrics summary: {artifacts.metrics_path}")
