"""Run the synthetic data, training, evaluation, and example prediction workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.performance_predictor.data import FEATURE_COLUMNS, TARGET, make_synthetic_data
from src.performance_predictor.insights import coaching_prompts
from src.performance_predictor.modeling import save_metrics, save_model, train_evaluate

ROOT = Path(__file__).resolve().parent


def create_charts(data: pd.DataFrame, metrics: dict) -> None:
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    counts = data[TARGET].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot.bar(ax=ax, color=["#d97757", "#e5b94c", "#4d9b78"])
    ax.set(title="Synthetic target class counts", xlabel="Simulated next-cycle band", ylabel="Rows")
    fig.tight_layout(); fig.savefig(out / "class_distribution.png", dpi=150); plt.close(fig)
    matrix = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(3), xticklabels=metrics["labels_order"], yticks=range(3),
           yticklabels=metrics["labels_order"], xlabel="Predicted", ylabel="Actual",
           title="Synthetic holdout confusion matrix")
    for row in range(3):
        for col in range(3): ax.text(col, row, matrix[row][col], ha="center", va="center")
    fig.colorbar(image, ax=ax); fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = make_synthetic_data(args.rows, args.seed)
    (ROOT / "data").mkdir(exist_ok=True)
    data.to_csv(ROOT / "data" / "employees.csv", index=False)
    model, metrics, predictions = train_evaluate(data, args.seed)
    save_model(model, ROOT / "models" / "performance_pipeline.joblib")
    save_metrics(metrics, ROOT / "outputs" / "metrics.json")
    predictions.to_csv(ROOT / "outputs" / "sample_predictions.csv", index=False)
    create_charts(data, metrics)
    print("Synthetic dataset created:", len(data), "rows")
    print("Class counts:", data[TARGET].value_counts().to_dict())
    print("Holdout metrics:", json.dumps({k: metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")}, indent=2))
    profile = data.iloc[0][FEATURE_COLUMNS].to_dict()
    print("Example fictional profile:", json.dumps(profile, default=str))
    print("Predicted band:", model.predict(pd.DataFrame([profile]))[0])
    print("Coaching discussion prompts:")
    for prompt in coaching_prompts(profile): print(" -", prompt)
    print("Outputs saved under data/, models/, and outputs/.")


if __name__ == "__main__":
    main()
