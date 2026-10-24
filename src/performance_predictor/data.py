"""Create and validate a fictional employee dataset for learning only."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "department", "experience_years", "training_hours", "workload_score",
    "attendance_rate", "projects_completed", "manager_feedback_score",
    "skills_score", "current_performance_score",
]
TARGET = "next_performance_rating"
REQUIRED_COLUMNS = ["employee_id", "age", *FEATURE_COLUMNS, TARGET]


def make_synthetic_data(rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Return seeded, fictional rows; generated ratings are not real appraisals."""
    if rows < 60:
        raise ValueError("Use at least 60 rows to create a useful train/test split.")
    rng = np.random.default_rng(seed)
    departments = np.array(["Engineering", "Sales", "Operations", "HR", "Finance"])
    department = rng.choice(departments, size=rows, p=[.30, .22, .22, .12, .14])
    experience = np.clip(rng.gamma(2.2, 3.0, rows), 0, 30).round(1)
    training = np.clip(rng.normal(24, 12, rows), 0, 80).round(1)
    workload = np.clip(rng.normal(5.7, 1.7, rows), 1, 10).round(1)
    attendance = np.clip(rng.normal(.94, .045, rows), .70, 1).round(3)
    projects = rng.poisson(np.clip(2.3 + experience * .12, 1.5, 6), rows)
    feedback = np.clip(rng.normal(3.4, .75, rows), 1, 5).round(1)
    skills = np.clip(rng.normal(66 + experience * .55 + training * .20, 12, rows), 20, 100).round(1)
    current = np.clip(
        .48 * skills + 5.5 * feedback + 4.0 * projects + 2.1 * training
        - 2.3 * np.maximum(workload - 6, 0) + rng.normal(0, 11, rows), 20, 100
    ).round(1)
    # The target is intentionally noisy and only reflects this simulation recipe.
    latent = (
        .34 * (skills - 60) / 12 + .27 * (feedback - 3) / .8
        + .22 * (current - 65) / 13 + .12 * (training - 24) / 14
        + .10 * (attendance - .94) / .04 - .11 * (workload - 5.7) / 1.7
        + rng.normal(0, .65, rows)
    )
    # Quantile bands avoid a nearly empty simulated class in small demo samples.
    low_cut, high_cut = np.quantile(latent, [1 / 3, 2 / 3])
    rating = np.where(latent <= low_cut, "Low", np.where(latent <= high_cut, "Medium", "High"))
    result = pd.DataFrame({
        "employee_id": [f"SYN-{i:05d}" for i in range(1, rows + 1)],
        "age": rng.integers(21, 61, rows), "department": department,
        "experience_years": experience, "training_hours": training,
        "workload_score": workload, "attendance_rate": attendance,
        "projects_completed": projects, "manager_feedback_score": feedback,
        "skills_score": skills, "current_performance_score": current,
        TARGET: rating,
    })
    validate_data(result)
    return result


def validate_data(data: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if data.empty:
        raise ValueError("Dataset has no rows.")
    if data[TARGET].isna().any() or not set(data[TARGET].unique()).issubset({"Low", "Medium", "High"}):
        raise ValueError("Target must contain only Low, Medium, and High labels.")
