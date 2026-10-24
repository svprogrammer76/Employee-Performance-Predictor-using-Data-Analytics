"""Optional Streamlit interface for the synthetic learning demo."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.performance_predictor.data import FEATURE_COLUMNS
from src.performance_predictor.insights import coaching_prompts
from src.performance_predictor.modeling import load_model

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Synthetic Performance Predictor", page_icon="📊", layout="wide")
st.title("Employee Performance Predictor")
st.caption("Learning demo using synthetic data only. Not for appraisal, promotion, pay, or employment decisions.")

data_path = ROOT / "data" / "employees.csv"
metrics_path = ROOT / "outputs" / "metrics.json"
if not data_path.exists() or not metrics_path.exists():
    st.warning("Create the synthetic dataset and model first: run `python main.py` in the project folder.")
    st.stop()

data = pd.read_csv(data_path)
metrics = pd.read_json(metrics_path, typ="series")
left, right = st.columns(2)
left.metric("Synthetic rows", len(data))
right.metric("Holdout macro F1", f"{float(metrics['macro_f1']):.3f}")
st.info(str(metrics["note"]))
st.subheader("Simulated rating distribution")
st.bar_chart(data["next_performance_rating"].value_counts())

st.subheader("Explore a fictional profile")
with st.form("profile"):
    c1, c2, c3 = st.columns(3)
    department = c1.selectbox("Department", ["Engineering", "Sales", "Operations", "HR", "Finance"])
    experience = c2.slider("Experience (years)", 0.0, 30.0, 5.0, .5)
    training = c3.slider("Training hours", 0.0, 80.0, 24.0, 1.0)
    workload = c1.slider("Workload score (1–10)", 1.0, 10.0, 5.5, .1)
    attendance = c2.slider("Attendance rate", .70, 1.0, .94, .01)
    projects = c3.slider("Projects completed", 0, 12, 3)
    feedback = c1.slider("Manager feedback score (1–5)", 1.0, 5.0, 3.4, .1)
    skills = c2.slider("Skills score", 20.0, 100.0, 66.0, 1.0)
    current = c3.slider("Current performance score", 20.0, 100.0, 70.0, 1.0)
    submitted = st.form_submit_button("Generate demo prediction")

if submitted:
    profile = {
        "department": department, "experience_years": experience, "training_hours": training,
        "workload_score": workload, "attendance_rate": attendance, "projects_completed": projects,
        "manager_feedback_score": feedback, "skills_score": skills,
        "current_performance_score": current,
    }
    model = load_model(ROOT / "models" / "performance_pipeline.joblib")
    row = pd.DataFrame([{column: profile[column] for column in FEATURE_COLUMNS}])
    prediction = model.predict(row)[0]
    scores = dict(zip(model.named_steps["classifier"].classes_, model.predict_proba(row)[0]))
    st.metric("Simulated predicted band", prediction)
    st.caption("Scores are model outputs, not calibrated probabilities or a judgment about an individual.")
    st.write("Class scores", {key: round(float(value), 3) for key, value in scores.items()})
    st.subheader("Possible coaching discussion prompts")
    for prompt in coaching_prompts(profile):
        st.write(f"- {prompt}")
