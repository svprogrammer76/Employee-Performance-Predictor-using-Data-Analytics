# Employee Performance Predictor using Data Analytics

A beginner-friendly, end-to-end **synthetic-data** people analytics project. It demonstrates how to create a reproducible dataset, inspect it, train and evaluate a classifier for a *simulated next-cycle performance band* (Low / Medium / High), and turn model signals into discussion prompts for coaching and learning support.

> **Responsible-use boundary:** This is a learning demonstration, not a validated HR decision system. The data and labels are synthetic; model results do not establish real-world validity. Never use this demo to make or automate hiring, firing, pay, promotion, discipline, or other consequential employment decisions. Do not infer an employee’s worth or potential from a model score. Use only with informed consent, human review, appropriate privacy controls, and a documented fairness and impact assessment. Protected characteristics are deliberately excluded from training, but that alone does not prevent proxy bias.

## What it does

- Generates a seeded, fictional workforce dataset (`data/employees.csv`). No real employee information is required.
- Builds a scikit-learn preprocessing and Random Forest classification pipeline.
- Uses a stratified holdout set and reports accuracy, balanced accuracy, macro F1, per-class precision/recall, and a confusion matrix.
- Saves a model and outputs: a class distribution chart, confusion matrix, and sample predictions.
- Produces transparent, rule-based coaching prompts from an individual’s input fields. These are discussion starters, not causal explanations of the model.
- Provides a Streamlit dashboard as an optional interface.

## Options and selected stack

| Path | Tools | Difficulty | Good for |
|---|---|---:|---|
| A — easy | Python, pandas, seaborn, scikit-learn, CLI | Beginner | First analytics project |
| B — intermediate (this project) | Python, pandas, scikit-learn pipeline, validation metrics, Streamlit | Intermediate | Student portfolio and end-to-end practice |
| C — advanced | Time-aware data, model registry, API, monitoring, privacy/fairness review, governed deployment | Advanced | Production engineering study |

Selected: **B**, with synthetic data only and explicit governance constraints.

## Architecture

```text
Seeded synthetic data
        ↓
Schema checks → EDA summaries and charts
        ↓
Train/test split → preprocessing pipeline → Random Forest
        ↓                              ↓
Evaluation metrics              saved model
                                       ↓
                          next-cycle band + probabilities
                                       ↓
               coaching discussion prompts (rule-based)
```

The target is generated from simulated work-context signals with noise. It is not a real appraisal outcome. The model learns patterns in this recipe, so strong results only show that it can recover patterns in the simulation.

## Repository layout

```text
├── app.py                   # Optional Streamlit dashboard
├── main.py                  # CLI workflow: generate, train, evaluate, predict
├── data/                    # Generated dataset; ignored by Git by default
├── models/                  # Serialized fitted pipeline; ignored by Git
├── outputs/                 # Metrics, predictions, charts; ignored by Git
├── src/performance_predictor/
│   ├── data.py              # Synthetic data generation and validation
│   ├── modeling.py          # Pipeline, split, fit, metrics, persistence
│   └── insights.py          # Human-readable coaching prompts
├── requirements.txt
└── .gitignore
```

## Setup and run

Python 3.10+ is recommended.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py --rows 1200 --seed 42
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py --rows 1200 --seed 42
```

The CLI creates `data/employees.csv`, `models/performance_pipeline.joblib`, and files under `outputs/`. To change the number of generated rows, pass `--rows N`.

### Dashboard

After running the CLI once (to create data and model):

```bash
streamlit run app.py
```

In the dashboard, review the dataset overview and metrics, then enter a fictional employee profile. The app shows model probabilities and coaching prompts. A low predicted band is a cue to review context and offer support, not to label an employee.

## Data dictionary

| Field | Meaning | Used by model? |
|---|---|---:|
| employee_id | Randomly assigned fictional identifier | No |
| age | Simulated age, included to demonstrate a sensitive attribute exclusion | No |
| department | Fictional business unit | Yes |
| experience_years | Simulated years of experience | Yes |
| training_hours | Recent learning time | Yes |
| workload_score | Self-reported workload scale, 1–10 | Yes |
| attendance_rate | Simulated attendance proportion, 0–1 | Yes |
| projects_completed | Simulated completed projects | Yes |
| manager_feedback_score | Structured fictional feedback scale, 1–5 | Yes |
| skills_score | Simulated skills assessment scale, 1–100 | Yes |
| current_performance_score | Current-cycle fictional score, 1–100 | Yes |
| next_performance_rating | Simulated next-cycle target: Low / Medium / High | **Target** |

The synthetic target includes stochastic noise and coarse bands. It does not represent validated performance measurement. In real settings, define a defensible outcome with stakeholders and examine rating consistency, missingness, label quality, and disparate impact before modeling.

## Workflow and phases

1. **Setup:** create environment and install dependencies. Output: reproducible Python environment. Common mistake: installing into a different Python interpreter.
2. **Data creation:** run `main.py`. Output: seeded CSV. Mistake: treating synthetic rows as actual workforce evidence.
3. **Data checks:** validate required columns, bounds, and missing values. Mistake: silently coercing invalid values.
4. **EDA:** inspect generated distribution chart and summary JSON. Mistake: interpreting correlations as causes.
5. **Features:** exclude ID and age; one-hot encode department; impute missing numeric/categorical values in the pipeline. Mistake: leaking target or identifiers into features.
6. **Model:** train Random Forest with stratified split. Mistake: tuning on the holdout set.
7. **Evaluation:** review multiple metrics and confusion matrix. Mistake: reporting accuracy alone, especially with imbalanced classes.
8. **Insights:** read individual coaching prompts alongside employee context. Mistake: calling prompts causal explanations.
9. **Visualization:** review generated charts and optional Streamlit dashboard. Mistake: hiding uncertainty or class-level errors.
10. **GitHub:** commit source, README, and optionally small demo charts; keep generated employee-like rows, model binaries, and virtual environments out of Git unless you explicitly choose to publish synthetic artifacts.

## Interpreting results

Metrics are written to `outputs/metrics.json`; class details and confusion matrix are included. Check macro F1 and per-class recall as well as accuracy. A confusion matrix row is the true class and column is the predicted class. Probabilities are model scores, not calibrated likelihoods unless calibration has been assessed. Do not promise a target accuracy: the result varies with the seed, sample, and synthetic recipe.

## Simulation walkthrough

1. Generate 1,200 fictional profiles with a fixed seed.
2. Inspect class balance and the generated fields.
3. Train on 80% and evaluate once on the stratified 20% holdout.
4. Review errors by class and ask what support might be useful.
5. Enter a fictional profile in the dashboard and examine the predicted band and class scores.
6. Treat suggestions (for example, workload review or skill practice) as prompts to discuss with the employee and manager.

Useful portfolio captures: CSV preview with synthetic-data label, target distribution chart, confusion matrix, metric summary, and a dashboard prediction showing all probabilities. Avoid screenshots that imply real employee surveillance or a validated decision tool. Save charts under `outputs/`.

## GitHub portfolio guide

Suggested repository: `employee-performance-predictor-analytics`  
Description: `End-to-end synthetic people analytics demo for next-cycle performance bands, model evaluation, and responsible coaching prompts.`  
Topics: `people-analytics`, `synthetic-data`, `scikit-learn`, `streamlit`, `machine-learning`, `responsible-ai`

```bash
git init
git add README.md .gitignore requirements.txt main.py app.py src
git commit -m "Add synthetic performance prediction workflow"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/employee-performance-predictor-analytics.git
git push -u origin main
```

Example staged milestones: `Add seeded synthetic HR dataset`; `Add EDA and validation`; `Train baseline classifier and report metrics`; `Add coaching prompts and dashboard`; `Document limitations and responsible use`. Do not publish private company or employee data.

## Interview preparation

1. **What problem does it address?** It demonstrates an analytics workflow for identifying simulated next-cycle performance bands and translating signals into supportive conversations.
2. **Why synthetic data?** Real HR data is sensitive and unavailable; simulation makes the project reproducible while limiting privacy risk. It cannot establish real-world validity.
3. **Why exclude age and employee ID?** Age is a sensitive attribute and ID is not a meaningful predictor. Exclusion does not guarantee fairness because other variables can proxy for protected traits.
4. **Why use a pipeline?** It keeps imputation, encoding, and the estimator together, reducing train/test leakage and ensuring inference uses the same transformations.
5. **Why stratify the split?** It preserves approximate class proportions in train and holdout sets.
6. **Why macro F1 and balanced accuracy?** They give each class more equal influence than overall accuracy, which can hide poor minority-class performance.
7. **What is leakage?** Information unavailable at prediction time entering training features or preprocessing; it makes evaluation unrealistically optimistic.
8. **What do feature signals mean?** They reflect associations learned from the synthetic recipe; they are not causal explanations or recommendations by themselves.
9. **How would you take this toward production?** First establish legitimate purpose and governance, validate target quality, assess fairness and privacy, test temporal/generalization performance, calibrate and monitor, and keep decisions with trained humans.
10. **What are key limitations?** Synthetic labels, simplified features, no temporal drift, no external validity, no causal evidence, and no formal fairness validation.

**To HR:** “I built a learning prototype that demonstrates how simulated workforce context can be analyzed and how model outputs can support coaching discussions. It is not an appraisal decision-maker.”

**To a technical interviewer:** Explain synthetic label generation, leakage-safe preprocessing pipeline, stratified holdout, class-level metrics, uncertainty, and the difference between predictive association and causation.

## Troubleshooting

- `ModuleNotFoundError`: activate the project venv and run `pip install -r requirements.txt`.
- Streamlit command missing: install dependencies in the active venv; run `python -m streamlit run app.py`.
- Model not found: run `python main.py` before opening the dashboard.
- PowerShell activation blocked: use the process-scoped execution policy command above.
- Different scores: use the same `--seed`; results still depend on library versions and generated sample size.
- CSV missing columns: regenerate using `python main.py`; do not manually rename required fields.

## Future extensions

Use a documented public dataset only after checking its license, provenance, representativeness, target definition, and permitted use. Other learning extensions include temporal validation, model cards, calibration, subgroup error analysis, drift monitoring, a prediction API, and an attrition model. Deep learning is not automatically an improvement for small tabular data. Any real employment use requires expert legal, privacy, security, labor, and fairness review.

## License

Choose a license before publishing. Add attribution and dataset licensing details if you later use any external data.
