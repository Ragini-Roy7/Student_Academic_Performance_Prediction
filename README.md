
# Student Academic Performance Prediction

A tabular regression project that predicts a student's exam score (`Exam_Score`, 55-101) from 19 academic, behavioral, and socio-demographic attributes, served through a Streamlit app.

The repository contains the dataset, three serialized artifacts (model, preprocessor, feature names), the Streamlit inference app, and an evaluation script. **The training code that produced the artifacts is not in this repository**, and the shipped model has no recorded held-out score. Section 7 documents what that means and what was measured instead.

## Evidence conventions

Claims in this document carry one of four labels.

| Label | Meaning |
|---|---|
| **[V]** Verified | Read directly from a file, artifact, or the dataset in this repository. |
| **[R]** Reproduced | Produced by `python evaluate.py` (seeded; exact output in Section 7). |
| **[I]** Interpretation | A reasoned explanation of verified facts. Not a recorded decision of the author. |
| **[U]** Untested | A risk or hypothesis that no experiment here has examined. |

Last verified: 2026-10-23, with Python 3.12.3, scikit-learn 1.6.1, numpy 2.2.6, pandas 2.2.3. The exact pins in `requirements.txt` (numpy 1.24.3, pandas 2.0.3) were not installed during verification.

## Contents

1. [Problem and intended use](#1-problem-and-intended-use)
2. [Implementation status](#2-implementation-status)
3. [Architecture and data flow](#3-architecture-and-data-flow)
4. [Dataset](#4-dataset)
5. [Methodology](#5-methodology)
6. [Engineering decisions](#6-engineering-decisions)
7. [Results](#7-results)
8. [Error analysis](#8-error-analysis)
9. [Limitations and validity threats](#9-limitations-and-validity-threats)
10. [Setup and usage](#10-setup-and-usage)
11. [Testing](#11-testing)
12. [Security, privacy, operations](#12-security-privacy-operations)
13. [Repository structure](#13-repository-structure)
14. [Known issues and roadmap](#14-known-issues-and-roadmap)
15. [Attribution and licensing](#15-attribution-and-licensing)

---

## 1. Problem and intended use

**Task.** Supervised regression: given one student's attributes, estimate the exam score. The target is an integer in [55, 101] with mean 67.2 and standard deviation 3.9 **[V]**.

**Intended use.** Exploration of how the listed factors relate to exam outcomes, and a demonstration of an end-to-end sklearn-to-Streamlit workflow.

**Non-goals and unsupported uses.**
- Decisions about real students (grading, placement, intervention eligibility, discipline). No evaluation here supports that use.
- Causal conclusions. The model is associational: a high importance for `Hours_Studied` does not show that increasing study hours raises a given student's score **[I]**.
- Generalization beyond this dataset's population. Its provenance is not established (Section 4).

## 2. Implementation status

| Component | Status | Evidence |
|---|---|---|
| Dataset | Present, 6,607 rows | `StudentPerformanceFactors.csv` **[V]** |
| Preprocessing | Implemented as a serialized `ColumnTransformer`; **source code not in repo** | `preprocessor.pkl` **[V]** |
| Model training | **Not in repo.** Shipped model is `RandomForestRegressor(random_state=42)`, default hyperparameters | `student_performance_model.pkl` **[V]** |
| Held-out evaluation of the shipped model | **Not recorded** | Section 7 |
| Inference app | Implemented (Streamlit). Not launched during verification | `app.py` **[V]** |
| Evaluation / reproduction script | Added in this change | `evaluate.py` **[V]** |
| Automated tests, CI | None | no test or workflow files **[V]** |
| Deployment configuration, monitoring, model versioning | None in repo | **[V]** |
| License | None in repo | **[V]** |

An earlier version of this README described deployment targets (Hugging Face, Render, Railway) and a `model/` directory. Neither exists in the repository.

## 3. Architecture and data flow

```
StudentPerformanceFactors.csv
        |
        |   training pipeline  --  NOT IN THIS REPOSITORY
        v
preprocessor.pkl   student_performance_model.pkl   feature_names.pkl
        \                    /                          (not read by app.py)
         v                  v
app.py (Streamlit, @st.cache_resource loads both pickles)
   12 widget values + 7 hard-coded constants
   -> 1-row DataFrame with 19 columns
   -> preprocessor.transform      -> 30 numeric columns
   -> RandomForestRegressor.predict
   -> reject NaN/inf -> clip to [0, 100] -> display with 1 decimal
```

**Boundaries.** The ML component is two pickles. Everything else (widgets, defaults, clipping, error messages) is application code that shapes what the model can be asked **[V]**.

**Input contract (inference).** A DataFrame with the 19 feature columns by name; column order does not matter because the transformer selects by name. Categorical values unseen at training time are silently ignored (`handle_unknown='ignore'`), producing an all-zero indicator block rather than an error **[V]**.

**Transformed feature space (30 columns) [V].** 6 numeric columns passed through unscaled, plus one-hot indicators with the first level dropped for 13 categorical columns. Three columns encode missing values as an explicit category: `Teacher_Quality_nan`, `Parental_Education_Level_nan`, `Distance_from_Home_nan`.

## 4. Dataset

**Provenance.** Source, version, collection method, and license are **not established** by the repository, and the filename alone is not evidence of origin. *[Author to confirm: source URL, license, whether the data is collected, public, or synthetic.]*

**Unit of observation.** One row per student. There is no student identifier or time column **[V]**, so the random split used below assumes rows are independent, and repeated records of the same student cannot be detected.

| Property | Value | Label |
|---|---|---|
| Rows x columns | 6,607 x 20 (19 features + target) | **[V]** |
| Feature types | 6 integer, 13 categorical | **[V]** |
| Exact duplicate rows | 0 | **[V]** |
| Missing cells | 235, only in 3 columns: `Teacher_Quality` 78, `Parental_Education_Level` 90, `Distance_from_Home` 67 | **[V]** |
| `Exam_Score` | min 55, median 67, max 101, mean 67.24, sd 3.89, skew 1.64, IQR 65-69 | **[V]** |
| Scores >= 80 | 48 rows (0.7%) | **[V]** |
| Scores > 100 | 1 row (101), outside a plausible 0-100 scale | **[V]** |

**Feature dictionary.** The repository has no data dictionary; units below come from the labels in `app.py` **[I]**.

| Feature | Range / levels (count) |
|---|---|
| `Hours_Studied` (hours/week) | 1-44 |
| `Attendance` (%) | 60-100 |
| `Sleep_Hours` (per day) | 4-10 |
| `Previous_Scores` (%) | 50-100 |
| `Tutoring_Sessions` (per month) | 0-8 |
| `Physical_Activity` (hours/week) | 0-6 |
| `Parental_Involvement` | Low 1,337 / Medium 3,362 / High 1,908 |
| `Access_to_Resources` | Low 1,313 / Medium 3,319 / High 1,975 |
| `Motivation_Level` | Low 1,937 / Medium 3,351 / High 1,319 |
| `Teacher_Quality` | Low 657 / Medium 3,925 / High 1,947 / missing 78 |
| `Family_Income` | Low 2,672 / Medium 2,666 / High 1,269 |
| `Parental_Education_Level` | High School 3,223 / College 1,989 / Postgraduate 1,305 / missing 90 |
| `Distance_from_Home` | Near 3,884 / Moderate 1,998 / Far 658 / missing 67 |
| `Peer_Influence` | Positive 2,638 / Neutral 2,592 / Negative 1,377 |
| `School_Type` | Public 4,598 / Private 2,009 |
| `Gender` | Male 3,814 / Female 2,793 |
| `Extracurricular_Activities` | Yes 3,938 / No 2,669 |
| `Internet_Access` | Yes 6,108 / No 499 |
| `Learning_Disabilities` | No 5,912 / Yes 695 |

**Leakage review.** No feature is derived from `Exam_Score`. `Previous_Scores` is a prior exam, not the target **[V]**. One leakage-adjacent issue appears in the exploratory workflow (Pearson filtering on the full dataset before splitting) and is covered in Section 6.

**Representativeness and bias [U].** Whether the population resembles any real institution is unknown. `Gender`, `Learning_Disabilities`, and `Family_Income` are model inputs, so subgroup behavior matters; none has been evaluated. The near-additive structure of the target (a linear model reaches R2 0.77 with a median absolute error of 0.29 points, Section 7) is *consistent with* simulated data. This is a hypothesis, not a finding **[U]**.

## 5. Methodology

**What the repository documents about training.** Only the artifacts. From them:

| Property | Value | Label |
|---|---|---|
| Model | `RandomForestRegressor(random_state=42)`: 100 trees, unlimited depth, all other defaults | **[V]** |
| Preprocessor | `ColumnTransformer`: numeric passthrough; `OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)` on 13 categoricals | **[V]** |
| Training rows | **All 6,607.** The shipped model's predictions are identical (max difference 0.0) to a model refit on all rows with the same settings | **[R]** |
| Hyperparameter tuning | None evident: every setting except `random_state` is at its default | **[V]** |

**Reference notebook.** Exploratory work is at <https://www.kaggle.com/code/ragini2002/student-academic-performance>. Its contents could not be retrieved for this documentation (the page renders client-side), so nothing below is taken from its output. A code excerpt supplied by the author describes a different pipeline from the shipped artifact:

| | Notebook excerpt | Shipped artifact |
|---|---|---|
| Features | Columns with abs(Pearson r) > 0.2 against `Exam_Score` | All 19 features (30 encoded columns) |
| Model | `LinearRegression` | `RandomForestRegressor` |
| Split | `train_test_split(test_size=0.2, random_state=42)` | Fit on all rows |

*[Author to confirm: which notebook run produced the shipped artifact, and the metrics it reported.]*

**Evaluation protocol used here (`evaluate.py`).**
- Split: `train_test_split(test_size=0.2, random_state=42)`; 5,285 train / 1,322 test rows.
- The preprocessor is refit inside each pipeline fit, so test rows never influence encoding.
- Cross-validation: 5-fold, shuffled, `random_state=42`, pipeline refit per fold.
- All models use default hyperparameters. No model is tuned, so comparisons are *defaults against defaults*.
- Test rows are used only for reporting; no selection is made on them.

**Metric definitions.** Errors are in exam-score points. RAE, RSE, and RRSE are relative to a predictor that always outputs the **training-set mean**.

- MAE = mean(abs(y - y_hat)); RMSE = sqrt(mean((y - y_hat)^2)); R2 = 1 - SSE / SST about the test mean.
- RAE = sum abs(y - y_hat) / sum abs(y - y_train_mean)
- RSE = sum (y - y_hat)^2 / sum (y - y_train_mean)^2; RRSE = sqrt(RSE)

A value of 1.0 matches the mean predictor; lower is better.

## 6. Engineering decisions

The original rationale for most choices is not recorded. The table separates what the artifacts show from what was measured; reasoning labeled **[I]** is a plausible reconstruction.

| # | Decision | Alternative | Trade-off and failure mode | Evidence | Revisit when |
|---|---|---|---|---|---|
| 1 | Regression on the raw score | Classify into grade bands | Keeps full resolution, but squared-error metrics are dominated by rare extreme scores (Section 8) | **[V]** target is numeric | The use case is flagging at-risk students; use a classification or ranking objective |
| 2 | One-hot, drop-first, for all 13 categoricals, including ordered Low/Medium/High | Ordinal encoding | Order is not encoded; trees can recover it, a linear model spends extra parameters. `handle_unknown='ignore'` avoids crashes but hides unseen values | **[V]** `preprocessor.pkl` | Using models that benefit from monotone structure |
| 3 | Missing values kept as an explicit `nan` category | Imputation | Avoids assuming a fill value and keeps any missingness signal; the app can never produce this category, so those columns are always 0 at inference | **[V]** `feature_names.pkl` | Missingness is shown to be uninformative, or inputs arrive from a form that cannot omit fields |
| 4 | Numeric features passed through unscaled | Standardization | Harmless for trees and for ordinary least squares; matters for regularized or distance-based models | **[V]** | Adding Ridge/Lasso, SVR, or k-NN |
| 5 | `RandomForestRegressor` with default settings | Linear Regression, Gradient Boosting, tuned forest | In the reproduction it is **worse** than Linear Regression by 0.08-0.16 R2 in every CV fold and by 0.12 R2 on the held-out split (Section 7). Fully grown trees fit training noise (train R2 0.944 vs held-out 0.647) | **[V]** model repr; **[R]** comparison | Already triggered: re-select with tuning |
| 6 | Final model trained on 100% of rows | Keep a held-out set for the shipped artifact | Uses all data, a standard practice *if* an unbiased estimate was obtained earlier via CV. None is recorded, so the shipped model's generalization is unmeasured | **[R]** identical to full-data refit | A recorded CV or held-out score exists |
| 7 | Pearson abs(r) > 0.2 feature filter (notebook excerpt only; **not** in the shipped artifact) | Use all features, or a model-based / regularized selection | Univariate filtering keeps only `Attendance` (r = 0.581) and `Hours_Studied` (0.445) and discards `Previous_Scores` (0.175), `Tutoring_Sessions` (0.157), and others. Held-out R2 falls from 0.770 to 0.589. Computing the correlations before the split is a leakage pattern; in this dataset the selected set is identical when computed on training rows only, so it caused no measurable harm here | **[R]** | Never as a final selector for this data |
| 8 | Slider bounds equal the observed min/max of each numeric feature | Free numeric entry | Keeps queries inside the training support (limits extrapolation); users cannot ask about values outside the data | **[V]** `app.py` vs dataset ranges | Retraining on wider data |
| 9 | 7 of 19 features fixed to constants in the app (`Extracurricular_Activities='No'`, `Internet_Access='Yes'`, `Family_Income='Medium'`, `Peer_Influence='Neutral'`, `Learning_Disabilities='No'`, `Parental_Education_Level='College'`, `Distance_from_Home='Near'`) | Expose all inputs | Simpler form, but predictions cannot respond to those factors. Measured cost on held-out rows is modest for the forest and larger for the linear model (Section 7) | **[V]** `app.py`; **[R]** measurement | Users need to evaluate those factors |
| 10 | Output clipped to [0, 100] after rejecting NaN/inf | No clipping | Matches a bounded exam scale. No held-out prediction fell outside [0, 100], so the clip never fired in testing; one training label (101) lies above the cap | **[V]**, **[R]** | The score scale changes |
| 11 | Two separate pickles (model, preprocessor), exact version pins | One `sklearn.Pipeline`; a non-pickle format | The pair can drift apart (the app checks only that both load). Pickles are coupled to the scikit-learn version: `preprocessor.pkl` **fails to load under 1.8.0** (`AttributeError: _RemainderColsList`) and loads under 1.6.1 | **[V]** | Before any upgrade or hosted deployment |

## 7. Results

All numbers below come from `python evaluate.py` unless stated, are in exam-score points, and use the protocol in Section 5. They describe **offline performance on one dataset** and are not production performance.

### 7.1 Held-out split (n_train = 5,285, n_test = 1,322)

| Model | MAE | RMSE | R2 | RAE | RSE | RRSE | Train R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean baseline | 2.824 | 3.761 | -0.001 | 1.000 | 1.000 | 1.000 | 0.000 |
| Linear Regression (all features) | **0.450** | **1.803** | **0.770** | 0.159 | 0.230 | 0.480 | 0.718 |
| Gradient Boosting (defaults) | 0.822 | 1.954 | 0.730 | 0.291 | 0.270 | 0.520 | 0.737 |
| Random Forest (defaults, **shipped config**) | 1.182 | 2.235 | 0.647 | 0.419 | 0.353 | 0.594 | 0.944 |
| Pearson filter + Linear Regression (2 features) | 1.469 | 2.410 | 0.589 | 0.520 | 0.411 | 0.641 | n/a |
| Decision Tree (defaults) | 1.794 | 3.512 | 0.128 | 0.635 | 0.872 | 0.934 | 1.000 |

The Pearson-filter row reconstructs the notebook excerpt with this repository's one-hot encoding; the notebook's encoding may differ **[I]**.

### 7.2 5-fold cross-validation (mean +/- std across folds)

| Model | R2 | MAE | RMSE |
|---|---|---|---|
| Mean baseline | -0.001 +/- 0.001 | 2.849 +/- 0.086 | 3.882 +/- 0.269 |
| Linear Regression | **0.737 +/- 0.102** | **0.486 +/- 0.070** | **1.971 +/- 0.532** |
| Gradient Boosting | 0.696 +/- 0.095 | 0.855 +/- 0.061 | 2.133 +/- 0.479 |
| Random Forest | 0.620 +/- 0.082 | 1.217 +/- 0.052 | 2.393 +/- 0.410 |
| Decision Tree | 0.190 +/- 0.108 | 1.810 +/- 0.104 | 3.491 +/- 0.413 |

Per-fold R2 advantage of Linear Regression: over Random Forest +0.121, +0.125, +0.155, +0.084, +0.102; over Gradient Boosting +0.040, +0.041, +0.056, +0.036, +0.034. The ordering is the same in all five folds.

### 7.3 The shipped artifact

| Quantity | Value |
|---|---|
| R2 on all 6,607 rows | 0.9445 |
| R2 on the `random_state=42` "test" rows | 0.9533 |
| Max prediction difference vs a Random Forest refit on all rows | 0.000000 |

**These are in-sample scores.** The shipped model was trained on every row, including those in the "test" split, so 0.95 must not be read as accuracy. The comparable figure for the same configuration trained on 80% of the data and scored on the other 20% is **R2 0.647 (MAE 1.18)**.

### 7.4 Effect of the app's fixed inputs (held-out rows, 7 features overwritten with the app constants)

| Model | Full inputs: MAE / R2 | App path: MAE / R2 |
|---|---|---|
| Random Forest | 1.182 / 0.647 | 1.254 / 0.632 |
| Linear Regression | 0.450 / 0.770 | 0.885 / 0.721 |

Printed by `evaluate.py` (section "Effect of app.py's fixed inputs").

### 7.5 Interpretation

- **Relative to baseline.** The score distribution is narrow (sd 3.9, IQR 65-69). The shipped configuration cuts MAE by about 58% against predicting the mean (1.18 vs 2.82) but leaves about 35% of the variance unexplained (RSE 0.353).
- **R2 and MAE tell different stories for the linear model.** Its median absolute error is 0.29 points (90th percentile 0.66, 99th 1.02), yet RMSE is 1.80 and the maximum error 28. Seven held-out rows with true scores of 80-98 produce **95.1% of its squared error**. Its R2 of 0.77 is therefore a statement about a few extreme rows, while its typical error is small. Report both.
- **Model comparison.** Linear Regression beats Gradient Boosting and Random Forest in every fold with default settings. That supports the claim that the *shipped* configuration is not the best of these candidates. It does not establish that a tuned forest or boosted model cannot match or beat the linear model.

### 7.6 What these results do not establish

- Any score for the shipped model that is not in-sample.
- Statistical significance. Five folds with fold standard deviations around 0.1 R2 give consistent ordering, not a significance test.
- Uncertainty on the single-split figures (no confidence intervals were computed).
- Behavior on any other population, time period, or institution.
- Performance with tuned hyperparameters for any model.

## 8. Error analysis

Signed error is `y - y_hat`: negative means over-prediction. Bands are by true score, on the held-out split **[R]**.

| True score | n | RF MAE | RF bias | Linear MAE | Linear bias |
|---|---:|---:|---:|---:|---:|
| 55-62 | 104 | 1.642 | -1.583 | 0.438 | -0.332 |
| 63-65 | 319 | 1.041 | -0.693 | 0.341 | -0.205 |
| 66-67 | 286 | 0.955 | -0.317 | 0.331 | -0.184 |
| 68-69 | 265 | 0.995 | -0.041 | 0.302 | -0.148 |
| 70-72 | 270 | 1.015 | +0.233 | 0.292 | -0.119 |
| 73-101 | 78 | 3.179 | +2.713 | 2.399 | +2.049 |

- **Random Forest shrinks toward the mean.** Low scores are over-predicted and high scores under-predicted, with a monotone bias across bands. Its errors exceed 3 points on 52 of 1,322 rows, which account for 73.5% of its squared error.
- **High scores are the weak segment for both models.** Only 48 of 6,607 rows score 80 or above, so there is little to learn from. Whether those rows are genuine high performers or noise or injected outliers is not known **[U]**.
- **Explanatory features (shipped forest, impurity importance) [V]:** `Attendance` 0.388, `Hours_Studied` 0.246, `Previous_Scores` 0.089, `Tutoring_Sessions` 0.036. Impurity importance favors high-cardinality numeric columns over one-hot indicators, so treat the ranking as indicative; permutation importance has not been computed. An unconstrained decision tree (train R2 1.00, held-out 0.13) is a poor basis for importance plots.

## 9. Limitations and validity threats

**Observed**

| Limitation | Evidence | Consequence | Mitigation / next experiment |
|---|---|---|---|
| Shipped model has no held-out estimate | Section 7.3 | Reported accuracy of the deployed artifact is unknown; in-sample 0.94 vs reproduced 0.65 | Run CV in the training script, store the result beside the artifact |
| Shipped configuration trails a linear baseline | Section 7.1-7.2 | A simpler, cheaper, more interpretable model is better on this data | Tune forest and boosting with nested CV; select on CV, report on a held-out set |
| High scores poorly predicted | Section 8 | Predictions for exceptional students are unreliable and pulled toward 67 | Investigate those rows; consider robust loss or quantile models |
| 7 of 19 inputs fixed in the app | `app.py` | The app cannot reflect those factors; accuracy cost measured in 7.4 | Expose the inputs or document the fixed profile in the UI |
| Training code absent | repo listing | Artifacts cannot be regenerated or audited | Commit the pipeline (Section 14) |
| Pickle/version coupling | load failure under 1.8.0 | Upgrades break the app; the app's `warnings.filterwarnings('ignore')` would also suppress scikit-learn's version-mismatch warning | Single `Pipeline` artifact, pinned and tested; remove the blanket warning filter |
| One label above 100 | Section 4 | Label validity issue; the app clips outputs at 100 | Inspect the row; decide to cap or drop it |

**Plausible but untested [U]:** distribution shift; subgroup error by gender, disability status, or income; calibration or prediction intervals (the app returns a point estimate only); sensitivity to the encoding or split seed; duplicate students across rows.

**Implementation gaps [V]:** no tests, CI, license, input schema validation outside the UI, or logging. The check `hours_studied <= 0 or attendance <= 0` in `app.py` is unreachable, because the sliders' minimums are 1 and 60.

## 10. Setup and usage

**Requirements.** Python 3.9-3.11 is the expected range **[I]**: the pinned numpy 1.24.3 and pandas 2.0.3 publish no wheels for 3.12, and commit `c42f775` is titled "Use Python 3.9 compatible versions". `requirements.txt` is UTF-16 encoded; re-save it as UTF-8 if your tooling fails to parse it. `pip install -r requirements.txt` was not run during verification.

```bash
git clone https://github.com/Ragini-Roy7/Student_Academic_Performance_Prediction.git
cd Student_Academic_Performance_Prediction

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Run the app.** It is a Streamlit app; `python app.py` will not serve it.

```bash
streamlit run app.py
```

**Reproduce the metrics.**

```bash
pip install tabulate
python evaluate.py                  # comparison, CV, error analysis, artifact check
python evaluate.py --skip-artifact  # without loading any pickle
```

**Programmatic inference.**

```python
import joblib, pandas as pd

model = joblib.load("student_performance_model.pkl")
pre = joblib.load("preprocessor.pkl")

row = pd.DataFrame([{
    "Hours_Studied": 20, "Attendance": 80, "Parental_Involvement": "Medium",
    "Access_to_Resources": "Medium", "Extracurricular_Activities": "No",
    "Sleep_Hours": 7, "Previous_Scores": 75, "Motivation_Level": "Medium",
    "Internet_Access": "Yes", "Tutoring_Sessions": 2, "Family_Income": "Medium",
    "Teacher_Quality": "Medium", "School_Type": "Public",
    "Peer_Influence": "Neutral", "Physical_Activity": 3,
    "Learning_Disabilities": "No", "Parental_Education_Level": "College",
    "Distance_from_Home": "Near", "Gender": "Male",
}])
print(model.predict(pre.transform(row))[0])   # 67.07
```

**Troubleshooting.**
- `AttributeError: Can't get attribute '_RemainderColsList'` when loading `preprocessor.pkl`: scikit-learn is newer than the version that wrote it. Install `scikit-learn==1.6.1` **[V]**.
- "Model files not found" warning in the app: run `streamlit run app.py` from the repository root, because the pickles are loaded by relative path **[V]**.

## 11. Testing

There is no automated test suite or CI. `evaluate.py` is a reproducibility check, not a test suite: it recomputes the reported metrics and runs the artifact-provenance check, with no pass/fail assertions. Nothing tests `app.py`, the input contract, or the clipping logic.

## 12. Security, privacy, operations

| Topic | Status |
|---|---|
| Untrusted pickles | `joblib.load` executes code embedded in a pickle. Load only artifacts you trust, including these. **Recommended:** a non-pickle format or hash-verified artifacts |
| Personal data | The CSV has no direct identifiers **[V]**. It includes sensitive attributes (gender, disability status, income); do not extend it with identifiable records |
| App behavior | `app.py` makes no network calls and writes nothing to disk or logs **[V]**. `unsafe_allow_html=True` is applied only to static markup and one numeric value |
| Input validation | Provided by slider/select bounds only. Any other caller of the model gets none **[V]** |
| Monitoring, drift detection, retraining, rollback, versioning | **Not implemented.** No deployed instance or production use is established |

## 13. Repository structure

```
.
├── README.md
├── app.py                          # Streamlit inference app
├── evaluate.py                     # reproduces Section 7 and 8 (added)
├── StudentPerformanceFactors.csv   # 6,607 x 20
├── student_performance_model.pkl   # RandomForestRegressor, 34.6 MB, trained on all rows
├── preprocessor.pkl                # fitted ColumnTransformer (scikit-learn 1.6.1)
├── feature_names.pkl               # 30 encoded column names (unused by app.py)
└── requirements.txt                # UTF-16 encoded
```

## 14. Known issues and roadmap

**Highest priority**

1. **Commit the training pipeline.** One script that builds a single `sklearn.Pipeline` (preprocessing + model), runs CV, evaluates on a held-out set, and saves the artifact together with its metrics and library versions.
2. **Redo model selection.** Tune Random Forest and Gradient Boosting against the Linear Regression baseline with nested CV; investigate the high-score rows before choosing a loss function.
3. **Fix the app's input contract.** Expose or document the seven fixed inputs, remove the unreachable check, drop the blanket warning filter, and add tests covering the input DataFrame and the clip.

**Next**

- Permutation importance and subgroup error analysis (gender, disability, income).
- Prediction intervals (for example quantile or conformal methods) instead of a single point.
- Re-save `requirements.txt` as UTF-8; add CI that installs it and runs `evaluate.py --skip-artifact`.
- Add a license and the dataset's source and terms.

## 15. Attribution and licensing

No license file exists, so default copyright applies. *[Author to add: dataset source, license, and citation; project license.]*

**Author:** Ragini Roy. [GitHub](https://github.com/Ragini-Roy7)
