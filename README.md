# Student Academic Performance Prediction

Tabular regression system that predicts a student's exam score from 19 academic, behavioral, and socio-demographic attributes, served through a Streamlit app. The repository includes a reproducible evaluation harness (`evaluate.py`) that compares the shipped model against alternatives under a leakage-safe protocol.

**Engineering summary.** The shipped model is a default-configuration Random Forest trained on all 6,607 rows. Held-out evaluation shows it is not the strongest candidate on this dataset: a plain Linear Regression reduces held-out MAE from 1.18 to 0.45 score points and raises R2 from 0.65 to 0.77. This README documents that finding, the decisions behind the current system, and what remains unvalidated.

## Contents

1. [Problem definition and scope](#1-problem-definition-and-scope)
2. [Current status](#2-current-status)
3. [Architecture and data flow](#3-architecture-and-data-flow)
4. [Dataset](#4-dataset)
5. [Experimental methodology](#5-experimental-methodology)
6. [Engineering decisions and trade-offs](#6-engineering-decisions-and-trade-offs)
7. [Evaluation results](#7-evaluation-results)
8. [Error analysis](#8-error-analysis)
9. [Failure modes and limitations](#9-failure-modes-and-limitations)
10. [Reproducibility and usage](#10-reproducibility-and-usage)
11. [Testing and quality assurance](#11-testing-and-quality-assurance)
12. [Security, privacy, and operations](#12-security-privacy-and-operations)
13. [Repository structure](#13-repository-structure)
14. [Roadmap](#14-roadmap)
15. [Attribution](#15-attribution)

---

## 1. Problem definition and scope

**Task.** Supervised regression. Input: one student record with 19 attributes. Output: predicted `Exam_Score`, an integer target in [55, 101] (mean 67.2, std 3.9).

**Intended use.** Studying how the listed factors relate to exam outcomes, and demonstrating a complete sklearn-to-Streamlit workflow with an honest evaluation.

**Out of scope.**
- Decisions about real students (grading, placement, intervention, discipline). No evaluation here supports that.
- Causal claims. The model is associational: feature importance for `Hours_Studied` does not show that raising study hours raises a student's score.
- Generalization beyond this dataset. Its source and population are undocumented.

## 2. Current status

| Component | Status |
|---|---|
| Dataset (`StudentPerformanceFactors.csv`) | Present, 6,607 rows |
| Preprocessing | Implemented, shipped as a fitted `ColumnTransformer` (`preprocessor.pkl`) |
| Model | Shipped: `RandomForestRegressor(random_state=42)`, defaults, trained on all rows |
| Training script | **Not in this repository** |
| Evaluation harness | Implemented and working: `evaluate.py` |
| Inference app | Implemented: Streamlit (`app.py`) |
| Automated tests / CI | None |
| Deployment, monitoring, model versioning | None |

## 3. Architecture and data flow

```
StudentPerformanceFactors.csv
        |
        |  training pipeline (not in this repository)
        v
preprocessor.pkl     student_performance_model.pkl     feature_names.pkl
        \                        /                       (not read by app.py)
         v                      v
   app.py (Streamlit; both pickles loaded once via @st.cache_resource)
     12 user inputs + 7 hard-coded constants
       -> one-row DataFrame, 19 columns
       -> preprocessor.transform        -> 30 numeric columns
       -> RandomForestRegressor.predict
       -> reject NaN/inf -> clip to [0, 100] -> display (1 decimal)

evaluate.py (offline, independent of the pickles except for one provenance check)
     CSV -> split / 5-fold CV -> per-fold preprocessor fit -> models -> metrics
```

**Component boundaries.** The ML component is two pickled objects. Widget bounds, the fixed defaults, output clipping, and error messages live in `app.py` and determine what the model can be asked.

**Input contract.** A DataFrame containing the 19 feature columns by name (order is irrelevant; the transformer selects by name). Categorical values not seen in training are silently ignored (`handle_unknown='ignore'`) and yield an all-zero indicator block rather than an error.

**Feature space (30 columns).** Six numeric columns passed through unscaled, plus one-hot indicators (first level dropped) for 13 categorical columns. Three columns encode missing values as their own category: `Teacher_Quality_nan`, `Parental_Education_Level_nan`, `Distance_from_Home_nan`.

## 4. Dataset

Source, license, and collection method are not documented in this repository. The unit of observation is one student per row. There is no student identifier or timestamp, so repeated students cannot be detected and all splits assume independent rows.

| Property | Value |
|---|---|
| Shape | 6,607 x 20 (19 features, 1 target) |
| Feature types | 6 integer, 13 categorical |
| Exact duplicates | 0 |
| Missing cells | 235, in three columns only: `Teacher_Quality` 78, `Parental_Education_Level` 90, `Distance_from_Home` 67 |
| Target | min 55, median 67, max 101, mean 67.24, std 3.89, skew 1.64, IQR 65-69 |
| Upper tail | 48 rows score >= 80 (0.7%); one row scores 101, above a 0-100 scale |

**Features.**

| Feature | Range / levels |
|---|---|
| `Hours_Studied` | 1-44 |
| `Attendance` | 60-100 |
| `Sleep_Hours` | 4-10 |
| `Previous_Scores` | 50-100 |
| `Tutoring_Sessions` | 0-8 |
| `Physical_Activity` | 0-6 |
| `Parental_Involvement`, `Access_to_Resources`, `Motivation_Level` | Low / Medium / High |
| `Teacher_Quality` | Low / Medium / High (78 missing) |
| `Family_Income` | Low / Medium / High |
| `Parental_Education_Level` | High School / College / Postgraduate (90 missing) |
| `Distance_from_Home` | Near / Moderate / Far (67 missing) |
| `Peer_Influence` | Positive / Neutral / Negative |
| `School_Type` | Public / Private |
| `Gender` | Male / Female |
| `Extracurricular_Activities`, `Internet_Access`, `Learning_Disabilities` | Yes / No |

There is no data dictionary in the repository; units (hours per week, percent, and so on) follow the labels used in `app.py`.

**Leakage review.** No feature is derived from `Exam_Score`; `Previous_Scores` is a prior exam. The one leakage-style pattern in the project's exploratory workflow is computing a correlation-based feature filter on the full dataset before splitting. On this data the selected features are identical when computed on training rows only, so it caused no measurable harm, but it is not a safe pattern in general.

**Representativeness.** Unknown. `Gender`, `Learning_Disabilities`, and `Family_Income` are model inputs, and error rates by group have not been evaluated.

## 5. Experimental methodology

**Task and target.** Regress `Exam_Score` (points).

**Protocol (`evaluate.py`).**
- Holdout: `train_test_split(test_size=0.2, random_state=42)` gives 5,285 training and 1,322 test rows.
- Preprocessing is part of each pipeline, so the encoder is refit on training data only; test rows never influence encoding.
- Cross-validation: 5 folds, shuffled, `random_state=42`, full pipeline refit per fold.
- All models use default hyperparameters. No model is tuned, so every comparison is defaults against defaults.
- Test rows are used only for reporting. No model or feature selection is made on them.

**Baselines and candidates.** Mean predictor (reference), Linear Regression, Gradient Boosting, Random Forest (the shipped configuration), Decision Tree, and a Linear Regression restricted to features with abs(Pearson r) > 0.2 against the target (a reconstruction of a filter used in the project notebook, using this repository's encoding).

**Metrics.** Errors are in exam-score points.
- MAE = mean abs(y - y_hat); RMSE = sqrt(mean (y - y_hat)^2); R2 = 1 - SSE/SST.
- RAE, RSE, RRSE are relative to a predictor that always outputs the **training-set mean**: RAE = sum abs(y - y_hat) / sum abs(y - y_train_mean); RSE = sum (y - y_hat)^2 / sum (y - y_train_mean)^2; RRSE = sqrt(RSE). A value of 1.0 equals the mean predictor.
- Both MAE and RMSE are reported because the error distribution is heavy-tailed (Section 8).

**Provenance check of the shipped model.** `evaluate.py` refits `RandomForestRegressor(random_state=42)` on all rows and compares its predictions to the shipped model. The maximum difference is 0.0, which establishes that the shipped model was trained on every row.

**Other sources.** The exploratory notebook is at <https://www.kaggle.com/code/ragini2002/student-academic-performance>. It describes a different pipeline from the shipped artifact (a correlation-filtered Linear Regression rather than a full-feature Random Forest). Its outputs are not used in this document.

## 6. Engineering decisions and trade-offs

The original rationale for most choices is not recorded. "Evidence" states what supports each row.

| Decision | Alternative | Trade-off and failure mode | Evidence | Reconsider when |
|---|---|---|---|---|
| Regress the raw score | Classify into grade bands | Keeps full resolution; squared-error metrics get dominated by rare extreme scores | Target is numeric | The goal is flagging at-risk students; use a classification or ranking objective |
| One-hot encoding, first level dropped, including ordered Low/Medium/High columns | Ordinal encoding | Order is not encoded; trees recover it, linear models spend extra parameters. `handle_unknown='ignore'` prevents crashes but hides unseen values | `preprocessor.pkl` | Models that benefit from monotone structure |
| Missing values as an explicit `nan` category | Imputation | No imputed value to defend and any missingness signal is kept; the app can never produce this category, so those columns are always 0 at inference | `feature_names.pkl` | Missingness is shown to be uninformative |
| Numeric columns unscaled | Standardization | Harmless for trees and OLS; matters for regularized or distance-based models | `preprocessor.pkl` | Adding Ridge, Lasso, SVR, or k-NN |
| Random Forest, default settings | Linear Regression, Gradient Boosting, tuned forest | Fully grown trees fit training noise (train R2 0.944 vs held-out 0.647). Linear Regression beats it by 0.08-0.16 R2 in every CV fold and 0.12 on the holdout | Section 7 | Already met: re-select with tuning |
| Final model trained on 100% of rows | Hold out data for the shipped artifact | Uses all data, which is standard practice when an unbiased estimate was obtained earlier by CV. No such estimate is recorded, so the shipped model's generalization is unmeasured | Provenance check, Section 5 | A recorded CV or held-out score exists |
| Correlation filter, abs(r) > 0.2 (notebook only, not in the shipped model) | All features, or model-based / regularized selection | Univariate filtering keeps `Attendance` (r 0.581) and `Hours_Studied` (0.445) and drops `Previous_Scores` (0.175), `Tutoring_Sessions` (0.157), and others. Held-out R2 falls from 0.770 to 0.589 | Section 7 | Not appropriate as a final selector on this data |
| App slider bounds equal the dataset's min/max per numeric feature | Free numeric entry | Keeps queries inside the training support and limits extrapolation | `app.py` vs dataset ranges | Retraining on wider data |
| 7 of 19 features fixed in the app | Expose all inputs | Simpler form; predictions cannot respond to those factors (measured cost in Section 7.4) | `app.py`, `evaluate.py` | Users need to evaluate those factors |
| Output clipped to [0, 100] after a NaN/inf check | No clipping | Fits a bounded scale. No held-out prediction fell outside [0, 100], so the clip never fired in testing; one label (101) exceeds the cap | `app.py`, `evaluate.py` | The score scale changes |
| Two pickles with exact version pins | One `sklearn.Pipeline`; a non-pickle format | The pair can drift apart (the app only checks both load). The preprocessor fails to load under scikit-learn 1.8.0 (`AttributeError: _RemainderColsList`) and loads under 1.6.1 | `evaluate.py` artifact check | Before any upgrade or hosted deployment |

## 7. Evaluation results

All figures below are printed by `python evaluate.py`. They are offline results on one dataset and are not production performance.

### 7.1 Held-out split (n_train = 5,285, n_test = 1,322)

| Model | MAE | RMSE | R2 | RAE | RSE | RRSE |
|---|---:|---:|---:|---:|---:|---:|
| Mean baseline | 2.824 | 3.761 | -0.001 | 1.000 | 1.000 | 1.000 |
| Linear Regression | **0.450** | **1.803** | **0.770** | 0.159 | 0.230 | 0.480 |
| Gradient Boosting | 0.822 | 1.954 | 0.730 | 0.291 | 0.270 | 0.520 |
| Random Forest (shipped configuration) | 1.182 | 2.235 | 0.647 | 0.419 | 0.353 | 0.594 |
| Linear Regression, abs(r) > 0.2 filter (2 features) | 1.469 | 2.410 | 0.589 | 0.520 | 0.411 | 0.641 |
| Decision Tree | 1.794 | 3.512 | 0.128 | 0.635 | 0.872 | 0.934 |

### 7.2 5-fold cross-validation (mean +/- std across folds)

| Model | R2 | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.737 +/- 0.102 | 0.486 +/- 0.070 | 1.971 +/- 0.532 |
| Gradient Boosting | 0.696 +/- 0.095 | 0.855 +/- 0.061 | 2.133 +/- 0.479 |
| Random Forest | 0.620 +/- 0.082 | 1.217 +/- 0.052 | 2.393 +/- 0.410 |
| Decision Tree | 0.190 +/- 0.108 | 1.810 +/- 0.104 | 3.491 +/- 0.413 |
| Mean baseline | -0.001 +/- 0.001 | 2.849 +/- 0.086 | 3.882 +/- 0.269 |

Per-fold R2 advantage of Linear Regression over Random Forest: +0.121, +0.125, +0.155, +0.084, +0.102. Over Gradient Boosting: +0.040, +0.041, +0.056, +0.036, +0.034. The ordering holds in all five folds.

### 7.3 The shipped model's own scores are in-sample

| Quantity | Value |
|---|---|
| R2 on all 6,607 rows | 0.9445 |
| R2 on the rows a `random_state=42` split would call "test" | 0.9533 |
| Max prediction difference vs a forest refit on all rows | 0.000000 |

The shipped model was trained on every row, so these scores measure fit, not generalization. The comparable held-out figure for the same configuration is **R2 0.647, MAE 1.18**.

### 7.4 Effect of the app's fixed inputs

Held-out rows with the 7 features the app fixes overwritten by the app's constants:

| Model | Full inputs: MAE / R2 | App path: MAE / R2 |
|---|---|---|
| Random Forest | 1.182 / 0.647 | 1.254 / 0.632 |
| Linear Regression | 0.450 / 0.770 | 0.885 / 0.721 |

### 7.5 Interpretation

- **Versus baseline.** The target is narrow (std 3.9, IQR 65-69). The shipped configuration cuts MAE by about 58% against predicting the mean (1.18 vs 2.82) and leaves about 35% of the variance unexplained (RSE 0.353).
- **R2 and MAE disagree for the linear model.** Its median absolute error is 0.29 points (90th percentile 0.66, 99th 1.02), but RMSE is 1.80 and the maximum error is 28. Seven held-out rows with true scores of 80-98 produce 95.1% of its squared error. Its R2 therefore reflects a few extreme rows, while typical error is small. Report both.
- **Model comparison.** With default settings Linear Regression beats Gradient Boosting and Random Forest on every fold. That shows the shipped configuration is not the best of these candidates. It does not show that a tuned forest or boosting model cannot match the linear model.

### 7.6 What these results do not establish

- Any out-of-sample score for the shipped artifact itself.
- Statistical significance: five folds with fold std near 0.1 R2 give a consistent ordering, not a test.
- Uncertainty on the single-split figures; no confidence intervals were computed.
- Behavior on other populations, other years, or other institutions.
- Performance of tuned models.

## 8. Error analysis

Signed error is `y - y_hat`, so negative means over-prediction. Bands are by true score on the held-out split.

| True score | n | RF MAE | RF bias | Linear MAE | Linear bias |
|---|---:|---:|---:|---:|---:|
| 55-62 | 104 | 1.642 | -1.583 | 0.438 | -0.332 |
| 63-65 | 319 | 1.041 | -0.693 | 0.341 | -0.205 |
| 66-67 | 286 | 0.955 | -0.317 | 0.331 | -0.184 |
| 68-69 | 265 | 0.995 | -0.041 | 0.302 | -0.148 |
| 70-72 | 270 | 1.015 | +0.233 | 0.292 | -0.119 |
| 73-101 | 78 | 3.179 | +2.713 | 2.399 | +2.049 |

- **Random Forest regresses toward the mean.** Low scores are over-predicted, high scores under-predicted, with bias changing monotonically across bands. It makes errors above 3 points on 52 of 1,322 rows, which carry 73.5% of its squared error.
- **High scores are the weak segment for both models.** Only 48 of 6,607 rows score 80 or above, so there is little signal to learn from. Whether those rows are genuine high performers or noisy outliers has not been investigated.
- **Importances of the shipped forest** (impurity-based): `Attendance` 0.388, `Hours_Studied` 0.246, `Previous_Scores` 0.089, `Tutoring_Sessions` 0.036. Impurity importance favors numeric over one-hot columns, so treat the ranking as indicative; permutation importance has not been computed. An unconstrained decision tree (train R2 1.00, held-out 0.13) is a poor basis for importance plots.

## 9. Failure modes and limitations

| Limitation | Evidence | Consequence | Mitigation / next experiment |
|---|---|---|---|
| Shipped model has no held-out estimate | Section 7.3 | True accuracy of the shipped artifact is unknown; in-sample 0.94 versus reproduced 0.65 | Run CV in a committed training script and store the result with the artifact |
| Shipped configuration trails a linear baseline | Sections 7.1-7.2 | A simpler, cheaper, more interpretable model is better on this data | Tune forest and boosting with nested CV; select on CV, report on a held-out set |
| Poor accuracy on high scores | Section 8 | Predictions for exceptional students are unreliable and pulled toward 67 | Inspect those rows; try robust loss or quantile models |
| 7 of 19 inputs fixed in the app | `app.py`, Section 7.4 | The app cannot reflect those factors | Expose them, or state the fixed profile in the UI |
| Training code absent | Repository contents | Artifacts cannot be regenerated or audited | Commit the pipeline (Section 14) |
| Pickle and version coupling | Load failure under 1.8.0 | Library upgrades break the app; `warnings.filterwarnings('ignore')` in `app.py` would also hide scikit-learn's version-mismatch warning | Ship a single pinned `Pipeline` artifact; remove the blanket warning filter |
| One label above 100 | Section 4 | Label-validity question; the app clips outputs at 100 | Inspect the row and decide whether to cap or drop it |
| Unused validation | `app.py` | The check `hours_studied <= 0 or attendance <= 0` is unreachable because slider minimums are 1 and 60 | Remove it or validate at the model boundary |

**Not evaluated:** distribution shift, per-group error (gender, disability status, income), calibration or prediction intervals (the app returns a point estimate only), sensitivity to the encoding choice or split seed, duplicate students across rows.

## 10. Reproducibility and usage

`requirements.txt` pins `streamlit==1.29.0`, `pandas==2.0.3`, `numpy==1.24.3`, `scikit-learn==1.6.1`, `joblib==1.3.2`, `plotly==5.17.0`. The numpy and pandas pins correspond to Python 3.9-3.11. The file is UTF-16 encoded; re-save it as UTF-8 if your tooling cannot parse it. `evaluate.py` was run with Python 3.12, scikit-learn 1.6.1, numpy 2.2.6, and pandas 2.2.3.

```bash
git clone https://github.com/Ragini-Roy7/Student_Academic_Performance_Prediction.git
cd Student_Academic_Performance_Prediction

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Run the app** (from the repository root; the pickles load by relative path):

```bash
streamlit run app.py
```

**Reproduce every number in Sections 7 and 8:**

```bash
pip install tabulate
python evaluate.py                  # holdout, CV, error analysis, filter reconstruction, app-path effect, artifact check
python evaluate.py --skip-artifact  # same, without loading any pickle
```

**Predict without the app:**

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

**Troubleshooting.** `AttributeError: Can't get attribute '_RemainderColsList'` when loading `preprocessor.pkl` means scikit-learn is newer than the version that saved it. Install `scikit-learn==1.6.1`. The artifact check in `evaluate.py` prints the same hint.

## 11. Testing and quality assurance

There is no automated test suite or CI. `evaluate.py` is a reproducibility check, not a test suite: it recomputes the reported metrics and runs the artifact provenance check, with no pass/fail assertions. Nothing currently tests `app.py`, the input contract, or the clipping logic.

## 12. Security, privacy, and operations

| Topic | Status |
|---|---|
| Untrusted pickles | `joblib.load` can execute code embedded in a pickle. Load only artifacts you trust. A non-pickle format or hash-verified artifacts would remove this risk |
| Personal data | The CSV has no direct identifiers. It includes sensitive attributes (gender, disability status, income); do not extend it with identifiable records |
| App behavior | `app.py` makes no network calls and does not write inputs to disk or logs. `unsafe_allow_html=True` is applied only to static markup and one numeric value |
| Input validation | Slider and selector bounds only; any other caller of the model gets none |
| Monitoring, drift detection, retraining, rollback, versioning | Not implemented. No deployed instance or production use is claimed |

## 13. Repository structure

```
.
├── README.md
├── app.py                          # Streamlit inference app
├── evaluate.py                     # reproduces Sections 7 and 8
├── StudentPerformanceFactors.csv   # 6,607 x 20
├── student_performance_model.pkl   # RandomForestRegressor, trained on all rows
├── preprocessor.pkl                # fitted ColumnTransformer (scikit-learn 1.6.1)
├── feature_names.pkl               # 30 encoded column names (unused by app.py)
└── requirements.txt                # UTF-16 encoded
```

## 14. Roadmap

Prioritized by how much each item would strengthen the project.

1. **Commit the training pipeline.** One script that builds a single `sklearn.Pipeline` (preprocessing and model), runs CV, evaluates on a held-out set, and saves the artifact together with its metrics and library versions.
2. **Redo model selection.** Tune Random Forest and Gradient Boosting against the Linear Regression baseline with nested CV; investigate the high-score rows before choosing a loss function.
3. **Fix the app's input contract.** Expose or document the seven fixed inputs, remove the unreachable check and the blanket warning filter, and add tests for the input DataFrame and the clipping.
4. Permutation importance and per-group error analysis (gender, disability, income).
5. Prediction intervals (quantile or conformal) instead of a single point.
6. Re-save `requirements.txt` as UTF-8, add CI that installs it and runs `evaluate.py --skip-artifact`.
7. Add a license and document the dataset's source and terms.

## 15. Attribution

The dataset's source and license are not documented in this repository, and the repository has no license file. Author: Ragini Roy, [GitHub](https://github.com/Ragini-Roy7).
