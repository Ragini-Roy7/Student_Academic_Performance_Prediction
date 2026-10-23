"""Reproduce the held-out and cross-validated metrics reported in README.md.

Trains fresh models from StudentPerformanceFactors.csv (it does not use the
shipped .pkl files for the comparison) and, separately, checks how the shipped
model artifact was trained.

Usage:
    python evaluate.py                    # comparison + artifact check
    python evaluate.py --skip-artifact    # comparison only (no pickle loading)

The artifact check needs the scikit-learn version pinned in requirements.txt;
preprocessor.pkl does not load under newer releases (verified with 1.8.0).
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

DATA = "StudentPerformanceFactors.csv"
TARGET = "Exam_Score"
SEED = 42
TEST_SIZE = 0.2
NUMERIC = ["Hours_Studied", "Attendance", "Sleep_Hours",
           "Previous_Scores", "Tutoring_Sessions", "Physical_Activity"]

MODELS = {
    "Mean baseline": DummyRegressor(),
    "Linear Regression": LinearRegression(),
    "Decision Tree (defaults)": DecisionTreeRegressor(random_state=SEED),
    "Random Forest (defaults)": RandomForestRegressor(random_state=SEED),
    "Gradient Boosting (defaults)": GradientBoostingRegressor(random_state=SEED),
}


def make_pipeline(estimator, categorical):
    """Same preprocessing as the shipped preprocessor.pkl, refit inside every fit()."""
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                              sparse_output=False), categorical),
    ])
    return Pipeline([("pre", pre), ("model", clone(estimator))])


def regression_metrics(y_true, y_pred, train_mean):
    """RAE/RSE/RRSE use the *training-set mean* as the reference predictor."""
    y_true = np.asarray(y_true)
    sse = ((y_true - y_pred) ** 2).sum()
    sst = ((y_true - train_mean) ** 2).sum()
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
        "RAE": np.abs(y_true - y_pred).sum() / np.abs(y_true - train_mean).sum(),
        "RSE": sse / sst,
        "RRSE": (sse / sst) ** 0.5,
    }


def held_out(X, y, categorical):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED)
    print(f"\n## Held-out split (test_size={TEST_SIZE}, random_state={SEED}): "
          f"train n={len(X_tr)}, test n={len(X_te)}\n")
    rows = []
    for name, est in MODELS.items():
        pipe = make_pipeline(est, categorical).fit(X_tr, y_tr)
        row = regression_metrics(y_te, pipe.predict(X_te), y_tr.mean())
        row["train_R2"] = r2_score(y_tr, pipe.predict(X_tr))
        rows.append(pd.Series(row, name=name))
    print(pd.DataFrame(rows).round(4).to_markdown())


def cross_validated(X, y, categorical):
    print("\n## 5-fold CV (shuffle, random_state=42); mean +/- std across folds\n")
    kf = KFold(5, shuffle=True, random_state=SEED)
    summary, per_fold_r2 = [], {}
    for name, est in MODELS.items():
        r2s, maes, rmses = [], [], []
        for tr, te in kf.split(X):
            pipe = make_pipeline(est, categorical).fit(X.iloc[tr], y.iloc[tr])
            pred = pipe.predict(X.iloc[te])
            r2s.append(r2_score(y.iloc[te], pred))
            maes.append(mean_absolute_error(y.iloc[te], pred))
            rmses.append(mean_squared_error(y.iloc[te], pred) ** 0.5)
        per_fold_r2[name] = r2s
        summary.append({
            "model": name,
            "R2": f"{np.mean(r2s):.3f} +/- {np.std(r2s):.3f}",
            "MAE": f"{np.mean(maes):.3f} +/- {np.std(maes):.3f}",
            "RMSE": f"{np.mean(rmses):.3f} +/- {np.std(rmses):.3f}",
        })
    print(pd.DataFrame(summary).set_index("model").to_markdown())
    lr, rf, gb = (np.array(per_fold_r2[k]) for k in
                  ("Linear Regression", "Random Forest (defaults)",
                   "Gradient Boosting (defaults)"))
    print("\nPaired per-fold R2 difference, Linear - RandomForest:", np.round(lr - rf, 3))
    print("Paired per-fold R2 difference, Linear - GradBoosting:", np.round(lr - gb, 3))


def error_analysis(X, y, categorical):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED)
    print("\n## Error analysis on the held-out split\n")
    for name in ("Linear Regression", "Random Forest (defaults)"):
        pipe = make_pipeline(MODELS[name], categorical).fit(X_tr, y_tr)
        resid = y_te.values - pipe.predict(X_te)
        abs_err = np.abs(resid)
        big = abs_err > 3
        print(f"{name}: abs-error median/90th/99th/max = "
              f"{np.round(np.quantile(abs_err, [.5, .9, .99, 1]), 2)}; "
              f"|error| > 3 points: {big.sum()} of {len(resid)}; "
              f"share of squared error from those rows: "
              f"{100 * (resid[big] ** 2).sum() / (resid ** 2).sum():.1f}%")
        bands = pd.cut(y_te.values, [54, 62, 65, 67, 69, 72, 102])
        print(pd.DataFrame({"band": bands, "abs": abs_err, "err": resid})
              .groupby("band", observed=True)
              .agg(n=("abs", "size"), MAE=("abs", "mean"), bias=("err", "mean"))
              .round(3).to_markdown(), "\n")


APP_CONSTANTS = {  # values app.py hard-codes for features it does not expose
    "Extracurricular_Activities": "No", "Internet_Access": "Yes",
    "Family_Income": "Medium", "Peer_Influence": "Neutral",
    "Learning_Disabilities": "No", "Parental_Education_Level": "College",
    "Distance_from_Home": "Near",
}


def app_path_effect(X, y, categorical):
    """Held-out cost of the 7 features app.py fixes to constants."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED)
    X_app = X_te.assign(**APP_CONSTANTS)
    print("\n## Effect of app.py's fixed inputs (held-out rows)\n")
    rows = []
    for name in ("Random Forest (defaults)", "Linear Regression"):
        pipe = make_pipeline(MODELS[name], categorical).fit(X_tr, y_tr)
        full = regression_metrics(y_te, pipe.predict(X_te), y_tr.mean())
        app = regression_metrics(y_te, pipe.predict(X_app), y_tr.mean())
        rows.append({"model": name,
                     "full inputs MAE/R2": f"{full['MAE']:.3f} / {full['R2']:.3f}",
                     "app path MAE/R2": f"{app['MAE']:.3f} / {app['R2']:.3f}"})
    print(pd.DataFrame(rows).set_index("model").to_markdown())


def pearson_filter(X, y, categorical):
    """Reconstruct the |r| > 0.2 feature filter + LinearRegression described in the
    Kaggle notebook (encoding there may differ: this uses the repo's one-hot encoding)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED)
    pre = make_pipeline(LinearRegression(), categorical).named_steps["pre"]
    full = pd.DataFrame(clone(pre).fit_transform(X),
                        columns=clone(pre).fit(X).get_feature_names_out())
    corr_full = full.assign(y=y.values).corr()["y"].drop("y")
    keep_full = list(corr_full[corr_full.abs() > 0.2].index)
    pre_tr = clone(pre).fit(X_tr)
    cols = pre_tr.get_feature_names_out()
    Z_tr = pd.DataFrame(pre_tr.transform(X_tr), columns=cols)
    Z_te = pd.DataFrame(pre_tr.transform(X_te), columns=cols)
    corr_tr = Z_tr.assign(y=y_tr.values).corr()["y"].drop("y")
    keep_tr = list(corr_tr[corr_tr.abs() > 0.2].index)
    print("\n## Pearson |r| > 0.2 filter + LinearRegression (reconstruction)\n")
    print("Selected on full data:", {k: round(corr_full[k], 3) for k in keep_full})
    print("Same selection when computed on the training split only:", set(keep_full) == set(keep_tr))
    print("Largest |r| among rejected features:",
          corr_full.drop(keep_full).abs().sort_values(ascending=False).head(3).round(3).to_dict())
    lin = LinearRegression().fit(Z_tr[keep_tr], y_tr)
    print("Held-out metrics:", {k: round(v, 4) for k, v in
          regression_metrics(y_te, lin.predict(Z_te[keep_tr]), y_tr.mean()).items()})


def artifact_check(X, y, categorical):
    import joblib
    print("\n## Shipped artifact check\n")
    try:
        deployed = joblib.load("student_performance_model.pkl")
        saved_pre = joblib.load("preprocessor.pkl")
    except Exception as exc:  # version mismatch is the expected failure here
        print(f"Could not load artifacts: {type(exc).__name__}: {exc}\n"
              "Install the scikit-learn version from requirements.txt and retry.")
        return
    deployed_pred = deployed.predict(saved_pre.transform(X))
    print("Shipped model:", deployed)
    print(f"In-sample R2 of shipped model on all {len(X)} rows: "
          f"{r2_score(y, deployed_pred):.4f}")
    refit = make_pipeline(RandomForestRegressor(random_state=SEED), categorical).fit(X, y)
    diff = np.abs(refit.predict(X) - deployed_pred).max()
    print(f"Max |prediction difference| vs RandomForestRegressor(random_state={SEED}) "
          f"refit on ALL rows: {diff:.6f}")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED)
    print(f"Shipped model R2 on the random_state={SEED} 'test' rows "
          f"(in-sample, NOT a held-out score): "
          f"{r2_score(y_te, deployed.predict(saved_pre.transform(X_te))):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-artifact", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(DATA)
    X, y = df.drop(columns=TARGET), df[TARGET]
    categorical = [c for c in X.columns if c not in NUMERIC]

    held_out(X, y, categorical)
    cross_validated(X, y, categorical)
    error_analysis(X, y, categorical)
    pearson_filter(X, y, categorical)
    app_path_effect(X, y, categorical)
    if not args.skip_artifact:
        artifact_check(X, y, categorical)


if __name__ == "__main__":
    main()
