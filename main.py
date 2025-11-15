

"""
Local Home Finder

- Trains multiple regressors (Ridge, RandomForest, GradientBoosting) on the house pricing dataset.
- Benchmarks each model with/without Linear Discriminant Analysis (LDA) dimensionality reduction.
- Saves PNG plots that compare accuracy (RMSE) and training time across feature spaces.
- Offers both a CLI workflow for affordability calculations/recommendations and a Flask UI
  where users can pick a model, toggle dimensionality reduction, and request predictions.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from flask import Flask, render_template, request
except ImportError:  # pragma: no cover - allows CLI even if Flask is absent
    Flask = None
    render_template = None
    request = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "house_prices_dataset.csv"
PLOT_DIR = ROOT_DIR / "static" / "plots"
RANDOM_STATE = 42
KM_PER_MILE = 1.60934

TARGET_COL = "price"
NUM_COLS = ["square_feet", "num_rooms", "age", "distance_to_city(km)"]
ALL_EXPECTED_COLS = NUM_COLS + [TARGET_COL]

MODEL_BUILDERS = {
    "ridge": lambda: Ridge(alpha=1.0),
    "rf": lambda: RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
    "gbr": lambda: GradientBoostingRegressor(random_state=RANDOM_STATE),
}

MODEL_LABELS = {
    "ridge": "Ridge Regression",
    "rf": "Random Forest",
    "gbr": "Gradient Boosting",
}

DIMENSION_LABELS = {False: "Original (No LDA)", True: "With LDA"}

# -----------------------------------------------------------------------------
# Finance helpers
# -----------------------------------------------------------------------------
def monthly_payment(principal: float, annual_rate: float, years: int) -> float:
    if principal <= 0:
        return 0.0
    r = (annual_rate / 100.0) / 12.0
    n = int(years * 12)
    if r == 0:
        return principal / n
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def max_affordable_price(
    annual_income: float,
    monthly_debt: float,
    down_payment: float,
    interest_rate_pct: float,
    loan_years: int,
    property_tax_pct: float = 1.2,
    insurance_per_month: float = 150.0,
    hoa_per_month: float = 0.0,
    front_end_dti: float = 0.31,
    back_end_dti: float = 0.40,
) -> Tuple[float, Dict[str, float]]:
    monthly_income = annual_income / 12.0
    piti_front_cap = front_end_dti * monthly_income
    piti_back_cap = max(0.0, back_end_dti * monthly_income - monthly_debt)
    piti_cap = min(piti_front_cap, piti_back_cap)
    if piti_cap <= 0:
        return 0.0, {"piti_cap": piti_cap}

    def total_piti(price: float) -> float:
        loan = max(0.0, price - down_payment)
        pi = monthly_payment(loan, interest_rate_pct, loan_years)
        tax = (property_tax_pct / 100.0) * price / 12.0
        return pi + tax + insurance_per_month + hoa_per_month

    lo, hi = 0.0, 5_000_000.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if total_piti(mid) <= piti_cap:
            lo = mid
        else:
            hi = mid

    price = lo
    loan = max(0.0, price - down_payment)
    pi = monthly_payment(loan, interest_rate_pct, loan_years)
    tax = (property_tax_pct / 100.0) * price / 12.0
    return price, {
        "piti_cap": piti_cap,
        "pi": pi,
        "tax": tax,
        "ins": insurance_per_month,
        "hoa": hoa_per_month,
    }


# -----------------------------------------------------------------------------
# Dataset and preprocessing helpers
# -----------------------------------------------------------------------------
class TrainedModels:
    def __init__(self, preprocessor, models, X_sim, df):
        self.preprocessor = preprocessor
        self.models = models
        self.X_sim = X_sim
        self.df = df


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in ALL_EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}\nFound: {list(df.columns)}")
    df = df.dropna(subset=[TARGET_COL])
    df = df[df[TARGET_COL] > 0]
    return df.reset_index(drop=True)


def build_column_transformer() -> ColumnTransformer:
    return ColumnTransformer([("num", StandardScaler(), NUM_COLS)])


def train_models(df: pd.DataFrame) -> TrainedModels:
    X = df[NUM_COLS]
    y = df[TARGET_COL].values

    preprocessor = build_column_transformer()

    models = {
        "ridge": Pipeline([("prep", preprocessor), ("est", Ridge(alpha=1.0))]),
        "rf": Pipeline(
            [("prep", preprocessor), ("est", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE))]
        ),
        "gbr": Pipeline(
            [("prep", preprocessor), ("est", GradientBoostingRegressor(random_state=RANDOM_STATE))]
        ),
    }

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    for m in models.values():
        m.fit(Xtr, ytr)

    preprocessor.fit(X)
    X_sim = preprocessor.transform(X)
    return TrainedModels(preprocessor, models, X_sim, df)


def ensemble_predict(models: Dict[str, Pipeline], X: pd.DataFrame) -> np.ndarray:
    preds = [m.predict(X) for m in models.values()]
    return np.mean(np.vstack(preds), axis=0)


def build_user_row(square_feet: float, num_rooms: int, age: float, distance_km: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "square_feet": square_feet,
                "num_rooms": num_rooms,
                "age": age,
                "distance_to_city(km)": distance_km,
            }
        ]
    )


def recommend(tm: TrainedModels, user_X: pd.DataFrame, max_price: float, top_k: int = 10) -> pd.DataFrame:
    preds = ensemble_predict(tm.models, tm.df[NUM_COLS])
    df = tm.df.copy()
    df["pred_price"] = preds

    affordable = df[df["pred_price"] <= max_price].copy()
    if affordable.empty:
        return pd.DataFrame()

    user_vec = tm.preprocessor.transform(user_X)
    aff_vecs = tm.preprocessor.transform(affordable[NUM_COLS])
    sims = cosine_similarity(user_vec, aff_vecs)[0]

    affordable["similarity"] = sims
    affordable["score"] = sims + (max_price - affordable["pred_price"]) / max_price
    affordable["distance_to_city_miles"] = affordable["distance_to_city(km)"] / KM_PER_MILE
    affordable = affordable.sort_values("score", ascending=False).head(top_k)
    return affordable.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Dimensionality reduction + evaluation helpers
# -----------------------------------------------------------------------------
class PriceLDATransformer(BaseEstimator, TransformerMixin):
    """Fits LDA on price quantiles to create a supervised 2D projection for regression input."""

    def __init__(self, n_components: int = 2, quantiles: int = 3):
        self.n_components = n_components
        self.quantiles = quantiles
        self.lda: LinearDiscriminantAnalysis | None = None
        self.enabled = False

    def fit(self, X, y):
        y_series = pd.Series(y)
        try:
            y_bins = pd.qcut(y_series, q=self.quantiles, labels=False, duplicates="drop")
        except ValueError:
            self.enabled = False
            self.lda = None
            return self

        unique_classes = pd.unique(y_bins)
        if len(unique_classes) <= 1:
            self.enabled = False
            self.lda = None
            return self

        n_components = min(self.n_components, len(unique_classes) - 1, X.shape[1])
        if n_components <= 0:
            self.enabled = False
            self.lda = None
            return self

        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        self.lda.fit(X, np.asarray(y_bins, dtype=int))
        self.enabled = True
        return self

    def transform(self, X):
        if not self.enabled or self.lda is None:
            return X
        return self.lda.transform(X)


def build_pipeline(model, use_dim_reduction: bool) -> Pipeline:
    steps = [("prep", build_column_transformer())]
    if use_dim_reduction:
        steps.append(("lda", PriceLDATransformer()))
    steps.append(("est", model))
    return Pipeline(steps)


def train_and_evaluate_models(df: pd.DataFrame):
    X = df[NUM_COLS]
    y = df[TARGET_COL].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    records = []
    model_store = {}

    for use_dim in (False, True):
        for key, builder in MODEL_BUILDERS.items():
            model = builder()
            pipeline = build_pipeline(model, use_dim)

            fit_start = perf_counter()
            pipeline.fit(Xtr, ytr)
            fit_time = perf_counter() - fit_start

            pred_start = perf_counter()
            preds = pipeline.predict(Xte)
            pred_time = perf_counter() - pred_start

            mse = mean_squared_error(yte, preds)
            records.append(
                {
                    "model": key,
                    "model_label": MODEL_LABELS[key],
                    "dim_reduction": use_dim,
                    "dim_label": DIMENSION_LABELS[use_dim],
                    "rmse": math.sqrt(mse),
                    "mae": mean_absolute_error(yte, preds),
                    "r2": r2_score(yte, preds),
                    "fit_time": fit_time,
                    "predict_time": pred_time,
                }
            )

            final_pipeline = build_pipeline(builder(), use_dim)
            final_pipeline.fit(X, y)
            model_store[(key, use_dim)] = final_pipeline

    metrics_df = pd.DataFrame(records)
    return model_store, metrics_df


def save_metric_plots(metrics_df: pd.DataFrame) -> None:
    if metrics_df.empty:
        return
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    def _plot(metric: str, ylabel: str, filename: str):
        pivot = metrics_df.pivot(index="model_label", columns="dim_label", values=metric)
        fig, ax = plt.subplots(figsize=(8, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model")
        ax.set_title(f"{ylabel} comparison")
        ax.legend(title="Feature Space")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / filename, bbox_inches="tight")
        plt.close(fig)

    plot_specs = [
        ("rmse", "RMSE (lower is better)", "model_accuracy_rmse.png"),
        ("mae", "MAE (lower is better)", "model_accuracy_mae.png"),
        ("r2", "R² (higher is better)", "model_accuracy_r2.png"),
        ("fit_time", "Training time (seconds)", "model_training_time.png"),
        ("predict_time", "Prediction time (seconds)", "model_inference_time.png"),
    ]
    for metric, label, filename in plot_specs:
        _plot(metric, label, filename)


# -----------------------------------------------------------------------------
# CLI helpers (input + prompts)
# -----------------------------------------------------------------------------
def ask_float(prompt, default=None):
    while True:
        raw = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if not raw and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")


def ask_int(prompt, default=None):
    while True:
        raw = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if not raw and default is not None:
            return int(default)
        try:
            return int(raw)
        except ValueError:
            print("Please enter an integer.")


# -----------------------------------------------------------------------------
# Artifact management + prediction API
# -----------------------------------------------------------------------------
ARTIFACTS: Dict[str, object] = {}


def ensure_artifacts() -> Dict[str, object]:
    if ARTIFACTS:
        return ARTIFACTS
    df = load_and_clean(DATA_PATH)
    tm = train_models(df)
    model_store, metrics_df = train_and_evaluate_models(df)
    save_metric_plots(metrics_df)
    ARTIFACTS.update({"df": df, "tm": tm, "model_store": model_store, "metrics_df": metrics_df})
    return ARTIFACTS


def predict_price_from_store(model_key: str, use_dim: bool, *, square_feet: float, num_rooms: int, age: float, distance_miles: float) -> float:
    artifacts = ensure_artifacts()
    model = artifacts["model_store"].get((model_key, use_dim))
    if model is None:
        raise ValueError("Requested model configuration is not available.")
    user_df = build_user_row(square_feet, num_rooms, age, distance_miles * KM_PER_MILE)
    return float(model.predict(user_df)[0])


# -----------------------------------------------------------------------------
# Flask UI
# -----------------------------------------------------------------------------
if Flask is not None:
    app = Flask(
        __name__,
        static_folder=str(ROOT_DIR / "static"),
        template_folder=str(ROOT_DIR / "templates"),
    )

    @app.route("/", methods=["GET", "POST"])
    def index():
        artifacts = ensure_artifacts()
        model_options = [(key, MODEL_LABELS[key]) for key in MODEL_BUILDERS.keys()]

        form_values = {
            "square_feet": request.form.get("square_feet", "1500"),
            "num_rooms": request.form.get("num_rooms", "4"),
            "age": request.form.get("age", "5"),
            "distance": request.form.get("distance", "10"),
        }
        selected_model = request.form.get("model", model_options[0][0])
        use_dim = request.form.get("use_dim") == "on"
        prediction = None
        error = None

        if request.method == "POST":
            try:
                prediction = predict_price_from_store(
                    selected_model,
                    use_dim,
                    square_feet=float(form_values["square_feet"]),
                    num_rooms=int(form_values["num_rooms"]),
                    age=float(form_values["age"]),
                    distance_miles=float(form_values["distance"]),
                )
            except ValueError as exc:
                error = str(exc)
            except Exception as exc:  # pragma: no cover - user input errors
                error = f"Unable to generate prediction: {exc}"

        metrics_df: pd.DataFrame = artifacts["metrics_df"]
        metrics_records = metrics_df.sort_values(["model_label", "dim_label"]).to_dict("records")

        return render_template(
            "index.html",
            model_options=model_options,
            selected_model=selected_model,
            use_dim=use_dim,
            prediction=prediction,
            error=error,
            form_values=form_values,
            metrics=metrics_records,
        )
else:
    app = None


# -----------------------------------------------------------------------------
# CLI workflow
# -----------------------------------------------------------------------------
def run_cli(artifacts: Dict[str, object]) -> None:
    tm: TrainedModels = artifacts["tm"]
    print("\n=== Local Home Finder (CLI) ===")
    print(f"Loaded {len(tm.df):,} rows. Models trained.\n")

    print("Enter your financial info:")
    annual_income = ask_float("Annual gross income (USD)", 85000)
    monthly_debt = ask_float("Monthly non-housing debt (USD)", 250)
    down_payment = ask_float("Down payment (USD)", 40000)
    rate = ask_float("Mortgage interest rate (%)", 6.5)
    years = ask_int("Loan term (years)", 30)

    max_price, piti = max_affordable_price(annual_income, monthly_debt, down_payment, rate, years)
    print(f"\nMax affordable price: ${max_price:,.0f}")
    print(f"Approx monthly P&I: ${piti['pi']:,.0f} | Taxes: ${piti['tax']:,.0f}\n")

    print("Enter your home wish-list:")
    square_feet = ask_float("Square feet", 1500)
    num_rooms = ask_int("Number of rooms", 4)
    age = ask_float("Age of home (years)", 5)
    distance_miles = ask_float("Distance to city center (miles)", 10)
    distance_km = distance_miles * KM_PER_MILE

    user_X = build_user_row(square_feet, num_rooms, age, distance_km)
    user_pred = ensemble_predict(tm.models, user_X)[0]
    print(f"\nPredicted price for your desired home: ${user_pred:,.0f}")
    if user_pred <= max_price:
        print("Within your affordability.")
    else:
        print(f"Exceeds affordability by about ${user_pred - max_price:,.0f}.")

    recs = recommend(tm, user_X, max_price, top_k=10)
    print("\nTop recommended affordable homes:\n")
    if recs.empty:
        print("No matches within budget.")
    else:
        display_cols = ["square_feet", "num_rooms", "age", "distance_to_city_miles", "pred_price"]
        print(recs[display_cols])


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Local Home Finder utilities")
    parser.add_argument(
        "--mode",
        choices=["cli", "serve"],
        default="cli",
        help="Run the interactive CLI or launch the Flask server.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Flask host when using --mode serve.")
    parser.add_argument("--port", type=int, default=5000, help="Flask port when using --mode serve.")
    args = parser.parse_args(argv)

    artifacts = ensure_artifacts()

    if args.mode == "serve":
        if app is None:
            print("Flask is not installed. Please `pip install flask` to use the web UI.")
            return
        print(f"Launching Flask app on http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False)
    else:
        run_cli(artifacts)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
