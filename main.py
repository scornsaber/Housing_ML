

"""
Local Home Finder (Simplified for Kaggle House Price Prediction Dataset)
Dataset columns: ['square_feet', 'num_rooms', 'age', 'distance_to_city(km)', 'price']

Trains 3 regression models (Ridge, RandomForest, GradientBoosting)
Computes affordability from user finances and recommends affordable homes.
"""

from __future__ import annotations
import math
import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# CONFIG 
DATA_PATH = r"/home/scorn556/Housing_ML/house_prices_dataset.csv"  # change if needed
RANDOM_STATE = 42

TARGET_COL = "price"
NUM_COLS = ["square_feet", "num_rooms", "age", "distance_to_city(km)"]
BIN_COLS = []
CAT_COLS = []
ALL_EXPECTED_COLS = NUM_COLS + [TARGET_COL]

# FINANCE HELPERS 
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
    return price, {"piti_cap": piti_cap, "pi": pi, "tax": tax,
                   "ins": insurance_per_month, "hoa": hoa_per_month}

# MODEL TRAINING 
class TrainedModels:
    def __init__(self, preprocessor, models, X_sim, df):
        self.preprocessor = preprocessor
        self.models = models
        self.X_sim = X_sim
        self.df = df

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in ALL_EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}\nFound: {list(df.columns)}")
    df = df.dropna(subset=[TARGET_COL])
    df = df[df[TARGET_COL] > 0]
    return df.reset_index(drop=True)

def train_models(df: pd.DataFrame) -> TrainedModels:
    X = df[NUM_COLS]
    y = df[TARGET_COL].values

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS)
    ])

    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    gbr = GradientBoostingRegressor(random_state=RANDOM_STATE)

    models = {
        "ridge": Pipeline([("prep", preprocessor), ("est", ridge)]),
        "rf": Pipeline([("prep", preprocessor), ("est", rf)]),
        "gbr": Pipeline([("prep", preprocessor), ("est", gbr)]),
    }

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    for m in models.values():
        m.fit(Xtr, ytr)

    X_sim = preprocessor.fit_transform(X)
    return TrainedModels(preprocessor, models, X_sim, df)

def ensemble_predict(models: Dict[str, Pipeline], X: pd.DataFrame) -> np.ndarray:
    preds = [m.predict(X) for m in models.values()]
    return np.mean(np.vstack(preds), axis=0)

#  USER / RECOMMEND 
def build_user_row(square_feet: float, num_rooms: int, age: float, distance: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "square_feet": square_feet,
        "num_rooms": num_rooms,
        "age": age,
        "distance_to_city(km)": distance
    }])

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
    affordable["score"] = (sims + (max_price - affordable["pred_price"]) / max_price)
    affordable = affordable.sort_values("score", ascending=False).head(top_k)
    return affordable.reset_index(drop=True)

#  CLI HELPERS 
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

#  MAIN 
def main():
    print("\n=== Local Home Finder (Simplified) ===")
    df = load_and_clean(DATA_PATH)
    tm = train_models(df)
    print(f"Loaded {len(df):,} rows. Models trained.\n")

    # Finance 
    print("Enter your financial info:")
    annual_income = ask_float("Annual gross income (USD)", 85000)
    monthly_debt = ask_float("Monthly non-housing debt (USD)", 250)
    down_payment = ask_float("Down payment (USD)", 40000)
    rate = ask_float("Mortgage interest rate (%)", 6.5)
    years = ask_int("Loan term (years)", 30)

    max_price, piti = max_affordable_price(
        annual_income, monthly_debt, down_payment, rate, years
    )
    print(f"\nMax affordable price: ${max_price:,.0f}")
    print(f"Approx monthly P&I: ${piti['pi']:,.0f} | Taxes: ${piti['tax']:,.0f}\n")

    # Wish-list 
    print("Enter your home wish-list:")
    square_feet = ask_float("Square feet", 1500)
    num_rooms = ask_int("Number of rooms", 4)
    age = ask_float("Age of home (years)", 5)
    distance = ask_float("Distance to city center (Miles)", 10)
    distance = distance*1.61

    user_X = build_user_row(square_feet, num_rooms, age, distance)
    user_pred = ensemble_predict(tm.models, user_X)[0]
    print(f"\nPredicted price for your desired home: ${user_pred:,.0f}")
    if user_pred <= max_price:
        print("Within your affordability.")
    else:
        print(f"Exceeds affordability by about ${user_pred - max_price:,.0f}.")

    # Recommendations 
    recs = recommend(tm, user_X, max_price, top_k=10)
    print("\nTop recommended affordable homes:\n")
    if recs.empty:
        print("No matches within budget.")
    else:
        print(recs[["square_feet", "num_rooms", "age", "distance_to_city(Miles)", "pred_price"]])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
