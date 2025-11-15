# Local Home Finder

Predicts home prices from a Kaggle-style dataset, benchmarks several regression models with and without dimensionality reduction, and exposes both a CLI workflow and a lightweight web UI so users can explore affordability and make predictions interactively.

## Dataset
- Expected file: `house_prices_dataset.csv` in the project root.
- Columns (case-insensitive): `square_feet`, `num_rooms`, `age`, `distance_to_city(km)`, `price`.
- Records with missing or non-positive prices are discarded automatically.

## Feature Highlights

- **Model ensemble** – Fits Ridge, Random Forest, and Gradient Boosting regressors. Each model is trained twice: once on standardized numeric features and once after a supervised Linear Discriminant Analysis (LDA) projection.
- **Benchmark plots** – Saves bar charts comparing RMSE, MAE, R², training time, and inference time for every (model, feature-space) pair under `static/plots/`. These PNGs are ready to embed in articles or presentations.
- **Affordability calculator** – The CLI estimates mortgage affordability given income, debt, down payment, and rate assumptions, then checks whether a user-defined “dream home” stays within budget.
- **Recommendation engine** – Suggests similar affordable homes by blending cosine similarity with how far listings fall under the user’s maximum price.
- **Web UI** – A Flask app that lets users:
  - Enter property attributes.
  - Choose the model via dropdown.
  - Toggle dimensionality reduction on/off.
  - View the resulting price plus per-model metrics in tabular form.
  - See contextual notes explaining how LDA influences each model when the toggle is active.

## Requirements
- Python 3.10+.
- Install dependencies (from the project root):
  ```bash
  pip install numpy pandas scikit-learn matplotlib flask
  ```

## Usage

### 1. Generate models and plots
Run once to train all pipelines (with and without LDA), cache them, and produce the PNG charts:
```bash
python -c "import main; main.ensure_artifacts()"
```

### 2. Interactive CLI (affordability + recommendations)
```bash
python main.py --mode cli
```
- Prompts for financials, computes a max affordable price, predicts the user’s desired home, and prints top affordable matches (converted to miles for readability).

### 3. Web UI for predictions
```bash
python main.py --mode serve --host 127.0.0.1 --port 5000
```
- Visit `http://127.0.0.1:5000/`.
- Enter home attributes (square feet, rooms, age, distance in miles).
- Select Ridge/Random Forest/Gradient Boosting and optionally enable dimensionality reduction.
- Submit to see the predicted price and a short note on how LDA impacts the selected model.
- Review the benchmarking table to compare RMSE/MAE/R² and timing across all configurations.

## Project Structure
```
Housing_ML-main/
├── house_prices_dataset.csv     # Input data
├── main.py                      # CLI, training pipeline, and Flask app
├── templates/
│   └── index.html               # Web UI template
└── static/
    ├── plots/                   # Generated PNG charts (created on first run)
    └── .gitkeep
```

## Notes
- Distance inputs in the UI use miles, but the model expects kilometers; conversion is handled internally.
- `ensure_artifacts()` caches the trained models for reuse, so later CLI/UI runs start instantly. Delete the `static/plots/` files or restart Python to force retraining.
