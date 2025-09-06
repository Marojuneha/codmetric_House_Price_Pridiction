import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt


# ---------- Config ----------
RANDOM_STATE = 42
TEST_SIZE = 0.2
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
CUSTOM_CSV = DATA_DIR / "house_prices.csv"  # If present, code will use it
TARGET_NAME = "target"  # For custom CSV, set/rename the target column to this
# ----------------------------


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """
    Loads a dataset as a Pandas DataFrame.
    - If data/house_prices.csv exists, use that (expects a column named 'target').
    - Otherwise, fetch scikit-learn's California Housing dataset and return it with
      target column named 'target' for consistency.
    """
    if CUSTOM_CSV.exists():
        logging.info(f"Loading custom dataset from {CUSTOM_CSV}")
        df = pd.read_csv(CUSTOM_CSV)
        if TARGET_NAME not in df.columns:
            raise ValueError(
                f"Custom CSV must have a '{TARGET_NAME}' column as the target."
            )
        return df

    logging.info("Custom dataset not found. Fetching California Housing dataset...")
    cali = fetch_california_housing(as_frame=True)
    df = cali.frame.copy()
    # Rename the target to a generic name for pipeline consistency
    df = df.rename(columns={"MedHouseVal": TARGET_NAME})
    logging.info("California Housing dataset loaded.")
    return df


def train_and_evaluate(df: pd.DataFrame):
    # Separate features/target
    if TARGET_NAME not in df.columns:
        raise ValueError(f"Target column '{TARGET_NAME}' not found in DataFrame.")
    X = df.drop(columns=[TARGET_NAME])
    y = df[TARGET_NAME]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Identify numeric columns (handles both custom CSV and California dataset)
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_features) != X_train.shape[1]:
        logging.info(
            "Non-numeric columns detected. They will be dropped for this simple baseline."
        )
        X_train = X_train[numeric_features]
        X_test = X_test[numeric_features]

    # Preprocess: impute missing with median + scale
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop",
    )

    # Model: Linear Regression
    model = LinearRegression()

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # Fit
    logging.info("Fitting the pipeline...")
    pipeline.fit(X_train, y_train)

    # Predict
    logging.info("Predicting on test set...")
    y_pred = pipeline.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"MSE  : {mse:.6f}")
    logging.info(f"RMSE : {rmse:.6f}")
    logging.info(f"R^2  : {r2:.6f}")

    # Save metrics
    metrics_path = OUTPUT_DIR / "metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"MSE: {mse:.6f}\nRMSE: {rmse:.6f}\nR2: {r2:.6f}\n")
    logging.info(f"Metrics saved to {metrics_path}")

    # Save predictions
    preds_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
    preds_path = OUTPUT_DIR / "predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    logging.info(f"Predictions saved to {preds_path}")

    # Plot: Predicted vs Actual
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    pva_path = OUTPUT_DIR / "pred_vs_actual.png"
    plt.tight_layout()
    plt.savefig(pva_path, dpi=150)
    plt.close()
    logging.info(f"Predicted vs Actual plot saved to {pva_path}")

    # Plot: Residuals
    residuals = y_test.values - y_pred
    plt.figure(figsize=(7, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residuals vs Predicted")
    res_path = OUTPUT_DIR / "residuals.png"
    plt.tight_layout()
    plt.savefig(res_path, dpi=150)
    plt.close()
    logging.info(f"Residuals plot saved to {res_path}")

    # Save model
    model_path = OUTPUT_DIR / "linear_regression_pipeline.joblib"
    dump(pipeline, model_path)
    logging.info(f"Trained pipeline saved to {model_path}")

    return {"mse": mse, "rmse": rmse, "r2": r2}


def main():
    setup_logging()
    ensure_dirs()
    df = load_dataset()

    # Basic cleaning (duplicates & sanity)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    if after != before:
        logging.info(f"Dropped {before - after} duplicate rows.")

    results = train_and_evaluate(df)
    logging.info("Done.")
    logging.info(f"Final -> MSE: {results['mse']:.6f} | RMSE: {results['rmse']:.6f} | R2: {results['r2']:.6f}")


if __name__ == "__main__":
    main()
