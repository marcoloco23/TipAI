"""
Model training module for football score prediction.
Trains XGBoost and LightGBM models with Poisson objective.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not installed")

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: LightGBM not installed")

# Base directories
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Feature columns used for training
FEATURE_COLS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_goals_scored_home_avg5",
    "home_goals_conceded_home_avg5",
    "home_goals_scored_home_avg10",
    "home_goals_conceded_home_avg10",
    "home_home_wins_last5",
    "away_goals_scored_away_avg5",
    "away_goals_conceded_away_avg5",
    "away_goals_scored_away_avg10",
    "away_goals_conceded_away_avg10",
    "away_away_wins_last5",
    "home_total_goals_avg5",
    "home_total_conceded_avg5",
    "home_form_points5",
    "away_total_goals_avg5",
    "away_total_conceded_avg5",
    "away_form_points5",
    "h2h_home_goals_avg",
    "h2h_away_goals_avg",
    "h2h_matches",
]


def train_xgboost_model(X_train, y_train, X_val, y_val, target_name):
    """Train XGBoost model with Poisson objective."""
    if not HAS_XGB:
        print("XGBoost not available, skipping...")
        return None

    print(f"\nTraining XGBoost for {target_name}...")

    params = {
        "objective": "count:poisson",
        "eval_metric": "poisson-nloglik",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    model = xgb.XGBRegressor(**params)

    # Train with early stopping
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    # Evaluate
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    print(f"XGBoost {target_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    return model


def train_lightgbm_model(X_train, y_train, X_val, y_val, target_name):
    """Train LightGBM model with Poisson objective."""
    if not HAS_LGB:
        print("LightGBM not available, skipping...")
        return None

    print(f"\nTraining LightGBM for {target_name}...")

    params = {
        "objective": "poisson",
        "metric": "poisson",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMRegressor(**params)

    # Train with early stopping
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Evaluate
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    print(f"LightGBM {target_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    return model


def train_ensemble(df, feature_cols):
    """Train ensemble of XGBoost and LightGBM models."""
    print("\n" + "=" * 50)
    print("Training Ensemble Models")
    print("=" * 50)

    # Prepare data
    df = df.sort_values("Date").reset_index(drop=True)

    # Time-based split: Use last 20% for validation
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    print(f"Training set: {len(train_df)} matches")
    print(f"Validation set: {len(val_df)} matches")
    print(f"Training period: {train_df['Date'].min()} to {train_df['Date'].max()}")
    print(f"Validation period: {val_df['Date'].min()} to {val_df['Date'].max()}")

    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values

    y_train_home = train_df["HomeGoals"].values
    y_train_away = train_df["AwayGoals"].values
    y_val_home = val_df["HomeGoals"].values
    y_val_away = val_df["AwayGoals"].values

    # Train models
    models = {}

    # XGBoost models
    models["xgb_home"] = train_xgboost_model(
        X_train, y_train_home, X_val, y_val_home, "HomeGoals"
    )
    models["xgb_away"] = train_xgboost_model(
        X_train, y_train_away, X_val, y_val_away, "AwayGoals"
    )

    # LightGBM models
    models["lgb_home"] = train_lightgbm_model(
        X_train, y_train_home, X_val, y_val_home, "HomeGoals"
    )
    models["lgb_away"] = train_lightgbm_model(
        X_train, y_train_away, X_val, y_val_away, "AwayGoals"
    )

    # Evaluate ensemble
    print("\n" + "=" * 50)
    print("Ensemble Evaluation")
    print("=" * 50)

    # Get ensemble predictions
    home_preds = []
    away_preds = []

    if models["xgb_home"] is not None:
        home_preds.append(models["xgb_home"].predict(X_val))
        away_preds.append(models["xgb_away"].predict(X_val))

    if models["lgb_home"] is not None:
        home_preds.append(models["lgb_home"].predict(X_val))
        away_preds.append(models["lgb_away"].predict(X_val))

    if home_preds:
        ensemble_home = np.mean(home_preds, axis=0)
        ensemble_away = np.mean(away_preds, axis=0)

        # Calculate metrics
        home_mae = mean_absolute_error(y_val_home, ensemble_home)
        away_mae = mean_absolute_error(y_val_away, ensemble_away)

        print(f"Ensemble Home Goals MAE: {home_mae:.4f}")
        print(f"Ensemble Away Goals MAE: {away_mae:.4f}")

        # Exact score accuracy
        pred_home_rounded = np.round(ensemble_home).astype(int)
        pred_away_rounded = np.round(ensemble_away).astype(int)

        exact_correct = (
            (pred_home_rounded == y_val_home) & (pred_away_rounded == y_val_away)
        ).sum()
        exact_accuracy = exact_correct / len(y_val_home)

        print(f"Exact Score Accuracy: {exact_accuracy:.2%}")

        # Result accuracy (H/D/A)
        actual_result = np.where(
            y_val_home > y_val_away, "H", np.where(y_val_home < y_val_away, "A", "D")
        )
        pred_result = np.where(
            pred_home_rounded > pred_away_rounded,
            "H",
            np.where(pred_home_rounded < pred_away_rounded, "A", "D"),
        )

        result_accuracy = (actual_result == pred_result).mean()
        print(f"Result Accuracy (H/D/A): {result_accuracy:.2%}")

    return models, feature_cols


def save_models(models, feature_cols):
    """Save trained models to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_paths = {}
    for name, model in models.items():
        if model is not None:
            path = MODELS_DIR / f"{name}_{timestamp}.joblib"
            joblib.dump(model, path)
            saved_paths[name] = path
            print(f"Saved {name} to {path}")

    # Save feature columns
    feature_path = MODELS_DIR / f"feature_cols_{timestamp}.joblib"
    joblib.dump(feature_cols, feature_path)
    saved_paths["feature_cols"] = feature_path

    # Save latest paths for easy loading
    latest_path = MODELS_DIR / "latest_models.joblib"
    joblib.dump(saved_paths, latest_path)
    print(f"\nSaved model paths to {latest_path}")

    return saved_paths


def load_latest_models():
    """Load the most recently trained models."""
    latest_path = MODELS_DIR / "latest_models.joblib"

    if not latest_path.exists():
        raise FileNotFoundError("No trained models found. Run training first.")

    saved_paths = joblib.load(latest_path)

    models = {}
    for name, path in saved_paths.items():
        if name != "feature_cols":
            models[name] = joblib.load(path)
        else:
            feature_cols = joblib.load(path)

    return models, feature_cols


def main():
    """Main training function."""
    # Load featured data
    featured_path = PROCESSED_DATA_DIR / "featured_matches.csv"

    if not featured_path.exists():
        print(f"Error: {featured_path} not found. Run features.py first.")
        return None

    df = pd.read_csv(featured_path, parse_dates=["Date"])
    print(f"Loaded {len(df)} featured matches")

    # Train ensemble
    models, feature_cols = train_ensemble(df, FEATURE_COLS)

    # Save models
    save_models(models, feature_cols)

    return models, feature_cols


if __name__ == "__main__":
    main()
