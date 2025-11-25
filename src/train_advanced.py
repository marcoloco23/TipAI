"""
Advanced model training with:
- Enhanced feature engineering (momentum, streaks, EWA, efficiency)
- Time-series cross-validation
- XGBoost, LightGBM, CatBoost ensemble
- Optuna hyperparameter tuning
- Model stacking with meta-learner
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
from collections import defaultdict
from tqdm import tqdm
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Elo config
ELO_K = 32  # Increased for faster adaptation
ELO_HOME_ADV = 100
INITIAL_ELO = 1500


class AdvancedFeatureBuilder:
    """Build comprehensive features for match prediction."""

    def __init__(self):
        self.elo_ratings = defaultdict(lambda: INITIAL_ELO)
        self.team_matches = defaultdict(list)
        self.h2h_matches = defaultdict(list)

    def reset(self):
        """Reset all state for fresh feature building."""
        self.elo_ratings = defaultdict(lambda: INITIAL_ELO)
        self.team_matches = defaultdict(list)
        self.h2h_matches = defaultdict(list)

    def update_elo(self, home, away, home_goals, away_goals):
        """Update Elo ratings after match."""
        home_elo = self.elo_ratings[home] + ELO_HOME_ADV
        away_elo = self.elo_ratings[away]

        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))

        if home_goals > away_goals:
            actual = 1
        elif home_goals < away_goals:
            actual = 0
        else:
            actual = 0.5

        # Goal difference multiplier
        gd = abs(home_goals - away_goals)
        mult = 1 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)

        delta = ELO_K * mult * (actual - expected_home)
        self.elo_ratings[home] += delta
        self.elo_ratings[away] -= delta

    def ewm_mean(self, values, span=5):
        """Exponential weighted mean - recent matches matter more."""
        if len(values) == 0:
            return None
        weights = np.exp(np.linspace(-1, 0, len(values)))
        weights = weights[-min(len(values), span * 2) :]
        vals = values[-min(len(values), span * 2) :]
        return np.average(vals, weights=weights)

    def calculate_streak(self, results):
        """Calculate current winning/losing streak."""
        if not results:
            return 0, 0
        streak = 0
        last_result = results[-1]
        for r in reversed(results):
            if r == last_result:
                streak += 1
            else:
                break
        win_streak = streak if last_result == "W" else 0
        lose_streak = streak if last_result == "L" else 0
        return win_streak, lose_streak

    def get_team_features(self, team, is_home, opponent):
        """Get comprehensive features for a team."""
        matches = self.team_matches[team]
        features = {}

        # Filter home/away specific
        if is_home:
            venue_matches = [m for m in matches if m["is_home"]]
        else:
            venue_matches = [m for m in matches if not m["is_home"]]

        # Basic stats - multiple windows
        for window in [3, 5, 10]:
            recent = matches[-window:] if len(matches) >= window else matches
            venue_recent = (
                venue_matches[-window:]
                if len(venue_matches) >= window
                else venue_matches
            )

            if recent:
                gf = [m["goals_for"] for m in recent]
                ga = [m["goals_against"] for m in recent]
                features[f"goals_for_avg{window}"] = np.mean(gf)
                features[f"goals_against_avg{window}"] = np.mean(ga)
                features[f"goal_diff_avg{window}"] = np.mean(gf) - np.mean(ga)
                features[f"clean_sheets{window}"] = sum(1 for g in ga if g == 0)
                features[f"failed_to_score{window}"] = sum(1 for g in gf if g == 0)

                # Points
                pts = [m["points"] for m in recent]
                features[f"points{window}"] = sum(pts)
                features[f"ppg{window}"] = np.mean(pts)
            else:
                features[f"goals_for_avg{window}"] = 1.4
                features[f"goals_against_avg{window}"] = 1.2
                features[f"goal_diff_avg{window}"] = 0.2
                features[f"clean_sheets{window}"] = 1
                features[f"failed_to_score{window}"] = 1
                features[f"points{window}"] = window
                features[f"ppg{window}"] = 1.0

            # Venue specific
            if venue_recent:
                features[f"venue_gf_avg{window}"] = np.mean(
                    [m["goals_for"] for m in venue_recent]
                )
                features[f"venue_ga_avg{window}"] = np.mean(
                    [m["goals_against"] for m in venue_recent]
                )
            else:
                features[f"venue_gf_avg{window}"] = 1.5 if is_home else 1.2
                features[f"venue_ga_avg{window}"] = 1.0 if is_home else 1.5

        # Exponential weighted averages (recency bias)
        if matches:
            gf = [m["goals_for"] for m in matches]
            ga = [m["goals_against"] for m in matches]
            features["gf_ewm"] = self.ewm_mean(gf, span=5)
            features["ga_ewm"] = self.ewm_mean(ga, span=5)
        else:
            features["gf_ewm"] = 1.4
            features["ga_ewm"] = 1.2

        # Momentum & streaks
        if matches:
            results = [m["result"] for m in matches]
            win_streak, lose_streak = self.calculate_streak(results)
            features["win_streak"] = win_streak
            features["lose_streak"] = lose_streak

            # Recent form momentum (weighted recent results)
            recent_results = results[-5:]
            momentum = sum(
                1 if r == "W" else (-1 if r == "L" else 0) for r in recent_results
            )
            features["momentum"] = momentum
        else:
            features["win_streak"] = 0
            features["lose_streak"] = 0
            features["momentum"] = 0

        features["days_rest"] = 7

        # Attack/defense ratings
        if len(matches) >= 5:
            features["attack_rating"] = features["gf_ewm"] / 1.4  # Normalized
            features["defense_rating"] = 1.2 / max(features["ga_ewm"], 0.5)
        else:
            features["attack_rating"] = 1.0
            features["defense_rating"] = 1.0

        # Head-to-head
        h2h_key = tuple(sorted([team, opponent]))
        h2h = self.h2h_matches[h2h_key]
        team_h2h = [m for m in h2h if m["team"] == team]

        if len(team_h2h) >= 2:
            features["h2h_gf_avg"] = np.mean([m["goals_for"] for m in team_h2h[-5:]])
            features["h2h_ga_avg"] = np.mean(
                [m["goals_against"] for m in team_h2h[-5:]]
            )
            features["h2h_wins"] = sum(1 for m in team_h2h[-5:] if m["result"] == "W")
        else:
            features["h2h_gf_avg"] = 1.3
            features["h2h_ga_avg"] = 1.3
            features["h2h_wins"] = 1

        return features

    def build_match_features(self, row):
        """Build all features for a single match."""
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        features = {
            # Elo ratings
            "home_elo": self.elo_ratings[home],
            "away_elo": self.elo_ratings[away],
            "elo_diff": self.elo_ratings[home] - self.elo_ratings[away],
        }

        # Home team features
        home_feats = self.get_team_features(home, is_home=True, opponent=away)
        for k, v in home_feats.items():
            features[f"home_{k}"] = v

        # Away team features
        away_feats = self.get_team_features(away, is_home=False, opponent=home)
        for k, v in away_feats.items():
            features[f"away_{k}"] = v

        # Interaction features
        features["attack_vs_defense_home"] = (
            home_feats["attack_rating"] * away_feats["defense_rating"]
        )
        features["attack_vs_defense_away"] = (
            away_feats["attack_rating"] * home_feats["defense_rating"]
        )
        features["momentum_diff"] = home_feats["momentum"] - away_feats["momentum"]
        features["form_diff"] = home_feats.get("ppg5", 1) - away_feats.get("ppg5", 1)

        return features

    def update_history(self, row):
        """Update match history after processing."""
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        hg = row["HomeGoals"]
        ag = row["AwayGoals"]
        date = row["Date"]

        # Determine results
        if hg > ag:
            home_result, away_result = "W", "L"
            home_pts, away_pts = 3, 0
        elif hg < ag:
            home_result, away_result = "L", "W"
            home_pts, away_pts = 0, 3
        else:
            home_result, away_result = "D", "D"
            home_pts, away_pts = 1, 1

        # Update team history
        self.team_matches[home].append(
            {
                "date": date,
                "opponent": away,
                "is_home": True,
                "goals_for": hg,
                "goals_against": ag,
                "result": home_result,
                "points": home_pts,
            }
        )

        self.team_matches[away].append(
            {
                "date": date,
                "opponent": home,
                "is_home": False,
                "goals_for": ag,
                "goals_against": hg,
                "result": away_result,
                "points": away_pts,
            }
        )

        # Update h2h
        h2h_key = tuple(sorted([home, away]))
        self.h2h_matches[h2h_key].append(
            {
                "date": date,
                "team": home,
                "goals_for": hg,
                "goals_against": ag,
                "result": home_result,
            }
        )
        self.h2h_matches[h2h_key].append(
            {
                "date": date,
                "team": away,
                "goals_for": ag,
                "goals_against": hg,
                "result": away_result,
            }
        )

        # Update Elo
        self.update_elo(home, away, hg, ag)


def build_dataset(df):
    """Build complete feature dataset."""
    print("Building advanced features...")
    df = df.sort_values("Date").reset_index(drop=True)

    builder = AdvancedFeatureBuilder()
    feature_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Feature engineering"):
        # Build features BEFORE updating history (no leakage)
        features = builder.build_match_features(row)
        features["match_idx"] = idx
        features["home_goals"] = row["HomeGoals"]
        features["away_goals"] = row["AwayGoals"]
        feature_rows.append(features)

        # Update history AFTER feature extraction
        builder.update_history(row)

    feature_df = pd.DataFrame(feature_rows)

    # Save Elo ratings
    elo_df = pd.DataFrame(
        [
            {"team": team, "elo_rating": rating}
            for team, rating in builder.elo_ratings.items()
        ]
    )
    elo_df.to_csv(PROCESSED_DATA_DIR / "elo_ratings.csv", index=False)

    return feature_df, builder


def get_feature_cols(df):
    """Get feature column names."""
    exclude = ["match_idx", "home_goals", "away_goals"]
    return [c for c in df.columns if c not in exclude]


def objective_xgb(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for XGBoost."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
        "n_jobs": -1,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred = model.predict(X_val)
    return mean_absolute_error(y_val, pred)


def objective_lgb(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for LightGBM."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    pred = model.predict(X_val)
    return mean_absolute_error(y_val, pred)


def objective_cat(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for CatBoost."""
    params = {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "random_strength": trial.suggest_float("random_strength", 0, 1),
        "random_seed": 42,
        "verbose": False,
    }

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    pred = model.predict(X_val)
    return mean_absolute_error(y_val, pred)


def tune_model(model_type, X_train, y_train, X_val, y_val, n_trials=30):
    """Tune hyperparameters with Optuna."""
    if model_type == "xgb":
        objective = lambda t: objective_xgb(t, X_train, y_train, X_val, y_val)
    elif model_type == "lgb":
        objective = lambda t: objective_lgb(t, X_train, y_train, X_val, y_val)
    else:
        objective = lambda t: objective_cat(t, X_train, y_train, X_val, y_val)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params


def train_models(feature_df, n_tune_trials=30):
    """Train all models with time-series CV and Optuna tuning."""
    print("\n" + "=" * 60)
    print("TRAINING ADVANCED MODELS")
    print("=" * 60)

    feature_cols = get_feature_cols(feature_df)
    X = feature_df[feature_cols].values
    y_home = feature_df["home_goals"].values
    y_away = feature_df["away_goals"].values

    # Use recent data for validation (last 20%)
    n = len(X)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)

    X_train = X[:train_size]
    X_val = X[train_size : train_size + val_size]
    X_test = X[train_size + val_size :]

    y_home_train = y_home[:train_size]
    y_home_val = y_home[train_size : train_size + val_size]
    y_home_test = y_home[train_size + val_size :]

    y_away_train = y_away[:train_size]
    y_away_val = y_away[train_size : train_size + val_size]
    y_away_test = y_away[train_size + val_size :]

    print(f"\nData split: Train={train_size}, Val={val_size}, Test={len(X_test)}")

    models = {}
    predictions = {"train": {}, "val": {}, "test": {}}

    # Train home goals models
    print("\n--- TRAINING HOME GOALS MODELS ---")

    for model_type in ["xgb", "lgb", "cat"]:
        print(f"\nTuning {model_type.upper()} for home goals...")
        best_params = tune_model(
            model_type, X_train, y_home_train, X_val, y_home_val, n_tune_trials
        )

        print(f"Training {model_type.upper()} with best params...")
        if model_type == "xgb":
            model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        elif model_type == "lgb":
            model = lgb.LGBMRegressor(
                **best_params, random_state=42, n_jobs=-1, verbose=-1
            )
        else:
            model = CatBoostRegressor(**best_params, random_seed=42, verbose=False)

        model.fit(X_train, y_home_train)
        models[f"{model_type}_home"] = model

        predictions["train"][f"{model_type}_home"] = model.predict(X_train)
        predictions["val"][f"{model_type}_home"] = model.predict(X_val)
        predictions["test"][f"{model_type}_home"] = model.predict(X_test)

    # Train away goals models
    print("\n--- TRAINING AWAY GOALS MODELS ---")

    for model_type in ["xgb", "lgb", "cat"]:
        print(f"\nTuning {model_type.upper()} for away goals...")
        best_params = tune_model(
            model_type, X_train, y_away_train, X_val, y_away_val, n_tune_trials
        )

        print(f"Training {model_type.upper()} with best params...")
        if model_type == "xgb":
            model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        elif model_type == "lgb":
            model = lgb.LGBMRegressor(
                **best_params, random_state=42, n_jobs=-1, verbose=-1
            )
        else:
            model = CatBoostRegressor(**best_params, random_seed=42, verbose=False)

        model.fit(X_train, y_away_train)
        models[f"{model_type}_away"] = model

        predictions["train"][f"{model_type}_away"] = model.predict(X_train)
        predictions["val"][f"{model_type}_away"] = model.predict(X_val)
        predictions["test"][f"{model_type}_away"] = model.predict(X_test)

    # Train meta-learner (stacking)
    print("\n--- TRAINING META-LEARNER (STACKING) ---")

    # Stack predictions for meta-learner
    meta_val_home = np.column_stack(
        [predictions["val"][f"{m}_home"] for m in ["xgb", "lgb", "cat"]]
    )
    meta_test_home = np.column_stack(
        [predictions["test"][f"{m}_home"] for m in ["xgb", "lgb", "cat"]]
    )
    meta_val_away = np.column_stack(
        [predictions["val"][f"{m}_away"] for m in ["xgb", "lgb", "cat"]]
    )
    meta_test_away = np.column_stack(
        [predictions["test"][f"{m}_away"] for m in ["xgb", "lgb", "cat"]]
    )

    # Use validation set to train meta-learner
    meta_home = Ridge(alpha=1.0)
    meta_home.fit(meta_val_home, y_home_val)
    models["meta_home"] = meta_home

    meta_away = Ridge(alpha=1.0)
    meta_away.fit(meta_val_away, y_away_val)
    models["meta_away"] = meta_away

    print(f"Meta-learner weights (home): {meta_home.coef_}")
    print(f"Meta-learner weights (away): {meta_away.coef_}")

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)

    # Individual model predictions
    for model_type in ["xgb", "lgb", "cat"]:
        pred_home = predictions["test"][f"{model_type}_home"]
        pred_away = predictions["test"][f"{model_type}_away"]

        mae_home = mean_absolute_error(y_home_test, pred_home)
        mae_away = mean_absolute_error(y_away_test, pred_away)

        exact = np.sum(
            (np.round(pred_home) == y_home_test) & (np.round(pred_away) == y_away_test)
        )
        exact_pct = exact / len(y_home_test) * 100

        print(
            f"\n{model_type.upper()}: MAE home={mae_home:.3f}, away={mae_away:.3f}, Exact={exact_pct:.1f}%"
        )

    # Ensemble (simple average)
    ensemble_home = np.mean(
        [predictions["test"][f"{m}_home"] for m in ["xgb", "lgb", "cat"]], axis=0
    )
    ensemble_away = np.mean(
        [predictions["test"][f"{m}_away"] for m in ["xgb", "lgb", "cat"]], axis=0
    )

    mae_home = mean_absolute_error(y_home_test, ensemble_home)
    mae_away = mean_absolute_error(y_away_test, ensemble_away)
    exact = np.sum(
        (np.round(ensemble_home) == y_home_test)
        & (np.round(ensemble_away) == y_away_test)
    )
    exact_pct = exact / len(y_home_test) * 100

    print(
        f"\nENSEMBLE (avg): MAE home={mae_home:.3f}, away={mae_away:.3f}, Exact={exact_pct:.1f}%"
    )

    # Stacked predictions
    stacked_home = meta_home.predict(meta_test_home)
    stacked_away = meta_away.predict(meta_test_away)

    mae_home = mean_absolute_error(y_home_test, stacked_home)
    mae_away = mean_absolute_error(y_away_test, stacked_away)
    exact = np.sum(
        (np.round(stacked_home) == y_home_test)
        & (np.round(stacked_away) == y_away_test)
    )
    exact_pct = exact / len(y_home_test) * 100

    print(
        f"\nSTACKED: MAE home={mae_home:.3f}, away={mae_away:.3f}, Exact={exact_pct:.1f}%"
    )

    # Result accuracy
    pred_result = np.sign(stacked_home - stacked_away)
    actual_result = np.sign(y_home_test - y_away_test)
    result_acc = np.mean(pred_result == actual_result) * 100
    print(f"Result accuracy: {result_acc:.1f}%")

    return models, feature_cols


def save_models(models, feature_cols):
    """Save all trained models."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths = {}
    for name, model in models.items():
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        saved_paths[name] = path

    # Save feature cols
    feature_path = MODELS_DIR / "feature_cols.joblib"
    joblib.dump(feature_cols, feature_path)
    saved_paths["feature_cols"] = feature_path

    # Save pointer to latest
    joblib.dump(saved_paths, MODELS_DIR / "latest_models.joblib")

    print(f"\nSaved {len(models)} models to {MODELS_DIR}")


def main():
    """Main training pipeline."""
    # Load data
    matches_path = PROCESSED_DATA_DIR / "all_matches.csv"
    if not matches_path.exists():
        print(f"Error: {matches_path} not found. Run preprocessing first.")
        return

    df = pd.read_csv(matches_path, parse_dates=["Date"])
    print(f"Loaded {len(df)} matches")

    # Build features
    feature_df, _ = build_dataset(df)

    # Filter valid rows (need some history)
    valid_mask = feature_df.notna().all(axis=1)
    feature_df = feature_df[valid_mask].reset_index(drop=True)
    print(f"Valid matches for training: {len(feature_df)}")

    # Save featured data
    feature_df.to_csv(PROCESSED_DATA_DIR / "featured_matches.csv", index=False)

    # Train models
    models, feature_cols = train_models(feature_df, n_tune_trials=30)

    # Save
    save_models(models, feature_cols)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
