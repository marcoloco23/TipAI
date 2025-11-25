"""
Football score prediction pipeline.
Uses ensemble of XGBoost and LightGBM with Dixon-Coles adjusted Poisson scoring.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from scipy.stats import poisson
from typing import List, Tuple, Dict
from tqdm import tqdm

# Base directories
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def dixon_coles_adjustment(home_goals: int, away_goals: int,
                           lambda_home: float, lambda_away: float,
                           rho: float = -0.20) -> float:
    """
    Dixon-Coles adjustment for correlated low-scoring outcomes.
    Reduces the probability of 0-0, 1-0, 0-1, 1-1 draws.
    """
    if home_goals == 0 and away_goals == 0:
        return 1 - lambda_home * lambda_away * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + lambda_home * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + lambda_away * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


class FootballScorePredictor:
    """Predict exact football scores using ensemble model with Dixon-Coles."""

    def __init__(self):
        self.models = None
        self.feature_cols = None
        self.elo_ratings = None
        self.team_stats = None
        self.global_home_avg = 1.55
        self.global_away_avg = 1.31

        # Calibration parameters (tuned for prediction variety)
        self.lambda_scale = 2.5
        self.rho = -0.20

    def load_models(self, verbose: bool = True):
        """Load trained models and supporting data."""
        if verbose:
            print("Loading models...")

        latest_path = MODELS_DIR / "latest_models.joblib"
        if not latest_path.exists():
            raise FileNotFoundError("No trained models found. Run: python run.py train")

        saved_paths = joblib.load(latest_path)

        self.models = {}
        for name, path in saved_paths.items():
            if name != 'feature_cols':
                self.models[name] = joblib.load(path)
            else:
                self.feature_cols = joblib.load(path)

        # Load Elo ratings
        elo_path = PROCESSED_DATA_DIR / "elo_ratings.csv"
        if elo_path.exists():
            elo_df = pd.read_csv(elo_path)
            self.elo_ratings = dict(zip(elo_df['team'], elo_df['elo_rating']))
        else:
            self.elo_ratings = {}

        # Load team stats
        featured_path = PROCESSED_DATA_DIR / "featured_matches.csv"
        if featured_path.exists():
            df = pd.read_csv(featured_path, parse_dates=['Date'])
            self._build_team_stats(df, verbose)
            recent = df.tail(2000)
            self.global_home_avg = recent['HomeGoals'].mean()
            self.global_away_avg = recent['AwayGoals'].mean()

        if verbose:
            print(f"Loaded {len(self.models)} models, {len(self.elo_ratings)} teams")

    def _build_team_stats(self, df: pd.DataFrame, verbose: bool = True):
        """Build latest team statistics from historical data."""
        self.team_stats = {}
        all_teams = list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))

        iterator = tqdm(all_teams, desc="Building team stats", disable=not verbose)
        for team in iterator:
            home_matches = df[df['HomeTeam'] == team].sort_values('Date')
            away_matches = df[df['AwayTeam'] == team].sort_values('Date')

            # Home stats
            if len(home_matches) > 0:
                recent_home = home_matches.tail(10)
                home_stats = {
                    'goals_scored_home_avg5': recent_home.tail(5)['HomeGoals'].mean() if len(recent_home) >= 5 else recent_home['HomeGoals'].mean(),
                    'goals_conceded_home_avg5': recent_home.tail(5)['AwayGoals'].mean() if len(recent_home) >= 5 else recent_home['AwayGoals'].mean(),
                    'goals_scored_home_avg10': recent_home['HomeGoals'].mean(),
                    'goals_conceded_home_avg10': recent_home['AwayGoals'].mean(),
                    'home_wins_last5': (recent_home.tail(5)['HomeGoals'] > recent_home.tail(5)['AwayGoals']).sum() if len(recent_home) >= 5 else 2,
                }
            else:
                home_stats = {
                    'goals_scored_home_avg5': 1.5, 'goals_conceded_home_avg5': 1.0,
                    'goals_scored_home_avg10': 1.5, 'goals_conceded_home_avg10': 1.0,
                    'home_wins_last5': 2,
                }

            # Away stats
            if len(away_matches) > 0:
                recent_away = away_matches.tail(10)
                away_stats = {
                    'goals_scored_away_avg5': recent_away.tail(5)['AwayGoals'].mean() if len(recent_away) >= 5 else recent_away['AwayGoals'].mean(),
                    'goals_conceded_away_avg5': recent_away.tail(5)['HomeGoals'].mean() if len(recent_away) >= 5 else recent_away['HomeGoals'].mean(),
                    'goals_scored_away_avg10': recent_away['AwayGoals'].mean(),
                    'goals_conceded_away_avg10': recent_away['HomeGoals'].mean(),
                    'away_wins_last5': (recent_away.tail(5)['AwayGoals'] > recent_away.tail(5)['HomeGoals']).sum() if len(recent_away) >= 5 else 1,
                }
            else:
                away_stats = {
                    'goals_scored_away_avg5': 1.2, 'goals_conceded_away_avg5': 1.5,
                    'goals_scored_away_avg10': 1.2, 'goals_conceded_away_avg10': 1.5,
                    'away_wins_last5': 1,
                }

            # Overall form
            all_home = df[df['HomeTeam'] == team].copy()
            all_away = df[df['AwayTeam'] == team].copy()
            all_home['goals_for'] = all_home['HomeGoals']
            all_home['goals_against'] = all_home['AwayGoals']
            all_away['goals_for'] = all_away['AwayGoals']
            all_away['goals_against'] = all_away['HomeGoals']
            all_matches = pd.concat([all_home[['Date', 'goals_for', 'goals_against']],
                                    all_away[['Date', 'goals_for', 'goals_against']]]).sort_values('Date')

            if len(all_matches) > 0:
                recent = all_matches.tail(5)
                form_stats = {
                    'total_goals_avg5': recent['goals_for'].mean(),
                    'total_conceded_avg5': recent['goals_against'].mean(),
                    'form_points5': ((recent['goals_for'] > recent['goals_against']).sum() * 3 +
                                    (recent['goals_for'] == recent['goals_against']).sum()),
                }
            else:
                form_stats = {'total_goals_avg5': 1.4, 'total_conceded_avg5': 1.2, 'form_points5': 7}

            self.team_stats[team] = {**home_stats, **away_stats, **form_stats}

    def get_team_elo(self, team: str) -> float:
        """Get Elo rating for a team with fuzzy matching."""
        if team in self.elo_ratings:
            return self.elo_ratings[team]
        team_lower = team.lower()
        for known_team, rating in self.elo_ratings.items():
            if team_lower in known_team.lower() or known_team.lower() in team_lower:
                return rating
        return 1500.0

    def get_team_stats(self, team: str) -> Dict:
        """Get statistics for a team with fuzzy matching."""
        if team in self.team_stats:
            return self.team_stats[team]
        team_lower = team.lower()
        for known_team, stats in self.team_stats.items():
            if team_lower in known_team.lower() or known_team.lower() in team_lower:
                return stats
        return {
            'goals_scored_home_avg5': 1.5, 'goals_conceded_home_avg5': 1.0,
            'goals_scored_home_avg10': 1.5, 'goals_conceded_home_avg10': 1.0,
            'home_wins_last5': 2, 'goals_scored_away_avg5': 1.2,
            'goals_conceded_away_avg5': 1.5, 'goals_scored_away_avg10': 1.2,
            'goals_conceded_away_avg10': 1.5, 'away_wins_last5': 1,
            'total_goals_avg5': 1.4, 'total_conceded_avg5': 1.2, 'form_points5': 7,
        }

    def build_features(self, home_team: str, away_team: str) -> np.ndarray:
        """Build feature vector for a match."""
        home_elo = self.get_team_elo(home_team)
        away_elo = self.get_team_elo(away_team)
        home_stats = self.get_team_stats(home_team)
        away_stats = self.get_team_stats(away_team)

        features = [
            home_elo, away_elo, home_elo - away_elo,
            home_stats['goals_scored_home_avg5'], home_stats['goals_conceded_home_avg5'],
            home_stats['goals_scored_home_avg10'], home_stats['goals_conceded_home_avg10'],
            home_stats['home_wins_last5'],
            away_stats['goals_scored_away_avg5'], away_stats['goals_conceded_away_avg5'],
            away_stats['goals_scored_away_avg10'], away_stats['goals_conceded_away_avg10'],
            away_stats['away_wins_last5'],
            home_stats['total_goals_avg5'], home_stats['total_conceded_avg5'], home_stats['form_points5'],
            away_stats['total_goals_avg5'], away_stats['total_conceded_avg5'], away_stats['form_points5'],
            1.3, 1.3, 0,  # h2h defaults
        ]
        return np.array(features).reshape(1, -1)

    def predict_lambda(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """Predict expected goals using team stats + model + Elo."""
        X = self.build_features(home_team, away_team)
        home_stats = self.get_team_stats(home_team)
        away_stats = self.get_team_stats(away_team)

        # Model predictions
        home_preds, away_preds = [], []
        if 'xgb_home' in self.models and self.models['xgb_home'] is not None:
            home_preds.append(self.models['xgb_home'].predict(X)[0])
            away_preds.append(self.models['xgb_away'].predict(X)[0])
        if 'lgb_home' in self.models and self.models['lgb_home'] is not None:
            home_preds.append(self.models['lgb_home'].predict(X)[0])
            away_preds.append(self.models['lgb_away'].predict(X)[0])

        if home_preds:
            model_home = np.mean(home_preds)
            model_away = np.mean(away_preds)
        else:
            model_home, model_away = 1.5, 1.2

        # Team-based lambda using attack vs defense
        home_attack = home_stats['goals_scored_home_avg5']
        away_defense = away_stats['goals_conceded_away_avg5']
        team_home = (home_attack + away_defense) / 2

        away_attack = away_stats['goals_scored_away_avg5']
        home_defense = home_stats['goals_conceded_home_avg5']
        team_away = (away_attack + home_defense) / 2

        # Elo-based adjustment: strong away teams should concede less
        home_elo = self.get_team_elo(home_team)
        away_elo = self.get_team_elo(away_team)
        elo_diff = home_elo - away_elo

        # Elo modifier: if away team is strong (negative elo_diff), reduce home lambda
        # Range: -300 to +300 elo diff -> -0.3 to +0.3 lambda adjustment
        elo_modifier = elo_diff / 1000

        # Blend: 50% model, 30% team stats, 20% Elo adjustment
        blended_home = 0.5 * model_home + 0.3 * team_home + 0.2 * (self.global_home_avg + elo_modifier)
        blended_away = 0.5 * model_away + 0.3 * team_away + 0.2 * (self.global_away_avg - elo_modifier)

        # Scale with reduced factor (2.0 instead of 2.5 for less extreme predictions)
        scale = 2.0
        scaled_home = self.global_home_avg + (blended_home - self.global_home_avg) * scale
        scaled_away = self.global_away_avg + (blended_away - self.global_away_avg) * scale

        return max(0.5, scaled_home), max(0.4, scaled_away)

    def predict_score_probabilities(self, home_team: str, away_team: str,
                                   max_goals: int = 8) -> Tuple[np.ndarray, float, float]:
        """Calculate probability matrix with Dixon-Coles adjustment."""
        lambda_home, lambda_away = self.predict_lambda(home_team, away_team)

        prob_matrix = np.zeros((max_goals, max_goals))
        for h in range(max_goals):
            for a in range(max_goals):
                base_prob = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
                dc_adj = dixon_coles_adjustment(h, a, lambda_home, lambda_away, self.rho)
                prob_matrix[h, a] = base_prob * dc_adj

        prob_matrix /= prob_matrix.sum()
        return prob_matrix, lambda_home, lambda_away

    def predict(self, home_team: str, away_team: str) -> Dict:
        """Predict match score with confidence."""
        prob_matrix, lambda_home, lambda_away = self.predict_score_probabilities(home_team, away_team)

        # Get top scores
        flat_probs = [(h, a, prob_matrix[h, a]) for h in range(8) for a in range(8)]
        flat_probs.sort(key=lambda x: x[2], reverse=True)

        # Calculate outcome probabilities
        home_win_prob = np.sum(np.tril(prob_matrix, -1))
        draw_prob = np.sum(np.diag(prob_matrix))
        away_win_prob = np.sum(np.triu(prob_matrix, 1))

        # Smart score selection: pick most likely outcome, then best score within it
        # This ensures we predict wins when a team is favored, not always 1-1
        if home_win_prob >= draw_prob and home_win_prob >= away_win_prob:
            # Home win is most likely - pick best home win score
            home_wins = [(h, a, p) for h, a, p in flat_probs if h > a]
            if home_wins:
                predicted_home, predicted_away, confidence = home_wins[0]
            else:
                predicted_home, predicted_away, confidence = flat_probs[0]
        elif away_win_prob >= draw_prob and away_win_prob >= home_win_prob:
            # Away win is most likely - pick best away win score
            away_wins = [(h, a, p) for h, a, p in flat_probs if a > h]
            if away_wins:
                predicted_home, predicted_away, confidence = away_wins[0]
            else:
                predicted_home, predicted_away, confidence = flat_probs[0]
        else:
            # Draw is most likely - pick best draw score
            draws = [(h, a, p) for h, a, p in flat_probs if h == a]
            if draws:
                predicted_home, predicted_away, confidence = draws[0]
            else:
                predicted_home, predicted_away, confidence = flat_probs[0]

        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_score': f"{predicted_home}-{predicted_away}",
            'home_goals': predicted_home,
            'away_goals': predicted_away,
            'confidence': confidence,
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'home_win_prob': home_win_prob,
            'draw_prob': draw_prob,
            'away_win_prob': away_win_prob,
            'prob_matrix': prob_matrix,
        }

    def predict_batch(self, fixtures: List[Tuple[str, str]], verbose: bool = True) -> List[Dict]:
        """Predict multiple matches with progress bar."""
        results = []
        iterator = tqdm(fixtures, desc="Predicting", disable=not verbose)
        for home_team, away_team in iterator:
            results.append(self.predict(home_team, away_team))
        return results

    def format_prediction(self, result: Dict) -> str:
        """Format prediction result for display."""
        return (f"{result['home_team']} vs {result['away_team']}: "
                f"{result['predicted_score']} ({result['confidence']:.1%})")


def main():
    """Test predictor."""
    predictor = FootballScorePredictor()
    predictor.load_models()

    fixtures = [
        ("Chelsea", "Barcelona"),
        ("Real Madrid", "Bayern München"),
        ("Liverpool", "Inter Milan"),
        ("Manchester City", "Paris Saint-Germain"),
        ("Arsenal", "Juventus"),
        ("Bayern München", "Bologna"),
        ("Bayer Leverkusen", "Celtic"),
    ]

    print("\n" + "="*60)
    print("PREDICTIONS")
    print("="*60 + "\n")

    results = predictor.predict_batch(fixtures)
    for result in results:
        print(f"{result['home_team']} vs {result['away_team']}: {result['predicted_score']} ({result['confidence']:.1%})")
        print(f"  λ={result['lambda_home']:.2f}/{result['lambda_away']:.2f}  "
              f"H:{result['home_win_prob']:.0%} D:{result['draw_prob']:.0%} A:{result['away_win_prob']:.0%}\n")


if __name__ == "__main__":
    main()
