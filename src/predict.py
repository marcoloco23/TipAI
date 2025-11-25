"""
Football score prediction using advanced stacked ensemble.
Uses XGBoost + LightGBM + CatBoost with meta-learner.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from scipy.stats import poisson
from typing import List, Tuple, Dict
from collections import defaultdict
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

INITIAL_ELO = 1500


def dixon_coles_adjustment(home_goals: int, away_goals: int,
                           lambda_home: float, lambda_away: float,
                           rho: float = -0.20) -> float:
    """Dixon-Coles adjustment for correlated low-scoring outcomes."""
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
    """Predict exact football scores using advanced stacked ensemble."""

    def __init__(self):
        self.models = {}
        self.feature_cols = None
        self.elo_ratings = {}
        self.team_matches = defaultdict(list)
        self.h2h_matches = defaultdict(list)
        self.global_home_avg = 1.55
        self.global_away_avg = 1.31
        self.rho = -0.20

    def load_models(self, verbose: bool = True):
        """Load trained models and supporting data."""
        if verbose:
            print("Loading models...")

        latest_path = MODELS_DIR / "latest_models.joblib"
        if not latest_path.exists():
            raise FileNotFoundError("No trained models found. Run: python src/train_advanced.py")

        saved_paths = joblib.load(latest_path)

        for name, path in saved_paths.items():
            if name == 'feature_cols':
                self.feature_cols = joblib.load(path)
            else:
                self.models[name] = joblib.load(path)

        # Load Elo ratings
        elo_path = PROCESSED_DATA_DIR / "elo_ratings.csv"
        if elo_path.exists():
            elo_df = pd.read_csv(elo_path)
            self.elo_ratings = dict(zip(elo_df['team'], elo_df['elo_rating']))

        # Build team history
        self._build_team_history(verbose)

        if verbose:
            print(f"Loaded {len(self.models)} models, {len(self.elo_ratings)} teams")

    def _build_team_history(self, verbose: bool = True):
        """Build team match history for feature calculation."""
        matches_path = PROCESSED_DATA_DIR / "all_matches.csv"
        if not matches_path.exists():
            return

        matches = pd.read_csv(matches_path, parse_dates=['Date']).sort_values('Date')
        iterator = tqdm(matches.iterrows(), total=len(matches),
                       desc="Building team stats", disable=not verbose)

        for _, row in iterator:
            home, away = row['HomeTeam'], row['AwayTeam']
            hg, ag, date = row['HomeGoals'], row['AwayGoals'], row['Date']

            if hg > ag:
                home_res, away_res, home_pts, away_pts = 'W', 'L', 3, 0
            elif hg < ag:
                home_res, away_res, home_pts, away_pts = 'L', 'W', 0, 3
            else:
                home_res, away_res, home_pts, away_pts = 'D', 'D', 1, 1

            self.team_matches[home].append({
                'date': date, 'opponent': away, 'is_home': True,
                'goals_for': hg, 'goals_against': ag,
                'result': home_res, 'points': home_pts
            })
            self.team_matches[away].append({
                'date': date, 'opponent': home, 'is_home': False,
                'goals_for': ag, 'goals_against': hg,
                'result': away_res, 'points': away_pts
            })

            h2h_key = tuple(sorted([home, away]))
            self.h2h_matches[h2h_key].extend([
                {'date': date, 'team': home, 'goals_for': hg, 'goals_against': ag, 'result': home_res},
                {'date': date, 'team': away, 'goals_for': ag, 'goals_against': hg, 'result': away_res}
            ])

        recent = matches.tail(2000)
        self.global_home_avg = recent['HomeGoals'].mean()
        self.global_away_avg = recent['AwayGoals'].mean()

    def get_team_elo(self, team: str) -> float:
        """Get Elo rating for a team with fuzzy matching."""
        if team in self.elo_ratings:
            return self.elo_ratings[team]
        team_lower = team.lower()
        for known_team, rating in self.elo_ratings.items():
            if team_lower in known_team.lower() or known_team.lower() in team_lower:
                return rating
        return INITIAL_ELO

    def _ewm_mean(self, values: List, span: int = 5) -> float:
        """Exponential weighted mean - recent values weighted more."""
        if not values:
            return 1.4
        n = min(len(values), span * 2)
        weights = np.exp(np.linspace(-1, 0, n))
        return np.average(values[-n:], weights=weights)

    def _get_streak(self, results: List[str]) -> Tuple[int, int]:
        """Calculate current winning/losing streak."""
        if not results:
            return 0, 0
        streak, last = 0, results[-1]
        for r in reversed(results):
            if r == last:
                streak += 1
            else:
                break
        return (streak, 0) if last == 'W' else (0, streak) if last == 'L' else (0, 0)

    def _get_team_features(self, team: str, is_home: bool, opponent: str) -> Dict:
        """Get comprehensive features for a team."""
        matches = self.team_matches.get(team, [])
        venue_matches = [m for m in matches if m['is_home'] == is_home]
        features = {}

        for window in [3, 5, 10]:
            recent = matches[-window:] if matches else []
            venue_recent = venue_matches[-window:] if venue_matches else []

            if recent:
                gf = [m['goals_for'] for m in recent]
                ga = [m['goals_against'] for m in recent]
                pts = [m['points'] for m in recent]
                features[f'goals_for_avg{window}'] = np.mean(gf)
                features[f'goals_against_avg{window}'] = np.mean(ga)
                features[f'goal_diff_avg{window}'] = np.mean(gf) - np.mean(ga)
                features[f'clean_sheets{window}'] = sum(1 for g in ga if g == 0)
                features[f'failed_to_score{window}'] = sum(1 for g in gf if g == 0)
                features[f'points{window}'] = sum(pts)
                features[f'ppg{window}'] = np.mean(pts)
            else:
                features[f'goals_for_avg{window}'] = 1.4
                features[f'goals_against_avg{window}'] = 1.2
                features[f'goal_diff_avg{window}'] = 0.2
                features[f'clean_sheets{window}'] = 1
                features[f'failed_to_score{window}'] = 1
                features[f'points{window}'] = window
                features[f'ppg{window}'] = 1.0

            if venue_recent:
                features[f'venue_gf_avg{window}'] = np.mean([m['goals_for'] for m in venue_recent])
                features[f'venue_ga_avg{window}'] = np.mean([m['goals_against'] for m in venue_recent])
            else:
                features[f'venue_gf_avg{window}'] = 1.5 if is_home else 1.2
                features[f'venue_ga_avg{window}'] = 1.0 if is_home else 1.5

        # EWM and streaks
        if matches:
            gf = [m['goals_for'] for m in matches]
            ga = [m['goals_against'] for m in matches]
            results = [m['result'] for m in matches]
            features['gf_ewm'] = self._ewm_mean(gf)
            features['ga_ewm'] = self._ewm_mean(ga)
            features['win_streak'], features['lose_streak'] = self._get_streak(results)
            features['momentum'] = sum(1 if r == 'W' else -1 if r == 'L' else 0 for r in results[-5:])
        else:
            features['gf_ewm'] = 1.4
            features['ga_ewm'] = 1.2
            features['win_streak'] = features['lose_streak'] = features['momentum'] = 0

        features['days_rest'] = 7
        features['attack_rating'] = features['gf_ewm'] / 1.4 if len(matches) >= 5 else 1.0
        features['defense_rating'] = 1.2 / max(features['ga_ewm'], 0.5) if len(matches) >= 5 else 1.0

        # Head-to-head
        h2h_key = tuple(sorted([team, opponent]))
        team_h2h = [m for m in self.h2h_matches.get(h2h_key, []) if m['team'] == team]
        if len(team_h2h) >= 2:
            features['h2h_gf_avg'] = np.mean([m['goals_for'] for m in team_h2h[-5:]])
            features['h2h_ga_avg'] = np.mean([m['goals_against'] for m in team_h2h[-5:]])
            features['h2h_wins'] = sum(1 for m in team_h2h[-5:] if m['result'] == 'W')
        else:
            features['h2h_gf_avg'] = features['h2h_ga_avg'] = 1.3
            features['h2h_wins'] = 1

        return features

    def build_features(self, home_team: str, away_team: str) -> pd.DataFrame:
        """Build feature DataFrame for a match (with column names to avoid warnings)."""
        features = {
            'home_elo': self.get_team_elo(home_team),
            'away_elo': self.get_team_elo(away_team),
            'elo_diff': self.get_team_elo(home_team) - self.get_team_elo(away_team),
        }

        home_feats = self._get_team_features(home_team, is_home=True, opponent=away_team)
        away_feats = self._get_team_features(away_team, is_home=False, opponent=home_team)

        for k, v in home_feats.items():
            features[f'home_{k}'] = v
        for k, v in away_feats.items():
            features[f'away_{k}'] = v

        features['attack_vs_defense_home'] = home_feats['attack_rating'] * away_feats['defense_rating']
        features['attack_vs_defense_away'] = away_feats['attack_rating'] * home_feats['defense_rating']
        features['momentum_diff'] = home_feats['momentum'] - away_feats['momentum']
        features['form_diff'] = home_feats.get('ppg5', 1) - away_feats.get('ppg5', 1)

        # Return DataFrame with proper column order
        if self.feature_cols:
            return pd.DataFrame([[features.get(col, 0) for col in self.feature_cols]],
                              columns=self.feature_cols)
        return pd.DataFrame([features])

    def predict_lambda(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """Predict expected goals using stacked ensemble."""
        X = self.build_features(home_team, away_team)

        home_preds, away_preds = [], []
        for model_type in ['xgb', 'lgb', 'cat']:
            if f'{model_type}_home' in self.models:
                home_preds.append(self.models[f'{model_type}_home'].predict(X)[0])
            if f'{model_type}_away' in self.models:
                away_preds.append(self.models[f'{model_type}_away'].predict(X)[0])

        if 'meta_home' in self.models and home_preds:
            meta_home = np.array(home_preds).reshape(1, -1)
            meta_away = np.array(away_preds).reshape(1, -1)
            lambda_home = self.models['meta_home'].predict(meta_home)[0]
            lambda_away = self.models['meta_away'].predict(meta_away)[0]
        elif home_preds:
            lambda_home = np.mean(home_preds)
            lambda_away = np.mean(away_preds)
        else:
            lambda_home = self.global_home_avg
            lambda_away = self.global_away_avg

        # Scale for variance
        scale = 1.8
        lambda_home = self.global_home_avg + (lambda_home - self.global_home_avg) * scale
        lambda_away = self.global_away_avg + (lambda_away - self.global_away_avg) * scale

        return max(0.5, lambda_home), max(0.4, lambda_away)

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

        flat_probs = [(h, a, prob_matrix[h, a]) for h in range(8) for a in range(8)]
        flat_probs.sort(key=lambda x: x[2], reverse=True)

        home_win_prob = np.sum(np.tril(prob_matrix, -1))
        draw_prob = np.sum(np.diag(prob_matrix))
        away_win_prob = np.sum(np.triu(prob_matrix, 1))

        # Select score based on most likely outcome
        if home_win_prob >= draw_prob and home_win_prob >= away_win_prob:
            scores = [(h, a, p) for h, a, p in flat_probs if h > a]
        elif away_win_prob >= draw_prob:
            scores = [(h, a, p) for h, a, p in flat_probs if a > h]
        else:
            scores = [(h, a, p) for h, a, p in flat_probs if h == a]

        predicted_home, predicted_away, confidence = scores[0] if scores else flat_probs[0]

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
        """Predict multiple matches."""
        iterator = tqdm(fixtures, desc="Predicting", disable=not verbose)
        return [self.predict(home, away) for home, away in iterator]

    def format_prediction(self, result: Dict) -> str:
        """Format prediction for display."""
        return (f"{result['home_team']} vs {result['away_team']}: "
                f"{result['predicted_score']} ({result['confidence']:.1%})")


if __name__ == "__main__":
    predictor = FootballScorePredictor()
    predictor.load_models()

    fixtures = [
        ("Borussia Dortmund", "Villarreal"),
        ("Chelsea", "Barcelona"),
        ("Arsenal", "Bayern München"),
        ("Manchester City", "Bayer Leverkusen"),
    ]

    print("\n" + "="*60)
    print("PREDICTIONS")
    print("="*60 + "\n")

    for result in predictor.predict_batch(fixtures):
        print(f"{result['home_team']} vs {result['away_team']}: {result['predicted_score']} ({result['confidence']:.1%})")
        print(f"  λ={result['lambda_home']:.2f}/{result['lambda_away']:.2f}  "
              f"H:{result['home_win_prob']:.0%} D:{result['draw_prob']:.0%} A:{result['away_win_prob']:.0%}\n")
