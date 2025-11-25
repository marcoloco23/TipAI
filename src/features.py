"""
Feature engineering module for football score prediction.
Calculates rolling statistics, Elo ratings, and head-to-head features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Base directories
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Elo configuration
ELO_K = 20  # K-factor for Elo updates
ELO_HOME_ADVANTAGE = 100  # Home team Elo bonus
INITIAL_ELO = 1500


class EloRatingSystem:
    """Track Elo ratings for all teams."""

    def __init__(self, k=ELO_K, home_advantage=ELO_HOME_ADVANTAGE, initial=INITIAL_ELO):
        self.k = k
        self.home_advantage = home_advantage
        self.initial = initial
        self.ratings = defaultdict(lambda: initial)

    def expected_score(self, rating_a, rating_b):
        """Calculate expected score for team A against team B."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, home_team, away_team, home_goals, away_goals):
        """Update Elo ratings after a match."""
        home_rating = self.ratings[home_team] + self.home_advantage
        away_rating = self.ratings[away_team]

        # Actual result
        if home_goals > away_goals:
            actual_home = 1
        elif home_goals < away_goals:
            actual_home = 0
        else:
            actual_home = 0.5

        # Expected results
        expected_home = self.expected_score(home_rating, away_rating)

        # Goal difference multiplier
        goal_diff = abs(home_goals - away_goals)
        if goal_diff <= 1:
            multiplier = 1
        elif goal_diff == 2:
            multiplier = 1.5
        else:
            multiplier = (11 + goal_diff) / 8

        # Update ratings
        delta = self.k * multiplier * (actual_home - expected_home)
        self.ratings[home_team] += delta
        self.ratings[away_team] -= delta

        return self.ratings[home_team], self.ratings[away_team]

    def get_rating(self, team):
        """Get current Elo rating for a team."""
        return self.ratings[team]


def calculate_rolling_stats(
    df, team_col, goals_for_col, goals_against_col, window=5, prefix=""
):
    """Calculate rolling statistics for a team."""
    stats = {}

    # Goals
    stats[f"{prefix}goals_scored_avg_{window}"] = (
        df[goals_for_col].rolling(window, min_periods=1).mean()
    )
    stats[f"{prefix}goals_conceded_avg_{window}"] = (
        df[goals_against_col].rolling(window, min_periods=1).mean()
    )

    # Clean sheets
    stats[f"{prefix}clean_sheets_{window}"] = (
        (df[goals_against_col] == 0).rolling(window, min_periods=1).sum()
    )

    # Form (points)
    def get_points(row):
        if row[goals_for_col] > row[goals_against_col]:
            return 3
        elif row[goals_for_col] == row[goals_against_col]:
            return 1
        return 0

    points = df.apply(get_points, axis=1)
    stats[f"{prefix}form_points_{window}"] = points.rolling(window, min_periods=1).sum()

    return pd.DataFrame(stats)


def build_team_stats(matches_df):
    """Build historical statistics for each team."""
    print("Building team statistics...")

    # Sort by date
    matches_df = matches_df.sort_values("Date").reset_index(drop=True)

    # Initialize tracking dictionaries
    team_home_matches = defaultdict(list)
    team_away_matches = defaultdict(list)
    team_all_matches = defaultdict(list)
    elo_system = EloRatingSystem()

    # Features to calculate per match
    feature_rows = []

    for idx, row in tqdm(
        matches_df.iterrows(), total=len(matches_df), desc="Building features"
    ):

        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        # Get current Elo ratings (before this match)
        home_elo = elo_system.get_rating(home_team)
        away_elo = elo_system.get_rating(away_team)

        # Calculate features from historical matches
        features = {
            "match_idx": idx,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
        }

        # Home team stats (from their home matches)
        home_history = team_home_matches[home_team]
        if len(home_history) >= 3:
            recent = pd.DataFrame(home_history[-10:])
            features["home_goals_scored_home_avg5"] = recent["HomeGoals"].tail(5).mean()
            features["home_goals_conceded_home_avg5"] = (
                recent["AwayGoals"].tail(5).mean()
            )
            features["home_goals_scored_home_avg10"] = recent["HomeGoals"].mean()
            features["home_goals_conceded_home_avg10"] = recent["AwayGoals"].mean()
            features["home_home_wins_last5"] = (
                recent["HomeGoals"].tail(5) > recent["AwayGoals"].tail(5)
            ).sum()
        else:
            features["home_goals_scored_home_avg5"] = 1.5
            features["home_goals_conceded_home_avg5"] = 1.0
            features["home_goals_scored_home_avg10"] = 1.5
            features["home_goals_conceded_home_avg10"] = 1.0
            features["home_home_wins_last5"] = 2

        # Away team stats (from their away matches)
        away_history = team_away_matches[away_team]
        if len(away_history) >= 3:
            recent = pd.DataFrame(away_history[-10:])
            features["away_goals_scored_away_avg5"] = recent["AwayGoals"].tail(5).mean()
            features["away_goals_conceded_away_avg5"] = (
                recent["HomeGoals"].tail(5).mean()
            )
            features["away_goals_scored_away_avg10"] = recent["AwayGoals"].mean()
            features["away_goals_conceded_away_avg10"] = recent["HomeGoals"].mean()
            features["away_away_wins_last5"] = (
                recent["AwayGoals"].tail(5) > recent["HomeGoals"].tail(5)
            ).sum()
        else:
            features["away_goals_scored_away_avg5"] = 1.2
            features["away_goals_conceded_away_avg5"] = 1.5
            features["away_goals_scored_away_avg10"] = 1.2
            features["away_goals_conceded_away_avg10"] = 1.5
            features["away_away_wins_last5"] = 1

        # Overall team form (home + away combined)
        home_all = team_all_matches[home_team]
        away_all = team_all_matches[away_team]

        if len(home_all) >= 3:
            recent = pd.DataFrame(home_all[-5:])
            features["home_total_goals_avg5"] = recent["goals_for"].mean()
            features["home_total_conceded_avg5"] = recent["goals_against"].mean()
            features["home_form_points5"] = recent["points"].sum()
        else:
            features["home_total_goals_avg5"] = 1.4
            features["home_total_conceded_avg5"] = 1.2
            features["home_form_points5"] = 7

        if len(away_all) >= 3:
            recent = pd.DataFrame(away_all[-5:])
            features["away_total_goals_avg5"] = recent["goals_for"].mean()
            features["away_total_conceded_avg5"] = recent["goals_against"].mean()
            features["away_form_points5"] = recent["points"].sum()
        else:
            features["away_total_goals_avg5"] = 1.4
            features["away_total_conceded_avg5"] = 1.2
            features["away_form_points5"] = 7

        # Head-to-head stats
        h2h_matches = [m for m in home_all if m.get("opponent") == away_team]
        h2h_matches += [m for m in away_all if m.get("opponent") == home_team]

        if len(h2h_matches) >= 2:
            h2h_df = pd.DataFrame(h2h_matches[-5:])
            # Perspective of home team in current match
            home_h2h_goals = h2h_df.apply(
                lambda x: (
                    x["goals_for"] if x.get("team") == home_team else x["goals_against"]
                ),
                axis=1,
            )
            away_h2h_goals = h2h_df.apply(
                lambda x: (
                    x["goals_for"] if x.get("team") == away_team else x["goals_against"]
                ),
                axis=1,
            )
            features["h2h_home_goals_avg"] = home_h2h_goals.mean()
            features["h2h_away_goals_avg"] = away_h2h_goals.mean()
            features["h2h_matches"] = len(h2h_matches)
        else:
            features["h2h_home_goals_avg"] = 1.3
            features["h2h_away_goals_avg"] = 1.3
            features["h2h_matches"] = 0

        # Add shots features if available
        if "HomeShots" in row and pd.notna(row.get("HomeShots")):
            features["home_shots"] = row["HomeShots"]
            features["away_shots"] = row.get("AwayShots", 0)
            features["home_shots_on_target"] = row.get("HomeShotsOnTarget", 0)
            features["away_shots_on_target"] = row.get("AwayShotsOnTarget", 0)

        feature_rows.append(features)

        # Update histories after processing
        home_goals = row["HomeGoals"]
        away_goals = row["AwayGoals"]

        # Update Elo
        elo_system.update(home_team, away_team, home_goals, away_goals)

        # Update home team history
        team_home_matches[home_team].append(
            {
                "Date": row["Date"],
                "HomeGoals": home_goals,
                "AwayGoals": away_goals,
            }
        )

        # Update away team history
        team_away_matches[away_team].append(
            {
                "Date": row["Date"],
                "HomeGoals": home_goals,
                "AwayGoals": away_goals,
            }
        )

        # Update overall history
        home_points = (
            3 if home_goals > away_goals else (1 if home_goals == away_goals else 0)
        )
        away_points = (
            3 if away_goals > home_goals else (1 if home_goals == away_goals else 0)
        )

        team_all_matches[home_team].append(
            {
                "Date": row["Date"],
                "team": home_team,
                "opponent": away_team,
                "goals_for": home_goals,
                "goals_against": away_goals,
                "points": home_points,
                "is_home": True,
            }
        )

        team_all_matches[away_team].append(
            {
                "Date": row["Date"],
                "team": away_team,
                "opponent": home_team,
                "goals_for": away_goals,
                "goals_against": home_goals,
                "points": away_points,
                "is_home": False,
            }
        )

    # Create features DataFrame
    features_df = pd.DataFrame(feature_rows)

    # Merge with original matches
    result = matches_df.copy()
    for col in features_df.columns:
        if col != "match_idx":
            result[col] = features_df[col].values

    print(f"Created {len(features_df.columns)} features for {len(result)} matches")

    return result, elo_system


def prepare_training_data(matches_df):
    """Prepare final training dataset."""
    print("\nPreparing training data...")

    # Build features
    featured_df, elo_system = build_team_stats(matches_df)

    # Define feature columns
    feature_cols = [
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

    # Targets
    target_home = "HomeGoals"
    target_away = "AwayGoals"

    # Filter to rows with all features
    valid_mask = featured_df[feature_cols].notna().all(axis=1)
    valid_mask &= featured_df[[target_home, target_away]].notna().all(axis=1)

    filtered_df = featured_df[valid_mask].copy()

    print(f"Valid matches for training: {len(filtered_df)}")

    return filtered_df, feature_cols, elo_system


def save_featured_data(featured_df, elo_system):
    """Save featured dataset and Elo ratings."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save featured matches
    output_path = PROCESSED_DATA_DIR / "featured_matches.csv"
    featured_df.to_csv(output_path, index=False)
    print(f"Saved featured data to {output_path}")

    # Save Elo ratings
    elo_df = pd.DataFrame(
        [
            {"team": team, "elo_rating": rating}
            for team, rating in elo_system.ratings.items()
        ]
    )
    elo_path = PROCESSED_DATA_DIR / "elo_ratings.csv"
    elo_df.to_csv(elo_path, index=False)
    print(f"Saved Elo ratings to {elo_path}")

    return output_path, elo_path


def main():
    """Main feature engineering function."""
    # Load processed matches
    matches_path = PROCESSED_DATA_DIR / "all_matches.csv"

    if not matches_path.exists():
        print(f"Error: {matches_path} not found. Run preprocess.py first.")
        return None

    matches_df = pd.read_csv(matches_path, parse_dates=["Date"])
    print(f"Loaded {len(matches_df)} matches")

    # Prepare training data
    featured_df, feature_cols, elo_system = prepare_training_data(matches_df)

    # Save
    save_featured_data(featured_df, elo_system)

    return featured_df, feature_cols, elo_system


if __name__ == "__main__":
    main()
