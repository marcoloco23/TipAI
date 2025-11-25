"""
Get predictions for Champions League matches.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from predict import FootballScorePredictor

# Champions League matches
matches = [
    ("Borussia Dortmund", "Villarreal"),
    ("Chelsea", "Barcelona"),
    ("Eintracht Frankfurt", "Atalanta BC"),
    ("Arsenal", "Bayern München"),
    ("Manchester City", "Bayer Leverkusen"),
    ("Olympique Marseille", "Newcastle United"),
    # ("Bodo/Glimt", "Juventus"),  # Not in database - Norwegian league
]


def main():
    print("Loading models...")
    predictor = FootballScorePredictor()
    predictor.load_models()

    print("\n" + "=" * 60)
    print("CHAMPIONS LEAGUE PREDICTIONS")
    print("=" * 60 + "\n")

    for home, away in matches:
        result = predictor.predict(home, away)

        print(f"{home} vs {away}")
        print(f"  Prediction: {result['predicted_score']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(
            f"  Home Win: {result['home_win_prob']:.0%}  Draw: {result['draw_prob']:.0%}  Away Win: {result['away_win_prob']:.0%}"
        )
        print()

    print(
        "\nNote: Bodo/Glimt vs Juventus not available (Norwegian league not in database)"
    )


if __name__ == "__main__":
    main()
