#!/usr/bin/env python3
"""
Main script to run the football score prediction pipeline.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_download():
    """Download all data."""
    from download_data import main as download_main

    download_main()


def run_preprocess():
    """Preprocess downloaded data."""
    from preprocess import main as preprocess_main

    preprocess_main()


def run_train():
    """Train advanced ensemble models with Optuna optimization."""
    from train_advanced import main as train_main

    train_main()


def run_predict(fixtures=None):
    """Run predictions."""
    from predict import FootballScorePredictor, parse_fixture

    predictor = FootballScorePredictor()
    predictor.load_models()

    if fixtures:
        print("\n" + "=" * 60)
        print("Football Score Predictions")
        print("=" * 60 + "\n")

        for fixture in fixtures:
            try:
                home, away = parse_fixture(fixture)
                result = predictor.predict(home, away)
                print(predictor.format_prediction(result))
            except ValueError as e:
                print(f"Error: {e}")
    else:
        # Default Champions League style fixtures
        default_fixtures = [
            ("Chelsea", "Barcelona"),
            ("Real Madrid", "Bayern München"),
            ("Liverpool", "Inter Milan"),
            ("Manchester City", "Paris Saint-Germain"),
            ("Arsenal", "Juventus"),
            ("Borussia Dortmund", "Atletico Madrid"),
        ]

        print("\n" + "=" * 60)
        print("Football Score Predictions (Example Champions League Fixtures)")
        print("=" * 60 + "\n")

        for home, away in default_fixtures:
            result = predictor.predict(home, away)
            print(predictor.format_prediction(result))

        # Show detailed prediction for first match
        result = predictor.predict(default_fixtures[0][0], default_fixtures[0][1])
        print(predictor.format_detailed_prediction(result))


def run_full_pipeline():
    """Run the complete pipeline: download -> preprocess -> train."""
    print("=" * 60)
    print("Running Full Football Prediction Pipeline")
    print("=" * 60)

    print("\n[1/3] Downloading data...")
    run_download()

    print("\n[2/3] Preprocessing data...")
    run_preprocess()

    print("\n[3/3] Training advanced ensemble models...")
    run_train()

    print("\n" + "=" * 60)
    print("Pipeline complete! You can now run predictions.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Football Score Prediction System")
    parser.add_argument(
        "command",
        choices=["download", "preprocess", "train", "predict", "all"],
        help="Command to run",
    )
    parser.add_argument(
        "--fixtures", nargs="*", help='Fixtures to predict (format: "Home vs Away")'
    )

    args = parser.parse_args()

    if args.command == "download":
        run_download()
    elif args.command == "preprocess":
        run_preprocess()
    elif args.command == "train":
        run_train()
    elif args.command == "predict":
        run_predict(args.fixtures)
    elif args.command == "all":
        run_full_pipeline()


if __name__ == "__main__":
    main()
