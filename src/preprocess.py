"""
Data preprocessing module for football match data.
- Standardize team names across different sources
- Parse dates and handle missing values
- Merge all data into unified format
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re

# Base directories
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Team name standardization mapping
TEAM_NAME_MAPPING = {
    # English teams
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Tottenham": "Tottenham Hotspur",
    "Spurs": "Tottenham Hotspur",
    "Wolves": "Wolverhampton",
    "Sheffield United": "Sheffield Utd",
    "West Ham": "West Ham United",
    "Newcastle": "Newcastle United",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Norwich": "Norwich City",
    "Brighton": "Brighton & Hove Albion",
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham": "Nottingham Forest",
    # Spanish teams
    "Ath Madrid": "Atletico Madrid",
    "Atlético Madrid": "Atletico Madrid",
    "Atlético": "Atletico Madrid",
    "Ath Bilbao": "Athletic Bilbao",
    "Athletic Club": "Athletic Bilbao",
    "Real Sociedad": "Real Sociedad",
    "Betis": "Real Betis",
    "Celta": "Celta Vigo",
    "Espanol": "Espanyol",
    "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia",
    "Villarreal CF": "Villarreal",
    # German teams
    "Bayern Munich": "Bayern München",
    "Bayern": "Bayern München",
    "FC Bayern München": "Bayern München",
    "Dortmund": "Borussia Dortmund",
    "Borussia Dortmund": "Borussia Dortmund",
    "M'gladbach": "Borussia Mönchengladbach",
    "Monchengladbach": "Borussia Mönchengladbach",
    "Leverkusen": "Bayer Leverkusen",
    "Bayer 04 Leverkusen": "Bayer Leverkusen",
    "RB Leipzig": "RB Leipzig",
    "Leipzig": "RB Leipzig",
    "Frankfurt": "Eintracht Frankfurt",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Wolfsburg": "VfL Wolfsburg",
    "Stuttgart": "VfB Stuttgart",
    "Freiburg": "SC Freiburg",
    # Italian teams
    "Inter": "Inter Milan",
    "Internazionale": "Inter Milan",
    "AC Milan": "AC Milan",
    "Milan": "AC Milan",
    "Juventus": "Juventus",
    "Napoli": "SSC Napoli",
    "Roma": "AS Roma",
    "Lazio": "SS Lazio",
    "Atalanta": "Atalanta BC",
    "Fiorentina": "ACF Fiorentina",
    # French teams
    "Paris SG": "Paris Saint-Germain",
    "Paris Saint Germain": "Paris Saint-Germain",
    "PSG": "Paris Saint-Germain",
    "Lyon": "Olympique Lyon",
    "Olympique Lyonnais": "Olympique Lyon",
    "Marseille": "Olympique Marseille",
    "Olympique de Marseille": "Olympique Marseille",
    "Monaco": "AS Monaco",
    "Lille": "LOSC Lille",
    "Rennes": "Stade Rennais",
    "Nice": "OGC Nice",
    "Lens": "RC Lens",
}


def standardize_team_name(name):
    """Standardize team name using mapping."""
    if pd.isna(name):
        return name
    name = str(name).strip()
    return TEAM_NAME_MAPPING.get(name, name)


def parse_date(date_str, date_format=None):
    """Parse date string to datetime object."""
    if pd.isna(date_str):
        return None

    date_str = str(date_str).strip()

    # Try various date formats
    formats = [
        "%d/%m/%Y",
        "%d/%m/%y",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%Y/%m/%d",
        "%d.%m.%Y",
        "%d %b %Y",
    ]

    if date_format:
        formats.insert(0, date_format)

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def load_football_data_co_uk():
    """Load and process football-data.co.uk CSV files."""
    print("Loading football-data.co.uk data...")

    all_matches = []
    league_mapping = {
        "E0": "England Premier League",
        "SP1": "Spain La Liga",
        "D1": "Germany Bundesliga",
        "I1": "Italy Serie A",
        "F1": "France Ligue 1",
    }

    for filepath in RAW_DATA_DIR.glob("*.csv"):
        filename = filepath.name

        # Skip Champions League files
        if "cl_" in filename or "champions" in filename.lower():
            continue

        # Extract league code and season from filename
        parts = filename.replace(".csv", "").split("_")
        if len(parts) != 2:
            continue

        league_code, season = parts

        if league_code not in league_mapping:
            continue

        try:
            df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")

            # Required columns
            required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
            if not all(col in df.columns for col in required):
                # Try alternate column names
                if "HG" in df.columns and "AG" in df.columns:
                    df = df.rename(columns={"HG": "FTHG", "AG": "FTAG"})
                else:
                    print(f"  Skipping {filename}: missing required columns")
                    continue

            # Select and rename columns
            match_df = pd.DataFrame(
                {
                    "Date": df["Date"].apply(parse_date),
                    "HomeTeam": df["HomeTeam"].apply(standardize_team_name),
                    "AwayTeam": df["AwayTeam"].apply(standardize_team_name),
                    "HomeGoals": pd.to_numeric(df["FTHG"], errors="coerce"),
                    "AwayGoals": pd.to_numeric(df["FTAG"], errors="coerce"),
                    "Competition": league_mapping[league_code],
                    "Season": f"20{season[:2]}-20{season[2:]}",
                }
            )

            # Add optional columns if available
            optional_cols = {
                "HS": "HomeShots",
                "AS": "AwayShots",
                "HST": "HomeShotsOnTarget",
                "AST": "AwayShotsOnTarget",
                "HC": "HomeCorners",
                "AC": "AwayCorners",
                "HF": "HomeFouls",
                "AF": "AwayFouls",
                "HY": "HomeYellowCards",
                "AY": "AwayYellowCards",
                "HR": "HomeRedCards",
                "AR": "AwayRedCards",
            }

            for old_col, new_col in optional_cols.items():
                if old_col in df.columns:
                    match_df[new_col] = pd.to_numeric(df[old_col], errors="coerce")

            # Remove rows with missing essential data
            match_df = match_df.dropna(
                subset=["Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"]
            )

            all_matches.append(match_df)
            print(f"  Loaded {filename}: {len(match_df)} matches")

        except Exception as e:
            print(f"  Error loading {filename}: {e}")

    if all_matches:
        combined = pd.concat(all_matches, ignore_index=True)
        print(f"Total league matches: {len(combined)}")
        return combined
    return pd.DataFrame()


def load_champions_league_data():
    """Load and process Champions League data."""
    print("\nLoading Champions League data...")

    all_matches = []

    # Load current season CL data from fixturedownload
    cl_current = RAW_DATA_DIR / "champions_league_2024.csv"
    if cl_current.exists():
        try:
            df = pd.read_csv(cl_current, encoding="utf-8")

            # fixturedownload format: Match Number,Round Number,Date,Location,Home Team,Away Team,Result
            if "Home Team" in df.columns and "Away Team" in df.columns:
                # Parse result (e.g., "2 - 1" or "2-1")
                def parse_result(result):
                    if pd.isna(result) or result == "" or result == "-":
                        return None, None
                    result = str(result).strip()
                    match = re.match(r"(\d+)\s*-\s*(\d+)", result)
                    if match:
                        return int(match.group(1)), int(match.group(2))
                    return None, None

                results = df["Result"].apply(lambda x: pd.Series(parse_result(x)))
                df["HomeGoals"] = results[0]
                df["AwayGoals"] = results[1]

                match_df = pd.DataFrame(
                    {
                        "Date": df["Date"].apply(parse_date),
                        "HomeTeam": df["Home Team"].apply(standardize_team_name),
                        "AwayTeam": df["Away Team"].apply(standardize_team_name),
                        "HomeGoals": df["HomeGoals"],
                        "AwayGoals": df["AwayGoals"],
                        "Competition": "Champions League",
                        "Season": "2024-2025",
                    }
                )

                # Only include completed matches
                match_df = match_df.dropna(
                    subset=["Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"]
                )
                all_matches.append(match_df)
                print(
                    f"  Loaded champions_league_2024.csv: {len(match_df)} completed matches"
                )

        except Exception as e:
            print(f"  Error loading CL 2024 data: {e}")

    # Load historical CL data from footballcsv
    for filepath in RAW_DATA_DIR.glob("cl_*.csv"):
        try:
            df = pd.read_csv(filepath, encoding="utf-8")

            # footballcsv format varies, handle different column names
            home_col = next(
                (c for c in df.columns if "home" in c.lower() or c == "Team 1"), None
            )
            away_col = next(
                (c for c in df.columns if "away" in c.lower() or c == "Team 2"), None
            )
            score_col = next(
                (c for c in df.columns if "score" in c.lower() or c == "FT"), None
            )
            date_col = next((c for c in df.columns if "date" in c.lower()), None)

            if not all([home_col, away_col]):
                print(f"  Skipping {filepath.name}: cannot identify columns")
                continue

            # Extract season from filename
            season_match = re.search(r"(\d{4})_(\d{2})", filepath.name)
            if season_match:
                season = f"{season_match.group(1)}-20{season_match.group(2)}"
            else:
                season = "Unknown"

            # Parse scores
            def parse_ft_score(score):
                if pd.isna(score):
                    return None, None
                score = str(score).strip()
                match = re.match(r"(\d+)\s*[-:]\s*(\d+)", score)
                if match:
                    return int(match.group(1)), int(match.group(2))
                return None, None

            if score_col:
                scores = df[score_col].apply(lambda x: pd.Series(parse_ft_score(x)))
                df["HomeGoals"] = scores[0]
                df["AwayGoals"] = scores[1]
            else:
                # Try to find separate goal columns
                hg_col = next(
                    (c for c in df.columns if c in ["HG", "FTHG", "Home Goals"]), None
                )
                ag_col = next(
                    (c for c in df.columns if c in ["AG", "FTAG", "Away Goals"]), None
                )
                if hg_col and ag_col:
                    df["HomeGoals"] = pd.to_numeric(df[hg_col], errors="coerce")
                    df["AwayGoals"] = pd.to_numeric(df[ag_col], errors="coerce")
                else:
                    print(f"  Skipping {filepath.name}: cannot find score columns")
                    continue

            match_df = pd.DataFrame(
                {
                    "Date": df[date_col].apply(parse_date) if date_col else None,
                    "HomeTeam": df[home_col].apply(standardize_team_name),
                    "AwayTeam": df[away_col].apply(standardize_team_name),
                    "HomeGoals": df["HomeGoals"],
                    "AwayGoals": df["AwayGoals"],
                    "Competition": "Champions League",
                    "Season": season,
                }
            )

            match_df = match_df.dropna(
                subset=["HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"]
            )
            all_matches.append(match_df)
            print(f"  Loaded {filepath.name}: {len(match_df)} matches")

        except Exception as e:
            print(f"  Error loading {filepath.name}: {e}")

    if all_matches:
        combined = pd.concat(all_matches, ignore_index=True)
        print(f"Total CL matches: {len(combined)}")
        return combined
    return pd.DataFrame()


def merge_and_save():
    """Merge all data sources and save to processed directory."""
    print("\n" + "=" * 50)
    print("Merging all data sources")
    print("=" * 50)

    # Load data from all sources
    league_data = load_football_data_co_uk()
    cl_data = load_champions_league_data()

    # Combine all data
    all_data = []
    if not league_data.empty:
        all_data.append(league_data)
    if not cl_data.empty:
        all_data.append(cl_data)

    if not all_data:
        print("No data to merge!")
        return None

    combined = pd.concat(all_data, ignore_index=True)

    # Sort by date
    combined = combined.sort_values("Date").reset_index(drop=True)

    # Add derived columns
    combined["TotalGoals"] = combined["HomeGoals"] + combined["AwayGoals"]
    combined["GoalDiff"] = combined["HomeGoals"] - combined["AwayGoals"]

    # Result column (H/D/A)
    def get_result(row):
        if row["HomeGoals"] > row["AwayGoals"]:
            return "H"
        elif row["HomeGoals"] < row["AwayGoals"]:
            return "A"
        return "D"

    combined["Result"] = combined.apply(get_result, axis=1)

    # Save to processed directory
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "all_matches.csv"
    combined.to_csv(output_path, index=False)

    print(f"\n" + "=" * 50)
    print(f"Saved {len(combined)} matches to {output_path}")
    print(f"Date range: {combined['Date'].min()} to {combined['Date'].max()}")
    print(f"Competitions: {combined['Competition'].unique()}")
    print("=" * 50)

    return combined


def main():
    """Main preprocessing function."""
    return merge_and_save()


if __name__ == "__main__":
    main()
