"""
Download historical football match data from various sources.
- football-data.co.uk: Top 5 European leagues (10 seasons)
- fixturedownload.com: Champions League current season
"""

import os
import requests
import pandas as pd
from pathlib import Path
import time

# Base directories
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# football-data.co.uk league codes
LEAGUES = {
    "E0": "England Premier League",
    "SP1": "Spain La Liga",
    "D1": "Germany Bundesliga",
    "I1": "Italy Serie A",
    "F1": "France Ligue 1",
}

# Seasons to download (last 10)
SEASONS = [
    "1516",
    "1617",
    "1718",
    "1819",
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
    "2425",
]


def download_football_data_co_uk():
    """Download CSV files from football-data.co.uk for top 5 leagues."""
    print("Downloading data from football-data.co.uk...")

    base_url = "https://www.football-data.co.uk/mmz4281"
    downloaded = 0
    failed = []

    for season in SEASONS:
        for league_code, league_name in LEAGUES.items():
            url = f"{base_url}/{season}/{league_code}.csv"
            filename = f"{league_code}_{season}.csv"
            filepath = RAW_DATA_DIR / filename

            # Skip if already downloaded
            if filepath.exists():
                print(f"  Skipping {filename} (already exists)")
                continue

            try:
                print(f"  Downloading {filename}...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                with open(filepath, "wb") as f:
                    f.write(response.content)
                downloaded += 1
                time.sleep(0.5)  # Be nice to the server

            except requests.exceptions.RequestException as e:
                print(f"    Failed to download {filename}: {e}")
                failed.append(filename)

    print(f"\nDownloaded {downloaded} files, {len(failed)} failed")
    if failed:
        print(f"Failed files: {failed}")

    return downloaded, failed


def download_champions_league_data():
    """Download Champions League data from fixturedownload.com."""
    print("\nDownloading Champions League data...")

    # Current season CL data
    cl_url = "https://fixturedownload.com/download/champions-league-2024-UTC.csv"
    filepath = RAW_DATA_DIR / "champions_league_2024.csv"

    if filepath.exists():
        print("  Skipping champions_league_2024.csv (already exists)")
        return True

    try:
        print(f"  Downloading champions_league_2024.csv...")
        response = requests.get(cl_url, timeout=30)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(response.content)
        print("  Downloaded successfully")
        return True

    except requests.exceptions.RequestException as e:
        print(f"  Failed to download CL data: {e}")
        return False


def download_historical_cl_data():
    """Download historical Champions League data from GitHub."""
    print("\nDownloading historical Champions League data from GitHub...")

    # footballcsv European Champions League data
    cl_urls = [
        (
            "https://raw.githubusercontent.com/footballcsv/cache.footballdata/refs/heads/master/1-europe.clubs/2023-24/cl.csv",
            "cl_2023_24.csv",
        ),
        (
            "https://raw.githubusercontent.com/footballcsv/cache.footballdata/refs/heads/master/1-europe.clubs/2022-23/cl.csv",
            "cl_2022_23.csv",
        ),
        (
            "https://raw.githubusercontent.com/footballcsv/cache.footballdata/refs/heads/master/1-europe.clubs/2021-22/cl.csv",
            "cl_2021_22.csv",
        ),
        (
            "https://raw.githubusercontent.com/footballcsv/cache.footballdata/refs/heads/master/1-europe.clubs/2020-21/cl.csv",
            "cl_2020_21.csv",
        ),
        (
            "https://raw.githubusercontent.com/footballcsv/cache.footballdata/refs/heads/master/1-europe.clubs/2019-20/cl.csv",
            "cl_2019_20.csv",
        ),
    ]

    downloaded = 0
    for url, filename in cl_urls:
        filepath = RAW_DATA_DIR / filename

        if filepath.exists():
            print(f"  Skipping {filename} (already exists)")
            continue

        try:
            print(f"  Downloading {filename}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(response.content)
            downloaded += 1
            time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            print(f"    Failed to download {filename}: {e}")

    print(f"Downloaded {downloaded} CL historical files")
    return downloaded


def verify_downloads():
    """Verify downloaded files and show summary."""
    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)

    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    print(f"Total CSV files: {len(csv_files)}")

    total_matches = 0
    for filepath in sorted(csv_files):
        try:
            df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
            matches = len(df)
            total_matches += matches
            print(f"  {filepath.name}: {matches} matches")
        except Exception as e:
            print(f"  {filepath.name}: Error reading - {e}")

    print(f"\nTotal matches across all files: {total_matches}")
    return total_matches


def main():
    """Main function to download all data."""
    # Create directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Football Score Prediction - Data Download")
    print("=" * 50)

    # Download from each source
    download_football_data_co_uk()
    download_champions_league_data()
    download_historical_cl_data()

    # Verify downloads
    verify_downloads()

    print("\nData download complete!")


if __name__ == "__main__":
    main()
