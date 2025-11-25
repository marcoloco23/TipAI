# TipAI - Football Score Predictor

Advanced football score prediction system using an ensemble of XGBoost, LightGBM, and CatBoost models with Optuna hyperparameter optimization.

## Features

- **Ensemble Stacking**: Combines XGBoost, LightGBM, and CatBoost with Ridge meta-learner
- **Optuna Optimization**: 30 trials per model for hyperparameter tuning
- **Elo Rating System**: Dynamic team strength tracking (K=32, home advantage=100)
- **Dixon-Coles Adjustment**: Corrects for correlated low-scoring outcomes
- **Advanced Features**: Momentum, win/lose streaks, exponential weighted averages, attack/defense efficiency
- **Elegant Web App**: Streamlit interface with team comparison and batch predictions

## Installation

```bash
git clone https://github.com/marcoloco23/TipAI.git
cd TipAI
pip install -r requirements.txt
```

## Usage

### Full Pipeline

Download data, preprocess, and train models:

```bash
python run.py all
```

### Individual Commands

```bash
python run.py download    # Download match data from football-data.co.uk
python run.py preprocess  # Clean and normalize data
python run.py train       # Train ensemble models with Optuna
python run.py predict     # Make predictions
```

### Web App

```bash
streamlit run app.py
```

### Quick Predictions

```bash
python get_predictions.py
```

## Project Structure

```
TipAI/
├── app.py                  # Streamlit web application
├── run.py                  # CLI pipeline runner
├── get_predictions.py      # Quick prediction script
├── requirements.txt        # Python dependencies
└── src/
    ├── download_data.py    # Downloads historical match data
    ├── preprocess.py       # Data cleaning and normalization
    ├── train_advanced.py   # Optuna-tuned ensemble training
    └── predict.py          # Score prediction with stacked models
```

## Model Architecture

1. **Base Models** (Poisson regression):
   - XGBoost (home goals, away goals)
   - LightGBM (home goals, away goals)
   - CatBoost (home goals, away goals)

2. **Meta-Learner**: Ridge regression stacking

3. **Score Selection**:
   - Bivariate Poisson distribution for score probabilities
   - Dixon-Coles adjustment (ρ=-0.20) for low scores
   - Lambda scaling (1.8x) for prediction variance
   - Smart selection based on most likely outcome

## Data

- **Source**: football-data.co.uk
- **Leagues**: Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Champions League
- **Matches**: 18,000+ historical matches
- **Features**: ~70 engineered features including Elo, form, H2H stats

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- xgboost, lightgbm, catboost
- optuna, streamlit

## License

MIT
