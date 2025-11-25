"""
Minimal Streamlit app for football score predictions.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from predict import FootballScorePredictor

# Page config
st.set_page_config(
    page_title="Football Score Predictor",
    page_icon="⚽",
    layout="centered"
)

@st.cache_resource
def load_predictor():
    """Load predictor once and cache."""
    predictor = FootballScorePredictor()
    predictor.load_models()
    return predictor

@st.cache_data
def get_teams():
    """Get sorted list of teams."""
    elo_path = Path(__file__).parent / "data" / "processed" / "elo_ratings.csv"
    df = pd.read_csv(elo_path)
    return df.sort_values('elo_rating', ascending=False)['team'].tolist()

# Load data
predictor = load_predictor()
teams = get_teams()

# UI
st.title("⚽ Football Score Predictor")

# Initialize session state for matches
if 'matches' not in st.session_state:
    st.session_state.matches = []

# Add match section
st.markdown("### Add Matches")
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    home_team = st.selectbox("Home", teams, key="home")

with col2:
    away_team = st.selectbox("Away", teams, index=1, key="away")

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("➕ Add", use_container_width=True):
        if home_team != away_team:
            match = (home_team, away_team)
            if match not in st.session_state.matches:
                st.session_state.matches.append(match)

# Show added matches
if st.session_state.matches:
    st.markdown("### Matches to Predict")

    for i, (home, away) in enumerate(st.session_state.matches):
        col1, col2 = st.columns([5, 1])
        col1.text(f"{home} vs {away}")
        if col2.button("✕", key=f"del_{i}"):
            st.session_state.matches.pop(i)
            st.rerun()

    col1, col2 = st.columns(2)
    if col1.button("🔮 Predict All", type="primary", use_container_width=True):
        st.markdown("---")
        st.markdown("### Predictions")

        for home, away in st.session_state.matches:
            result = predictor.predict(home, away)

            with st.container():
                cols = st.columns([3, 1, 1, 1])
                cols[0].markdown(f"**{home}** vs **{away}**")
                cols[1].markdown(f"### {result['predicted_score']}")
                cols[2].markdown(f"*{result['confidence']:.1%}*")

                # Mini probabilities
                probs = f"H:{result['home_win_prob']:.0%} D:{result['draw_prob']:.0%} A:{result['away_win_prob']:.0%}"
                cols[3].caption(probs)

            st.markdown("---")

    if col2.button("🗑️ Clear All", use_container_width=True):
        st.session_state.matches = []
        st.rerun()
