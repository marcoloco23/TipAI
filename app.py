"""
TipAI - Minimal Football Score Predictor
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from predict import FootballScorePredictor

st.set_page_config(
    page_title="TipAI",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Minimal mobile-friendly CSS
st.markdown("""
<style>
    .main .block-container {
        padding: 1rem;
        max-width: 500px;
    }

    .title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }

    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1.5rem 0;
    }

    .teams {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }

    .score {
        font-size: 4rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        margin: 0.5rem 0;
    }

    .probs {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin-top: 1rem;
        font-size: 0.95rem;
    }

    .prob-item {
        text-align: center;
    }

    .prob-value {
        font-size: 1.3rem;
        font-weight: 700;
    }

    .prob-label {
        opacity: 0.8;
        font-size: 0.8rem;
    }

    .stSelectbox > div > div {
        border-radius: 12px;
        font-size: 1.1rem;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: 600;
    }

    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    predictor = FootballScorePredictor()
    predictor.load_models(verbose=False)
    return predictor


@st.cache_data
def get_teams():
    elo_path = Path(__file__).parent / "data" / "processed" / "elo_ratings.csv"
    df = pd.read_csv(elo_path)
    return df.sort_values("elo_rating", ascending=False)["team"].tolist()


# Load
predictor = load_predictor()
teams = get_teams()

# Title
st.markdown('<div class="title">⚽ TipAI</div>', unsafe_allow_html=True)

# Team selection
home = st.selectbox("Home", teams, index=0, label_visibility="collapsed",
                    placeholder="Select home team")
away = st.selectbox("Away", teams, index=1, label_visibility="collapsed",
                    placeholder="Select away team")

# Predict button
if st.button("Predict", type="primary", use_container_width=True):
    if home == away:
        st.error("Select different teams")
    else:
        result = predictor.predict(home, away)
        st.session_state.result = result

# Show result
if "result" in st.session_state:
    r = st.session_state.result

    st.markdown(f"""
    <div class="result-card">
        <div class="teams">{r['home_team']} vs {r['away_team']}</div>
        <div class="score">{r['predicted_score']}</div>
        <div class="probs">
            <div class="prob-item">
                <div class="prob-value">{r['home_win_prob']:.0%}</div>
                <div class="prob-label">Home</div>
            </div>
            <div class="prob-item">
                <div class="prob-value">{r['draw_prob']:.0%}</div>
                <div class="prob-label">Draw</div>
            </div>
            <div class="prob-item">
                <div class="prob-value">{r['away_win_prob']:.0%}</div>
                <div class="prob-label">Away</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
