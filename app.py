"""
Football Score Predictor - Elegant Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from predict import FootballScorePredictor

# Page config
st.set_page_config(
    page_title="TipAI - Football Predictions",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for elegant styling
st.markdown(
    """
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Header styling */
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .header-subtitle {
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Match card styling */
    .match-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .match-teams {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }

    .match-score {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }

    .confidence-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Probability bar */
    .prob-container {
        display: flex;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }

    .prob-home { background: #3b82f6; }
    .prob-draw { background: #6b7280; }
    .prob-away { background: #ef4444; }

    /* Team selector */
    .stSelectbox > div > div {
        border-radius: 12px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    /* Queue item */
    .queue-item {
        background: #f8fafc;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #e2e8f0;
    }

    /* Stats box */
    .stat-box {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
    }

    .stat-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor():
    """Load predictor once and cache."""
    predictor = FootballScorePredictor()
    predictor.load_models(verbose=False)
    return predictor


@st.cache_data
def get_teams_df():
    """Get teams with Elo ratings."""
    elo_path = Path(__file__).parent / "data" / "processed" / "elo_ratings.csv"
    df = pd.read_csv(elo_path)
    return df.sort_values("elo_rating", ascending=False)


def get_team_tier(elo):
    """Get team tier based on Elo."""
    if elo >= 1800:
        return "Elite", "🏆"
    elif elo >= 1700:
        return "Top", "⭐"
    elif elo >= 1600:
        return "Strong", "💪"
    elif elo >= 1500:
        return "Average", "📊"
    else:
        return "Developing", "📈"


def render_probability_bar(home_prob, draw_prob, away_prob):
    """Render a visual probability bar."""
    return f"""
    <div class="prob-container">
        <div class="prob-home" style="width: {home_prob*100}%"></div>
        <div class="prob-draw" style="width: {draw_prob*100}%"></div>
        <div class="prob-away" style="width: {away_prob*100}%"></div>
    </div>
    """


def render_match_card(result):
    """Render a prediction result card."""
    home = result["home_team"]
    away = result["away_team"]
    score = result["predicted_score"]
    conf = result["confidence"]
    home_prob = result["home_win_prob"]
    draw_prob = result["draw_prob"]
    away_prob = result["away_win_prob"]

    # Determine favorite
    if home_prob > away_prob and home_prob > draw_prob:
        outcome = "Home Win"
        outcome_color = "#3b82f6"
    elif away_prob > home_prob and away_prob > draw_prob:
        outcome = "Away Win"
        outcome_color = "#ef4444"
    else:
        outcome = "Draw"
        outcome_color = "#6b7280"

    st.markdown(
        f"""
    <div class="match-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="flex: 1;">
                <div class="match-teams">{home}</div>
                <div style="color: #64748b; font-size: 0.9rem;">vs</div>
                <div class="match-teams">{away}</div>
            </div>
            <div style="text-align: center; padding: 0 2rem;">
                <div class="match-score">{score}</div>
                <div class="confidence-badge">{conf:.1%} confidence</div>
            </div>
            <div style="flex: 1; text-align: right;">
                <div style="color: {outcome_color}; font-weight: 600; font-size: 1.1rem;">{outcome}</div>
                <div style="margin-top: 0.5rem;">
                    <span style="color: #3b82f6;">H {home_prob:.0%}</span> ·
                    <span style="color: #6b7280;">D {draw_prob:.0%}</span> ·
                    <span style="color: #ef4444;">A {away_prob:.0%}</span>
                </div>
            </div>
        </div>
        {render_probability_bar(home_prob, draw_prob, away_prob)}
    </div>
    """,
        unsafe_allow_html=True,
    )


# Initialize
predictor = load_predictor()
teams_df = get_teams_df()
teams = teams_df["team"].tolist()

# Initialize session state
if "matches" not in st.session_state:
    st.session_state.matches = []
if "predictions" not in st.session_state:
    st.session_state.predictions = []

# Header
st.markdown(
    '<div class="header-title">TipAI Football Predictor</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="header-subtitle">Advanced ML predictions using XGBoost, LightGBM & CatBoost ensemble</div>',
    unsafe_allow_html=True,
)

# Main layout
col_left, col_right = st.columns([1, 1.5])

with col_left:
    st.markdown("### Add Match")

    # Team selection with search
    home_team = st.selectbox(
        "🏠 Home Team", teams, index=0, help="Select the home team"
    )

    away_team = st.selectbox(
        "✈️ Away Team",
        teams,
        index=1 if len(teams) > 1 else 0,
        help="Select the away team",
    )

    # Show team comparison
    if home_team and away_team and home_team != away_team:
        home_elo = teams_df[teams_df["team"] == home_team]["elo_rating"].values[0]
        away_elo = teams_df[teams_df["team"] == away_team]["elo_rating"].values[0]

        home_tier, home_icon = get_team_tier(home_elo)
        away_tier, away_icon = get_team_tier(away_elo)

        st.markdown("#### Team Comparison")
        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.markdown(
                f"""
            <div class="stat-box">
                <div style="font-size: 1.5rem;">{home_icon}</div>
                <div class="stat-value">{home_elo:.0f}</div>
                <div class="stat-label">{home_tier}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with comp_col2:
            st.markdown(
                f"""
            <div class="stat-box">
                <div style="font-size: 1.5rem;">{away_icon}</div>
                <div class="stat-value">{away_elo:.0f}</div>
                <div class="stat-label">{away_tier}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        elo_diff = home_elo - away_elo
        if abs(elo_diff) > 100:
            st.info(
                f"{'Home' if elo_diff > 0 else 'Away'} team rated {abs(elo_diff):.0f} Elo points higher"
            )

    st.markdown("")

    # Add match button
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button("➕ Add to Queue", use_container_width=True, type="primary"):
            if home_team != away_team:
                match = (home_team, away_team)
                if match not in st.session_state.matches:
                    st.session_state.matches.append(match)
                    st.rerun()
                else:
                    st.warning("Match already in queue")
            else:
                st.error("Select different teams")

    with btn_col2:
        if st.button("⚡ Quick Predict", use_container_width=True):
            if home_team != away_team:
                result = predictor.predict(home_team, away_team)
                st.session_state.predictions = [result]
                st.rerun()

    # Match Queue
    if st.session_state.matches:
        st.markdown("---")
        st.markdown(f"### Match Queue ({len(st.session_state.matches)})")

        for i, (home, away) in enumerate(st.session_state.matches):
            queue_col1, queue_col2 = st.columns([5, 1])
            with queue_col1:
                st.markdown(f"**{i+1}.** {home} vs {away}")
            with queue_col2:
                if st.button("✕", key=f"del_{i}", help="Remove"):
                    st.session_state.matches.pop(i)
                    st.rerun()

        st.markdown("")
        action_col1, action_col2 = st.columns(2)

        with action_col1:
            if st.button("🔮 Predict All", use_container_width=True, type="primary"):
                with st.spinner("Running predictions..."):
                    st.session_state.predictions = [
                        predictor.predict(home, away)
                        for home, away in st.session_state.matches
                    ]
                st.rerun()

        with action_col2:
            if st.button("🗑️ Clear Queue", use_container_width=True):
                st.session_state.matches = []
                st.session_state.predictions = []
                st.rerun()

with col_right:
    st.markdown("### Predictions")

    if st.session_state.predictions:
        for result in st.session_state.predictions:
            render_match_card(result)

        # Summary stats
        st.markdown("---")
        st.markdown("#### Summary")

        home_wins = sum(
            1
            for r in st.session_state.predictions
            if r["home_win_prob"] > r["away_win_prob"]
            and r["home_win_prob"] > r["draw_prob"]
        )
        away_wins = sum(
            1
            for r in st.session_state.predictions
            if r["away_win_prob"] > r["home_win_prob"]
            and r["away_win_prob"] > r["draw_prob"]
        )
        draws = len(st.session_state.predictions) - home_wins - away_wins

        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        sum_col1.metric("Total", len(st.session_state.predictions))
        sum_col2.metric("Home Wins", home_wins)
        sum_col3.metric("Draws", draws)
        sum_col4.metric("Away Wins", away_wins)

        # Export option
        st.markdown("")
        if st.button("📋 Copy Results"):
            results_text = "\n".join(
                [
                    f"{r['home_team']} vs {r['away_team']}: {r['predicted_score']} ({r['confidence']:.1%})"
                    for r in st.session_state.predictions
                ]
            )
            st.code(results_text, language=None)
    else:
        st.markdown(
            """
        <div style="text-align: center; padding: 3rem; color: #94a3b8;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚽</div>
            <div style="font-size: 1.2rem;">No predictions yet</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem;">Add matches and click Predict to see results</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem;">
    Powered by XGBoost + LightGBM + CatBoost ensemble · Trained on 18,000+ matches · Elo-weighted predictions
</div>
""",
    unsafe_allow_html=True,
)
