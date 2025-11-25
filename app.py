"""
TipAI - Football Score Predictor

Elegant UI for predicting football match scores using ensemble ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent / "src"))
from predict import FootballScorePredictor

# =============================================================================
# Theme Configuration
# =============================================================================

THEME = {
    # Colors
    "bg_deep": "#0a0a0b",
    "bg_card": "#111113",
    "bg_elevated": "#18181b",
    "border": "rgba(255,255,255,0.06)",
    "text_primary": "#fafafa",
    "text_muted": "rgba(255,255,255,0.35)",
    "text_secondary": "rgba(255,255,255,0.6)",
    "accent_gold": "#d4a574",
    "accent_gold_dim": "rgba(212,165,116,0.15)",
    "color_home": "#34d399",
    "color_draw": "#64748b",
    "color_away": "#f43f5e",
    # Fonts
    "font_serif": "Cormorant Garamond,Georgia,serif",
    "font_sans": "DM Sans,sans-serif",
}


# =============================================================================
# HTML Components
# =============================================================================

def html(content: str) -> None:
    """Render HTML with unsafe_allow_html enabled."""
    st.markdown(content, unsafe_allow_html=True)


def render_brand() -> None:
    """Render the brand header."""
    html(
        f'<div style="text-align:center;margin-bottom:2rem;">'
        f'<div style="font-size:2.2rem;margin-bottom:0.5rem;">⚽</div>'
        f'<div style="font-family:{THEME["font_serif"]};font-size:2rem;font-weight:600;'
        f'letter-spacing:0.15em;color:{THEME["text_primary"]};">TIPAI</div>'
        f'<div style="font-family:{THEME["font_sans"]};font-size:0.7rem;letter-spacing:0.25em;'
        f'color:{THEME["text_muted"]};margin-top:0.3rem;">SCORE PREDICTION</div>'
        f'</div>'
    )


def render_match_header(home_team: str, away_team: str) -> None:
    """Render the match header with team names."""
    html(
        f'<div style="padding:1.5rem;border-bottom:1px solid {THEME["border"]};'
        f'background:linear-gradient(180deg,rgba(255,255,255,0.02) 0%,transparent 100%);">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div style="font-family:{THEME["font_sans"]};font-size:0.9rem;font-weight:600;'
        f'color:{THEME["text_primary"]};max-width:40%;">{home_team}</div>'
        f'<span style="font-family:{THEME["font_serif"]};font-size:0.75rem;'
        f'color:{THEME["text_muted"]};letter-spacing:0.1em;">vs</span>'
        f'<div style="font-family:{THEME["font_sans"]};font-size:0.9rem;font-weight:600;'
        f'color:{THEME["text_primary"]};max-width:40%;text-align:right;">{away_team}</div>'
        f'</div></div>'
    )


def render_score(home_goals: int, away_goals: int, confidence: float) -> None:
    """Render the predicted score section."""
    html(
        f'<div style="padding:2rem 1.5rem;text-align:center;'
        f'background:radial-gradient(ellipse at 50% 0%,rgba(212,165,116,0.04) 0%,transparent 70%);">'
        f'<div style="font-family:{THEME["font_sans"]};font-size:0.6rem;letter-spacing:0.2em;'
        f'text-transform:uppercase;color:{THEME["accent_gold"]};margin-bottom:0.75rem;">Predicted Score</div>'
        f'<div style="font-family:{THEME["font_serif"]};font-size:4.5rem;font-weight:600;'
        f'color:{THEME["text_primary"]};letter-spacing:0.08em;line-height:1;">'
        f'{home_goals}<span style="color:{THEME["text_muted"]};font-weight:400;margin:0 0.1em;">–</span>{away_goals}</div>'
        f'<div style="display:inline-flex;align-items:center;gap:0.4rem;margin-top:1rem;'
        f'padding:0.4rem 0.9rem;background:{THEME["accent_gold_dim"]};'
        f'border:1px solid rgba(212,165,116,0.2);border-radius:100px;">'
        f'<span style="width:5px;height:5px;border-radius:50%;background:{THEME["accent_gold"]};"></span>'
        f'<span style="font-family:{THEME["font_sans"]};font-size:0.72rem;font-weight:500;'
        f'color:{THEME["accent_gold"]};">{confidence:.1%} probability</span>'
        f'</div></div>'
    )


def render_xg_row(xg_home: float, xg_away: float) -> None:
    """Render expected goals row."""
    xg_total = xg_home + xg_away
    items = [("xG Home", xg_home), ("Total", xg_total), ("xG Away", xg_away)]

    content = f'<div style="display:flex;border-top:1px solid {THEME["border"]};border-bottom:1px solid {THEME["border"]};">'
    for i, (label, value) in enumerate(items):
        border = f'border-right:1px solid {THEME["border"]};' if i < 2 else ""
        content += (
            f'<div style="flex:1;padding:1rem;text-align:center;{border}">'
            f'<div style="font-family:{THEME["font_serif"]};font-size:1.5rem;font-weight:600;'
            f'color:{THEME["text_primary"]};">{value:.2f}</div>'
            f'<div style="font-family:{THEME["font_sans"]};font-size:0.6rem;letter-spacing:0.12em;'
            f'text-transform:uppercase;color:{THEME["text_muted"]};margin-top:0.25rem;">{label}</div></div>'
        )
    content += '</div>'
    html(content)


def render_distribution(scores: List[Tuple[int, int, float]]) -> None:
    """Render score distribution bars."""
    html(
        f'<div style="padding:1.25rem 1.5rem 0.5rem;">'
        f'<div style="font-family:{THEME["font_sans"]};font-size:0.6rem;letter-spacing:0.15em;'
        f'text-transform:uppercase;color:{THEME["text_muted"]};">Score Distribution</div></div>'
    )

    max_prob = scores[0][2]
    for home, away, prob in scores:
        width = (prob / max_prob) * 100
        color = THEME["color_home"] if home > away else THEME["color_away"] if away > home else THEME["color_draw"]
        html(
            f'<div style="display:flex;align-items:center;gap:0.75rem;padding:0 1.5rem;margin-bottom:0.5rem;">'
            f'<span style="font-family:{THEME["font_serif"]};font-size:1rem;font-weight:500;'
            f'color:{THEME["text_secondary"]};width:36px;text-align:center;">{home}–{away}</span>'
            f'<div style="flex:1;height:3px;background:{THEME["bg_elevated"]};border-radius:2px;overflow:hidden;">'
            f'<div style="height:100%;width:{width}%;background:{color};border-radius:2px;"></div></div>'
            f'<span style="font-family:{THEME["font_sans"]};font-size:0.72rem;font-weight:500;'
            f'color:{THEME["text_muted"]};width:40px;text-align:right;">{prob:.1%}</span></div>'
        )


def render_outcomes(home_prob: float, draw_prob: float, away_prob: float) -> None:
    """Render 1X2 outcome probabilities."""
    outcomes = [
        ("Home", home_prob, THEME["color_home"]),
        ("Draw", draw_prob, THEME["color_draw"]),
        ("Away", away_prob, THEME["color_away"]),
    ]

    content = f'<div style="display:flex;padding:1.25rem 1.5rem;background:{THEME["bg_elevated"]};margin-top:0.75rem;">'
    for i, (label, prob, color) in enumerate(outcomes):
        if i > 0:
            content += f'<div style="width:1px;background:{THEME["border"]};"></div>'
        content += (
            f'<div style="flex:1;text-align:center;">'
            f'<div style="font-family:{THEME["font_serif"]};font-size:1.4rem;font-weight:600;color:{color};">{prob:.0%}</div>'
            f'<div style="font-family:{THEME["font_sans"]};font-size:0.58rem;letter-spacing:0.1em;'
            f'text-transform:uppercase;color:{THEME["text_muted"]};margin-top:0.2rem;">{label}</div></div>'
        )
    content += '</div>'
    html(content)


def render_prediction_card(result: Dict) -> None:
    """Render the complete prediction card."""
    top_scores = get_top_scores(result["prob_matrix"], n=5)

    # Card container
    html(
        f'<div style="background:{THEME["bg_card"]};border:1px solid {THEME["border"]};'
        f'border-radius:20px;overflow:hidden;box-shadow:0 20px 50px -12px rgba(0,0,0,0.5);margin:1.5rem 0;">'
    )

    render_match_header(result["home_team"], result["away_team"])
    render_score(result["home_goals"], result["away_goals"], result["confidence"])
    render_xg_row(result["lambda_home"], result["lambda_away"])
    render_distribution(top_scores)
    render_outcomes(result["home_win_prob"], result["draw_prob"], result["away_win_prob"])

    html('</div>')


# =============================================================================
# Data Functions
# =============================================================================

@st.cache_resource
def load_predictor() -> FootballScorePredictor:
    """Load and cache the predictor model."""
    predictor = FootballScorePredictor()
    predictor.load_models(verbose=False)
    return predictor


@st.cache_data
def get_teams() -> List[str]:
    """Load teams sorted by Elo rating."""
    elo_path = Path(__file__).parent / "data" / "processed" / "elo_ratings.csv"
    df = pd.read_csv(elo_path)
    return df.sort_values("elo_rating", ascending=False)["team"].tolist()


def get_top_scores(prob_matrix: np.ndarray, n: int = 5) -> List[Tuple[int, int, float]]:
    """Extract top N most likely scores from probability matrix."""
    scores = [
        (h, a, prob_matrix[h, a])
        for h in range(prob_matrix.shape[0])
        for a in range(prob_matrix.shape[1])
    ]
    return sorted(scores, key=lambda x: x[2], reverse=True)[:n]


# =============================================================================
# Page Setup & Styles
# =============================================================================

st.set_page_config(
    page_title="TipAI",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed",
)

html(f"""
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    .stApp {{ background: {THEME["bg_deep"]} !important; }}
    .main .block-container {{ padding: 2rem 1rem !important; max-width: 460px !important; }}
    .stSelectbox > div > div {{ background: {THEME["bg_card"]} !important; border: 1px solid {THEME["border"]} !important; border-radius: 12px !important; }}
    .stSelectbox label {{ display: none !important; }}
    .stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, {THEME["accent_gold"]} 0%, #c4956a 100%) !important;
        border: none !important; border-radius: 12px !important; padding: 0.85rem !important;
        font-family: {THEME["font_sans"]} !important; font-size: 0.85rem !important;
        font-weight: 600 !important; letter-spacing: 0.08em !important;
        text-transform: uppercase !important; color: {THEME["bg_deep"]} !important;
        box-shadow: 0 4px 12px rgba(212,165,116,0.25) !important;
    }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .stDeployButton {{ display: none; }}
</style>
""")


# =============================================================================
# Main App
# =============================================================================

def main():
    """Main application entry point."""
    predictor = load_predictor()
    teams = get_teams()

    render_brand()

    home = st.selectbox("Home", teams, index=0, label_visibility="collapsed", placeholder="Home team")
    away = st.selectbox("Away", teams, index=1, label_visibility="collapsed", placeholder="Away team")

    if st.button("Predict", type="primary", use_container_width=True):
        if home == away:
            st.error("Select different teams")
        else:
            st.session_state.result = predictor.predict(home, away)

    if "result" in st.session_state:
        render_prediction_card(st.session_state.result)


if __name__ == "__main__":
    main()
