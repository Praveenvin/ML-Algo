"""Theme constants and CSS injection for the ML Studio app."""

import streamlit as st

PRIMARY = "#8E44AD"
PRIMARY_DARK = "#6C3483"
LAVENDER = "#F4E6FD"
BG = "#FFFFFF"
BG_ALT = "#FAFAFC"
BORDER = "#E8E3EE"
TEXT = "#2B2430"
TEXT_MUTED = "#6B6373"
SUCCESS = "#2E7D32"
WARNING = "#B8860B"
DANGER = "#C0392B"

APP_CSS = f"""
<style>
    .stApp {{
        background-color: {BG};
    }}

    #MainMenu, footer, header {{ visibility: hidden; }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}

    h1, h2, h3, h4 {{
        color: {TEXT};
        font-weight: 600;
    }}

    p, label, span, div {{
        color: {TEXT};
    }}

    .ml-studio-header {{
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
        padding-bottom: 1.25rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid {BORDER};
    }}

    .ml-studio-title {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {TEXT};
        margin: 0;
    }}

    .ml-studio-title span {{
        color: {PRIMARY};
    }}

    .ml-studio-subtitle {{
        font-size: 0.95rem;
        color: {TEXT_MUTED};
        margin: 0;
    }}

    .ms-card {{
        background-color: {BG_ALT};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }}

    .ms-metric-label {{
        font-size: 0.78rem;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }}

    .ms-metric-value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {TEXT};
    }}

    .ms-metric-value.accent {{
        color: {PRIMARY};
    }}

    .ms-badge {{
        display: inline-block;
        background-color: {LAVENDER};
        color: {PRIMARY_DARK};
        border-radius: 999px;
        padding: 0.2rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 600;
    }}

    .ms-empty-state {{
        background-color: {BG_ALT};
        border: 1px dashed {BORDER};
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        color: {TEXT_MUTED};
    }}

    .ms-section-title {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {TEXT};
        margin-top: 1.5rem;
        margin-bottom: 0.6rem;
    }}

    div[data-testid="stSidebar"] {{
        background-color: {BG_ALT};
        border-right: 1px solid {BORDER};
    }}

    div[data-testid="stSidebar"] .stRadio label {{
        font-size: 0.95rem;
    }}

    .stButton > button, .stDownloadButton > button {{
        background-color: {PRIMARY};
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }}

    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: {PRIMARY_DARK};
        color: white;
    }}

    div[data-testid="stMetric"] {{
        background-color: {BG_ALT};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }}

    .stDataFrame {{
        border: 1px solid {BORDER};
        border-radius: 10px;
    }}

    .ms-footer-note {{
        color: {TEXT_MUTED};
        font-size: 0.8rem;
        margin-top: 2rem;
    }}
</style>
"""


def inject_global_styles() -> None:
    """Inject the shared ML Studio CSS theme into the current Streamlit page."""
    st.markdown(APP_CSS, unsafe_allow_html=True)


def render_header(subtitle: str = "Explore datasets. Train models. Understand results.") -> None:
    """Render the consistent top-of-page app header."""
    st.markdown(
        f"""
        <div class="ml-studio-header">
            <p class="ml-studio-title">ML <span>Studio</span></p>
            <p class="ml-studio-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
