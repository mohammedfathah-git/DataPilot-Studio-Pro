"""
DataPilot Studio Pro - Modern High-Contrast Theme System
Guarantees crisp, dark typography in Light Mode and radiant clarity in Dark Mode.
"""
import streamlit as st

def inject_modern_theme(is_dark: bool = True):
    if is_dark:
        BG = "#0B0F19"
        CARD_BG = "#131C31"
        CARD_ALT = "#1A2542"
        BORDER = "rgba(139, 92, 246, 0.25)"
        BORDER_LIGHT = "rgba(255, 255, 255, 0.10)"
        TEXT = "#F8FAFC"
        TEXT_MUTED = "#94A3B8"
        ACCENT = "#6366F1"
        ACCENT_GRADIENT = "linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)"
        INPUT_BG = "#111827"
        INPUT_BORDER = "rgba(255, 255, 255, 0.18)"
        POPOVER_BG = "#131C31"
        SIDEBAR_BG = "#0E1526"
    else:
        BG = "#F4F6FB"
        CARD_BG = "#FFFFFF"
        CARD_ALT = "#F8FAFC"
        BORDER = "rgba(99, 102, 241, 0.25)"
        BORDER_LIGHT = "#E2E8F0"
        TEXT = "#0F172A"          # Deep dark charcoal
        TEXT_MUTED = "#334155"    # Dark slate (ultra readable on white)
        ACCENT = "#4F46E5"
        ACCENT_GRADIENT = "linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%)"
        INPUT_BG = "#FFFFFF"
        INPUT_BORDER = "#CBD5E1"
        POPOVER_BG = "#FFFFFF"
        SIDEBAR_BG = "#FFFFFF"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {{
        color-scheme: {"dark" if is_dark else "light"} !important;
        --dp-bg: {BG};
        --dp-card: {CARD_BG};
        --dp-card-alt: {CARD_ALT};
        --dp-border: {BORDER};
        --dp-border-light: {BORDER_LIGHT};
        --dp-text: {TEXT};
        --dp-muted: {TEXT_MUTED};
        --dp-accent: {ACCENT};
        --dp-gradient: {ACCENT_GRADIENT};
        --dp-input: {INPUT_BG};
        --dp-input-border: {INPUT_BORDER};
        --dp-popover: {POPOVER_BG};
        --dp-sidebar: {SIDEBAR_BG};
    }}

    html, body, [class*="css"], .stApp {{
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
    }}

    .stApp {{
        background: radial-gradient(circle at 85% 5%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
                    var(--dp-bg) !important;
        color: var(--dp-text) !important;
    }}

    .block-container {{
        max-width: 1200px !important;
        padding-top: 1rem !important;
        padding-bottom: 3.5rem !important;
    }}

    /* Global Headings & Text */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--dp-text) !important;
        font-weight: 750 !important;
        letter-spacing: -0.02em !important;
    }}
    p, span, div, li, strong, b {{
        color: var(--dp-text);
    }}
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] div,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li {{
        color: var(--dp-text) !important;
    }}

    /* Captions Everywhere */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] *,
    .stCaption,
    .stCaption * {{
        color: var(--dp-muted) !important;
        font-weight: 550 !important;
    }}

    /* Form Widget Labels (Selectboxes, radios, checkboxes) */
    label,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    .stSelectbox label,
    .stSelectbox label *,
    .stRadio label,
    .stRadio label *,
    .stCheckbox label,
    .stCheckbox label * {{
        color: var(--dp-text) !important;
        font-weight: 650 !important;
        font-size: 13px !important;
    }}

    /* Inputs, Selectboxes, Dropdowns */
    .stApp input,
    .stApp textarea,
    [data-baseweb="select"] > div {{
        background-color: var(--dp-input) !important;
        color: var(--dp-text) !important;
        border: 1px solid var(--dp-input-border) !important;
        border-radius: 8px !important;
    }}
    [data-baseweb="select"] * {{
        color: var(--dp-text) !important;
    }}
    [data-baseweb="popover"],
    [data-baseweb="menu"] {{
        background-color: var(--dp-popover) !important;
        border: 1px solid var(--dp-input-border) !important;
        border-radius: 10px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15) !important;
    }}
    [data-baseweb="menu"] li,
    [data-baseweb="popover"] li,
    [data-baseweb="menu"] li *,
    [data-baseweb="popover"] li * {{
        color: var(--dp-text) !important;
    }}

    /* Metrics (Forces labels to be 100% visible) */
    div[data-testid="stMetric"] {{
        background: var(--dp-card) !important;
        border: 1px solid var(--dp-border-light) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02) !important;
    }}
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricLabel"] label,
    div[data-testid="stMetricLabel"] div,
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] span {{
        color: var(--dp-muted) !important;
        font-size: 13px !important;
        font-weight: 650 !important;
    }}
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] *,
    div[data-testid="stMetricValue"] div {{
        color: var(--dp-text) !important;
        font-weight: 800 !important;
        font-size: 25px !important;
    }}

    /* Expanders */
    [data-testid="stExpander"],
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {{
        color: var(--dp-text) !important;
        font-weight: 650 !important;
    }}

    /* Cards */
    .dp-card {{
        background: var(--dp-card) !important;
        border: 1px solid var(--dp-border-light) !important;
        border-radius: 14px !important;
        padding: 20px 24px !important;
        margin-bottom: 18px !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.04) !important;
    }}
    .dp-card,
    .dp-card p,
    .dp-card div,
    .dp-card span,
    .dp-card li,
    .dp-card strong {{
        color: var(--dp-text) !important;
    }}
    .dp-card:hover {{
        border-color: var(--dp-border) !important;
    }}

    .dp-hero {{
        display: flex;
        align-items: center;
        gap: 18px;
        background: var(--dp-card) !important;
        border: 1px solid var(--dp-border) !important;
        border-radius: 16px !important;
        padding: 20px 24px !important;
        margin-bottom: 22px !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.05) !important;
    }}
    .dp-hero * {{
        color: var(--dp-text);
    }}
    .dp-hero-sub {{
        font-size: 13.5px !important;
        color: var(--dp-muted) !important;
        margin-top: 4px !important;
    }}
    .dp-hero-icon {{
        width: 52px;
        height: 52px;
        min-width: 52px;
        border-radius: 14px;
        background: var(--dp-gradient) !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        color: #ffffff !important;
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.3) !important;
    }}
    .dp-hero-icon * {{
        color: #ffffff !important;
    }}

    /* Buttons */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"],
    .stDownloadButton > button[kind="primary"] {{
        background: var(--dp-gradient) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.25rem !important;
        font-weight: 650 !important;
        font-size: 13.5px !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3) !important;
    }}
    .stButton > button[kind="primary"] *,
    .stDownloadButton > button[kind="primary"] * {{
        color: #FFFFFF !important;
    }}

    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="baseButton-secondary"],
    .stDownloadButton > button[kind="secondary"] {{
        background: var(--dp-card-alt) !important;
        color: var(--dp-text) !important;
        border: 1px solid var(--dp-border-light) !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }}
    .stButton > button[kind="secondary"] *,
    .stDownloadButton > button[kind="secondary"] * {{
        color: var(--dp-text) !important;
    }}

    /* Sidebar Clean Styling */
    [data-testid="stSidebar"] {{
        background-color: var(--dp-sidebar) !important;
        border-right: 1px solid var(--dp-border-light) !important;
    }}
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {{
        color: var(--dp-text) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
    [data-testid="stSidebar"] caption {{
        color: var(--dp-muted) !important;
    }}
    [data-testid="stSidebar"] .stButton > button {{
        text-align: left !important;
        justify-content: flex-start !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
