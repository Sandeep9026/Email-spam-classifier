from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import load_artifact, preprocess_series


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "spam_classifier.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics_summary.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

SAMPLE_EMAILS = {
    "Bank phishing": (
        "Urgent notice: your account access will be suspended. Verify your banking "
        "details immediately to avoid service interruption."
    ),
    "Team update": (
        "Hi team, the client approved the draft proposal. Please review the final "
        "changes before tomorrow's presentation."
    ),
    "Promo scam": (
        "Congratulations! You have been selected for a limited-time cash reward. "
        "Click the secure link now to claim your bonus before midnight."
    ),
}

THEMES = {
    "Light": {
        "badge": "Light mode",
        "app_bg": "linear-gradient(180deg, #f6f7fb 0%, #eef2f9 100%)",
        "shell_bg": "rgba(255, 255, 255, 0.82)",
        "shell_border": "rgba(15, 23, 42, 0.08)",
        "panel_bg": "linear-gradient(180deg, rgba(255,255,255,0.96), rgba(246,248,252,0.96))",
        "panel_alt": "linear-gradient(180deg, #0f172a 0%, #16233a 100%)",
        "text": "#0f172a",
        "muted": "#5b6b85",
        "line": "rgba(15, 23, 42, 0.08)",
        "accent": "#1d4ed8",
        "accent_2": "#06b6d4",
        "accent_soft": "rgba(29, 78, 216, 0.08)",
        "positive": "#15803d",
        "negative": "#b91c1c",
        "theme_button_text": "#f8fafc",
        "theme_button_border": "rgba(255,255,255,0.16)",
        "input_bg": "#ffffff",
        "tab_bg": "rgba(15, 23, 42, 0.06)",
        "tab_active": "#0f172a",
        "tab_text_active": "#f8fafc",
        "button_bg": "linear-gradient(135deg, #1d4ed8, #06b6d4)",
        "button_hover": "linear-gradient(135deg, #1e40af, #0891b2)",
        "hero_title": "#eff6ff",
        "hero_text": "rgba(226, 232, 240, 0.9)",
        "progress": "#1d4ed8",
        "shadow": "0 22px 45px rgba(15, 23, 42, 0.12)",
    },
    "Dark": {
        "badge": "Dark mode",
        "app_bg": "linear-gradient(180deg, #020617 0%, #081224 100%)",
        "shell_bg": "rgba(4, 15, 30, 0.72)",
        "shell_border": "rgba(148, 163, 184, 0.14)",
        "panel_bg": "linear-gradient(180deg, rgba(8,15,30,0.98), rgba(11,20,38,0.96))",
        "panel_alt": "linear-gradient(180deg, #111827 0%, #050816 100%)",
        "text": "#e5eefc",
        "muted": "#97a6ba",
        "line": "rgba(148, 163, 184, 0.14)",
        "accent": "#38bdf8",
        "accent_2": "#818cf8",
        "accent_soft": "rgba(56, 189, 248, 0.10)",
        "positive": "#4ade80",
        "negative": "#fb7185",
        "theme_button_text": "#071120",
        "theme_button_border": "rgba(148, 163, 184, 0.18)",
        "input_bg": "#071120",
        "tab_bg": "rgba(255, 255, 255, 0.05)",
        "tab_active": "#38bdf8",
        "tab_text_active": "#08111f",
        "button_bg": "linear-gradient(135deg, #38bdf8, #818cf8)",
        "button_hover": "linear-gradient(135deg, #0ea5e9, #6366f1)",
        "hero_title": "#f8fbff",
        "hero_text": "rgba(203, 213, 225, 0.88)",
        "progress": "#38bdf8",
        "shadow": "0 24px 60px rgba(0, 0, 0, 0.35)",
    },
}


st.set_page_config(
    page_title="Spam Shield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run `python train.py` first.")
    return load_artifact(MODEL_PATH)


@st.cache_data
def load_metrics() -> tuple[pd.DataFrame | None, dict]:
    metrics_df = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else None
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    return metrics_df, metadata


def classify_messages(model, messages: list[str]) -> pd.DataFrame:
    cleaned_messages = preprocess_series(messages)
    probabilities = model.predict_proba(cleaned_messages)[:, 1]
    predictions = model.predict(cleaned_messages)
    return pd.DataFrame(
        {
            "email_text": messages,
            "prediction": ["Spam" if pred == 1 else "Not Spam" for pred in predictions],
            "spam_probability": probabilities,
            "confidence": [max(prob, 1 - prob) for prob in probabilities],
        }
    )


def probability_tier(probability: float) -> str:
    if probability >= 0.85:
        return "Critical"
    if probability >= 0.6:
        return "Elevated"
    if probability >= 0.35:
        return "Watch"
    return "Safe"


def metric_value(metrics_df: pd.DataFrame | None, metric_name: str) -> str:
    if metrics_df is None or metrics_df.empty:
        return "N/A"
    value = metrics_df.iloc[0].get(metric_name)
    return f"{value:.2%}" if isinstance(value, (int, float)) else "N/A"


def set_theme(theme_name: str) -> None:
    st.session_state["theme_mode"] = theme_name


def toggle_theme() -> None:
    current_theme = st.session_state.get("theme_mode", "Dark")
    st.session_state["theme_mode"] = "Light" if current_theme == "Dark" else "Dark"


def build_theme_css(theme_name: str) -> str:
    theme = THEMES[theme_name]
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    .stApp {{
        background: {theme["app_bg"]};
        color: {theme["text"]};
        font-family: "Outfit", sans-serif;
    }}

    .block-container {{
        max-width: 1280px;
        padding-top: 1.35rem;
        padding-bottom: 2rem;
    }}

    h1, h2, h3, h4, h5, p, label, span, li {{
        color: {theme["text"]};
        font-family: "Outfit", sans-serif !important;
    }}

    code, .mono {{
        font-family: "IBM Plex Mono", monospace !important;
    }}

    .app-shell {{
        border: 1px solid {theme["shell_border"]};
        background: {theme["shell_bg"]};
        backdrop-filter: blur(20px);
        border-radius: 30px;
        padding: 1rem;
        box-shadow: {theme["shadow"]};
    }}

    .topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1rem;
        padding: 0.45rem 0.55rem 0.9rem;
        border-bottom: 1px solid {theme["line"]};
    }}

    .brand-wrap {{
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }}

    .brand-mark {{
        width: 44px;
        height: 44px;
        border-radius: 14px;
        background: linear-gradient(135deg, {theme["accent"]}, {theme["accent_2"]});
        display: grid;
        place-items: center;
        font-size: 1.2rem;
        font-weight: 800;
        color: white;
        box-shadow: 0 12px 24px rgba(0,0,0,0.18);
    }}

    .brand-title {{
        font-size: 1.05rem;
        font-weight: 700;
        line-height: 1.05;
    }}

    .brand-subtitle {{
        color: {theme["muted"]};
        font-size: 0.84rem;
        margin-top: 0.15rem;
        font-family: "IBM Plex Mono", monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    .theme-status {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.55rem 0.8rem;
        border-radius: 999px;
        background: {theme["accent_soft"]};
        border: 1px solid {theme["line"]};
        color: {theme["text"]};
        font-size: 0.88rem;
        font-weight: 600;
    }}

    .topbar-right {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        flex-wrap: wrap;
    }}

    .hero-grid {{
        display: grid;
        grid-template-columns: 1.35fr 0.95fr;
        gap: 1rem;
        margin-bottom: 1rem;
    }}

    .hero-panel {{
        padding: 1.5rem;
        border-radius: 26px;
        background: {theme["panel_alt"]};
        border: 1px solid {theme["line"]};
        min-height: 280px;
        position: relative;
        overflow: hidden;
    }}

    .hero-panel::before {{
        content: "";
        position: absolute;
        inset: auto -40px -50px auto;
        width: 220px;
        height: 220px;
        border-radius: 999px;
        background: radial-gradient(circle, {theme["accent"]}33, transparent 68%);
        filter: blur(8px);
    }}

    .hero-kicker {{
        display: inline-block;
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.1);
        color: {theme["hero_text"]};
        font-size: 0.74rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-family: "IBM Plex Mono", monospace !important;
    }}

    .hero-title {{
        font-size: clamp(2.2rem, 4vw, 4.1rem);
        line-height: 0.9;
        margin: 0.9rem 0 0;
        color: {theme["hero_title"]};
        font-weight: 800;
        max-width: 8ch;
    }}

    .hero-copy {{
        margin-top: 1rem;
        color: {theme["hero_text"]};
        font-size: 1.02rem;
        max-width: 56ch;
        line-height: 1.55;
    }}

    .kpi-panel {{
        padding: 1rem;
        border-radius: 26px;
        background: {theme["panel_bg"]};
        border: 1px solid {theme["line"]};
        display: grid;
        gap: 0.8rem;
    }}

    .kpi-card {{
        border-radius: 20px;
        padding: 1rem;
        background: {theme["accent_soft"]};
        border: 1px solid {theme["line"]};
    }}

    .kpi-label {{
        color: {theme["muted"]};
        font-size: 0.74rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-family: "IBM Plex Mono", monospace !important;
    }}

    .kpi-value {{
        margin-top: 0.35rem;
        font-size: 1.5rem;
        font-weight: 800;
    }}

    .section-grid {{
        display: grid;
        grid-template-columns: 1.25fr 0.75fr;
        gap: 1rem;
        margin-bottom: 1rem;
    }}

    .panel {{
        padding: 1.2rem;
        border-radius: 24px;
        background: {theme["panel_bg"]};
        border: 1px solid {theme["line"]};
        box-shadow: {theme["shadow"]};
    }}

    .panel-title {{
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }}

    .panel-subtitle {{
        color: {theme["muted"]};
        margin-bottom: 0.9rem;
    }}

    .micro-label {{
        color: {theme["muted"]};
        font-size: 0.76rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-family: "IBM Plex Mono", monospace !important;
        margin-bottom: 0.55rem;
    }}

    .result-card {{
        margin-top: 0.9rem;
        border-radius: 22px;
        padding: 1.1rem;
        border: 1px solid {theme["line"]};
        background: {theme["accent_soft"]};
    }}

    .result-title {{
        font-size: 1.9rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }}

    .signal-chip {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.6rem;
        border-radius: 999px;
        background: {theme["tab_bg"]};
        border: 1px solid {theme["line"]};
        color: {theme["muted"]};
        font-size: 0.82rem;
        margin-top: 0.45rem;
    }}

    .intel-list {{
        display: grid;
        gap: 0.75rem;
    }}

    .intel-item {{
        padding: 0.95rem;
        border-radius: 18px;
        border: 1px solid {theme["line"]};
        background: {theme["tab_bg"]};
    }}

    .intel-item strong {{
        display: block;
        margin-bottom: 0.25rem;
    }}

    .intel-item span {{
        color: {theme["muted"]};
        line-height: 1.45;
    }}

    div[data-testid="stTextArea"] textarea {{
        border-radius: 18px !important;
        background: {theme["input_bg"]} !important;
        border: 1px solid {theme["line"]} !important;
        color: {theme["text"]} !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        box-shadow: none !important;
    }}

    div[data-testid="stTextArea"] textarea::placeholder {{
        color: {theme["muted"]} !important;
    }}

    div[data-baseweb="radio"] > div {{
        gap: 0.4rem;
    }}

    div[data-baseweb="radio"] label {{
        padding: 0.4rem 0.75rem;
        border-radius: 999px;
        background: {theme["tab_bg"]};
        border: 1px solid {theme["line"]};
    }}

    div[data-baseweb="radio"] label p {{
        color: {theme["text"]} !important;
        font-weight: 600 !important;
    }}

    div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
        gap: 0.55rem;
        background: transparent;
        padding: 0;
        border-bottom: 0;
    }}

    div[data-testid="stTabs"] button {{
        height: auto;
        padding: 0.75rem 1rem;
        border-radius: 16px;
        border: 1px solid {theme["line"]};
        background: {theme["tab_bg"]};
        color: {theme["text"]};
        font-weight: 700;
    }}

    div[data-testid="stTabs"] button[aria-selected="true"] {{
        background: {theme["tab_active"]};
        color: {theme["tab_text_active"]} !important;
        border-color: transparent;
    }}

    div[data-testid="stButton"] button {{
        border-radius: 14px !important;
        border: 1px solid transparent !important;
        background: {theme["button_bg"]} !important;
        color: white !important;
        font-weight: 700 !important;
        min-height: 44px;
    }}

    div[data-testid="stButton"] button:hover {{
        background: {theme["button_hover"]} !important;
    }}

    div[data-testid="stButton"] button[kind="secondary"] {{
        background: {theme["button_bg"]} !important;
        color: {theme["theme_button_text"]} !important;
        border: 1px solid {theme["theme_button_border"]} !important;
    }}

    div[data-testid="stButton"] button[kind="secondary"]:hover {{
        background: {theme["button_hover"]} !important;
    }}

    .ghost-btn-note {{
        color: {theme["muted"]};
        font-size: 0.83rem;
        margin-top: 0.45rem;
    }}

    [data-testid="stDataFrame"] {{
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid {theme["line"]};
    }}

    div[data-testid="stProgressBar"] > div > div {{
        background-color: {theme["progress"]};
    }}

    div[data-testid="stImage"] img {{
        border-radius: 18px;
        border: 1px solid {theme["line"]};
    }}

    @media (max-width: 980px) {{
        .hero-grid, .section-grid {{
            grid-template-columns: 1fr;
        }}
        .topbar {{
            flex-direction: column;
            align-items: flex-start;
        }}
    }}
    </style>
    """


def render_result_card(result: pd.Series, theme_name: str) -> None:
    probability = float(result["spam_probability"])
    label = result["prediction"]
    accent_label = "Threat detected" if label == "Spam" else "Message looks clean"
    st.markdown(
        f"""
        <div class="result-card">
            <div class="micro-label">{accent_label}</div>
            <div class="result-title">{label}</div>
            <div style="color:{THEMES[theme_name]["muted"]}; line-height:1.5;">
                Risk tier: <strong>{probability_tier(probability)}</strong><br/>
                Model confidence: <strong>{float(result["confidence"]):.2%}</strong>
            </div>
            <div class="signal-chip">Spam probability {probability:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(probability)


def render_example_buttons() -> None:
    st.markdown('<div class="micro-label">Quick launch samples</div>', unsafe_allow_html=True)
    sample_cols = st.columns(len(SAMPLE_EMAILS))
    for col, (label, sample_text) in zip(sample_cols, SAMPLE_EMAILS.items()):
        with col:
            if st.button(label, width="stretch", key=f"sample_{label}"):
                st.session_state["single_email_text"] = sample_text


if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Dark"

metrics_df, metadata = load_metrics()
class_distribution = metadata.get("class_distribution", {})
ham_count = class_distribution.get("ham", 0)
spam_count = class_distribution.get("spam", 0)

selected_theme = st.session_state["theme_mode"]
st.markdown(build_theme_css(selected_theme), unsafe_allow_html=True)
next_theme = "Light" if selected_theme == "Dark" else "Dark"

st.markdown(
    f"""
    <div class="app-shell">
        <div class="topbar">
            <div class="brand-wrap">
                <div class="brand-mark">S</div>
                <div>
                    <div class="brand-title">Spam Shield AI</div>
                    <div class="brand-subtitle">Message Risk Operations Console</div>
                </div>
            </div>
            <div class="topbar-right">
                <div class="theme-status">{THEMES[selected_theme]["badge"]} active</div>
            </div>
        </div>
    """,
    unsafe_allow_html=True,
)

theme_button_cols = st.columns([0.78, 0.22])
with theme_button_cols[1]:
    st.button(
        f"Switch To {next_theme}",
        width="stretch",
        key="theme_toggle",
        type="secondary",
        on_click=toggle_theme,
    )

st.markdown(
    f"""
    <div class="hero-grid">
        <div class="hero-panel">
            <div class="hero-kicker">Live classification console</div>
            <div class="hero-title">Spot bad messages before they hit the inbox.</div>
            <div class="hero-copy">
                This version uses the real UCI SMS Spam Collection dataset, a cleaner dark and light switch,
                and a more technical dashboard style for demos, interviews, and portfolio walkthroughs.
            </div>
        </div>
        <div class="kpi-panel">
            <div class="kpi-card">
                <div class="kpi-label">Best model</div>
                <div class="kpi-value">{metadata.get("best_model_name", "Not trained")}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Accuracy</div>
                <div class="kpi-value">{metric_value(metrics_df, "accuracy")}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Dataset profile</div>
                <div class="kpi-value">{ham_count} ham / {spam_count} spam</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = load_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

main_col, side_col = st.columns([1.28, 0.72], gap="large")

with main_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Classifier Workbench</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-subtitle">Switch modes, try sample threats, or paste a real message to score it instantly.</div>',
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Input mode",
        ["Single email", "Batch emails"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Single email":
        render_example_buttons()
        message = st.text_area(
            "Email text",
            key="single_email_text",
            placeholder="Paste an email or message body here...",
            height=220,
            label_visibility="collapsed",
        )
        if st.button("Scan Message", type="primary", width="stretch", key="scan_single"):
            if not message.strip():
                st.warning("Enter an email message before scanning.")
            else:
                result_df = classify_messages(model, [message])
                render_result_card(result_df.iloc[0], selected_theme)
    else:
        st.markdown('<div class="micro-label">Batch scan</div>', unsafe_allow_html=True)
        batch_text = st.text_area(
            "Batch email input",
            placeholder="Enter one message per line to run a mini batch screening.",
            height=240,
            label_visibility="collapsed",
        )
        if st.button("Scan Batch", type="primary", width="stretch", key="scan_batch"):
            messages = [line.strip() for line in batch_text.splitlines() if line.strip()]
            if not messages:
                st.warning("Enter at least one message to run the batch screen.")
            else:
                batch_results = classify_messages(model, messages)
                display_df = batch_results.rename(
                    columns={
                        "email_text": "Message",
                        "prediction": "Prediction",
                        "spam_probability": "Spam probability",
                        "confidence": "Confidence",
                    }
                )
                display_df["Risk tier"] = batch_results["spam_probability"].map(probability_tier)
                display_df["Spam probability"] = batch_results["spam_probability"].map(lambda value: f"{value:.2%}")
                display_df["Confidence"] = batch_results["confidence"].map(lambda value: f"{value:.2%}")
                st.dataframe(display_df, width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with side_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Operator Notes</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="intel-list">
            <div class="intel-item">
                <strong>Theme control fixed</strong>
                <span>The app now uses actual light and dark theme buttons instead of a radio-style selector.</span>
            </div>
            <div class="intel-item">
                <strong>Navigation colors corrected</strong>
                <span>Tabs and selection controls now inherit the active theme instead of fighting the background.</span>
            </div>
            <div class="intel-item">
                <strong>Fresh visual direction</strong>
                <span>This UI leans into a cleaner operations-console feel instead of repeating the same soft hero layout.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

overview_tab, metrics_tab, visuals_tab = st.tabs(["Briefing", "Metrics", "Visual Intel"])

with overview_tab:
    left, right = st.columns([1.0, 1.0], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Model Briefing</div>', unsafe_allow_html=True)
        st.write(
            "The project compares Naive Bayes, Logistic Regression, and SVM using TF-IDF features. "
            "The top model is persisted and reused directly in the Streamlit interface."
        )
        st.markdown(
            f"""
            - Best model: `{metadata.get("best_model_name", "N/A")}`
            - Total samples: `{metadata.get("total_samples", 0)}`
            - Precision: `{metric_value(metrics_df, "precision")}`
            - Recall: `{metric_value(metrics_df, "recall")}`
            - F1-score: `{metric_value(metrics_df, "f1")}`
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Why These Metrics Matter</div>', unsafe_allow_html=True)
        st.write(
            "High precision avoids burying good messages in spam. High recall keeps suspicious content from leaking into the inbox. "
            "A spam product that looks good on accuracy alone can still behave badly in real use."
        )
        st.markdown("</div>", unsafe_allow_html=True)

with metrics_tab:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Model Comparison Table</div>', unsafe_allow_html=True)
    if metrics_df is None:
        st.info("Run `python train.py` to generate metrics.")
    else:
        formatted = metrics_df.copy()
        for column in ["accuracy", "precision", "recall", "f1", "avg_spam_probability"]:
            if column in formatted.columns:
                formatted[column] = formatted[column].map(lambda value: f"{value:.2%}")
        st.dataframe(formatted, width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with visuals_tab:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Evaluation Charts</div>', unsafe_allow_html=True)
    figure_cols = st.columns(2, gap="large")
    figure_paths = [
        FIGURES_DIR / "model_comparison.png",
        FIGURES_DIR / "class_distribution.png",
        FIGURES_DIR / "confusion_matrix_naive_bayes.png",
        FIGURES_DIR / "confusion_matrix_logistic_regression.png",
        FIGURES_DIR / "confusion_matrix_svm.png",
    ]
    existing_paths = [path for path in figure_paths if path.exists()]
    if not existing_paths:
        st.info("Training visuals will appear here after running `python train.py`.")
    else:
        for index, path in enumerate(existing_paths):
            with figure_cols[index % 2]:
                st.image(str(path), caption=path.stem.replace("_", " ").title())
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
