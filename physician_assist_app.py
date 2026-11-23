"""
Emergency Physician Assist — Interactive Clinical Decision-Support & Symptom Mapping UI.

Built on top of the existing `ehr_chart_summary_agent` HL7 FHIR R4, NIH RxNorm, OpenFDA,
CDC 5-Digit ZIP Surveillance, Interactive 4-Track Clinical Swimlane, and Google ADK
Multi-Agent Symptom Mapping architecture.

Run with:
    .venv/bin/streamlit run physician_assist_app.py
"""

from __future__ import annotations

import html
import os
import re
import ssl
from datetime import datetime
from pathlib import Path
from typing import Any

import certifi
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

if not os.environ.get("SSL_CERT_FILE") and ssl.get_default_verify_paths().cafile is None:
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

for _env_candidate in (
    Path(__file__).resolve().parent / "ehr_chart_summary_agent" / ".env",
    Path(__file__).resolve().parent / ".env",
):
    if _env_candidate.exists():
        load_dotenv(_env_candidate, override=True)
        break

from ehr_chart_summary_agent.clinical_tools import (
    audit_patient_medications_rxnorm_openfda,
    check_drug_brand_or_generic,
    get_location_outbreak_and_disease_data,
    list_emergency_ward_patients,
)
from ehr_chart_summary_agent.ehr_chart_tools import (
    generate_clinical_swimlane_graph,
    get_hospitalizations_and_clinical_notes,
    get_longitudinal_labs_and_imaging,
    get_medication_reconciliation_history,
    get_open_loops_and_care_gaps,
    get_patient_demographics_and_triage,
)
from ehr_chart_summary_agent.sub_agents.symptom_mapper import run_adk_symptom_mapper
from ehr_chart_summary_agent.symptom_mapping_tools import (
    build_competing_hypotheses,
    compute_symptom_mappings,
)

# ---------------------------------------------------------------------------
# 1. Page Config & Session State Initialization
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Emergency Physician Assist",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "selected_patient_id" not in st.session_state:
    st.session_state.selected_patient_id = "pat-001"
if "patient_workflow_state" not in st.session_state:
    st.session_state.patient_workflow_state = {}
if "active_view" not in st.session_state:
    st.session_state.active_view = "roster"


def toggle_theme() -> None:
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"


IS_DARK: bool = st.session_state.theme == "dark"


def get_pstate(patient_id: str) -> dict[str, Any]:
    """Returns per-patient workflow progression, validation gate, and symptom override state."""
    store: dict[str, dict[str, Any]] = st.session_state.patient_workflow_state
    if patient_id not in store:
        store[patient_id] = {
            "unlocked_stage": 1,
            "active_stage": 1,
            "drug_gate_validated_no_match": False,
            "drug_gate_confirmed_match": False,
            "drug_gate_confirmed_timestamp": "",
            "drug_gate_note": "",
            "drug_gate_timestamp": "",
            "drug_decisions": {},       # med_name -> "Undecided" | "Ruled Out (No Match)" | "Matches Symptoms"
            "symptom_overrides": {},    # symptom_id -> "Auto (Agent)" | "Confirmed Match" | "Ruled Out (No Match)"
            "adk_ai_mapping_md": "",
            "adk_ai_stage": 0,
        }
    return store[patient_id]


def reset_pstate(patient_id: str) -> None:
    """Resets a patient's workflow progression, validation gate, and overrides back to Step 1."""
    st.session_state.patient_workflow_state.pop(patient_id, None)
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and (
            key.startswith(f"drug_dec_{patient_id}_")
            or key.startswith(f"sym_ov_{patient_id}_")
            or key == f"gate_note_input_{patient_id}"
        ):
            st.session_state.pop(key, None)
    get_pstate(patient_id)


def select_patient(patient_id: str) -> None:
    st.session_state.selected_patient_id = patient_id
    st.session_state.header_patient_select = patient_id
    st.session_state.active_view = "workspace"
    get_pstate(patient_id)


def _on_header_patient_change() -> None:
    chosen = st.session_state.get("header_patient_select")
    if chosen:
        select_patient(chosen)


# ---------------------------------------------------------------------------
# 2. Formatting Helpers & Cached Clinical Data Loaders
# ---------------------------------------------------------------------------
def fmt_date(raw: Any) -> str:
    """Formats YYYY-MM-DD strings into readable 'MMM dd, yyyy' clinical dates."""
    if not raw:
        return "—"
    s = str(raw).strip()[:10]
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.strftime("%b %d, %Y")
    except ValueError:
        return esc(raw)


def esc(val: Any) -> str:
    """HTML-escapes text and collapses embedded newlines so CommonMark HTML blocks never split."""
    normalized = re.sub(r"[\r\n]+", " ", str(val if val is not None else "")).strip()
    return html.escape(normalized)


def render_html(raw_html: str) -> None:
    """Renders custom HTML as a single unindented block so Markdown never leaks literal </div> tags."""
    compact = re.sub(r"\n\s*", " ", raw_html).strip()
    st.markdown(compact, unsafe_allow_html=True)


def clean_agent_md(md_text: str) -> str:
    """Strips stray HTML wrapper tags (e.g., <div>, </div>, <span>) from live LLM Markdown outputs."""
    if not md_text:
        return ""
    cleaned = re.sub(r"<br\s*/?>", "\n", str(md_text), flags=re.IGNORECASE)
    cleaned = re.sub(r"</?(?:div|span|section|article)\b[^>]*>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


@st.cache_data(ttl=180, show_spinner=False)
def cached_ward_roster() -> dict[str, Any]:
    return list_emergency_ward_patients()


@st.cache_data(ttl=180, show_spinner=False)
def cached_longitudinal_bundle(patient_id: str) -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        f_intake = pool.submit(get_patient_demographics_and_triage, patient_id)
        f_labs = pool.submit(get_longitudinal_labs_and_imaging, patient_id)
        f_notes = pool.submit(get_hospitalizations_and_clinical_notes, patient_id)
        f_loops = pool.submit(get_open_loops_and_care_gaps, patient_id)
        return {
            "intake": f_intake.result(),
            "labs": f_labs.result(),
            "notes": f_notes.result(),
            "loops": f_loops.result(),
        }


@st.cache_data(ttl=180, show_spinner=False)
def cached_swimlane(patient_id: str) -> dict[str, Any]:
    res = generate_clinical_swimlane_graph(patient_id)
    html_content = ""
    html_path = res.get("interactive_html_path", "")
    if html_path and Path(html_path).exists():
        html_content = Path(html_path).read_text(encoding="utf-8")
    return {**res, "html_content": html_content}


@st.cache_data(ttl=300, show_spinner=False)
def cached_drug_audit(patient_id: str) -> dict[str, Any]:
    med_hist = get_medication_reconciliation_history(patient_id)
    batch = audit_patient_medications_rxnorm_openfda(patient_id)
    rx_by_name: dict[str, dict[str, Any]] = {}
    for m in batch.get("active_prescriptions_audited", []):
        rx_by_name[m["medication"]] = m.get("rxnorm_and_openfda_audit", {})
    for h in batch.get("held_or_discontinued_medications_audited", []):
        rx_by_name[h["medication"]] = h.get("rxnorm_and_openfda_audit", {})
    return {
        "med_hist": med_hist,
        "rx_by_name": rx_by_name,
        "duplicate_alerts": batch.get("duplicate_active_ingredient_alerts", []),
    }


@st.cache_data(ttl=300, show_spinner=False)
def cached_zip_outbreaks(postal_code: str, city: str = "") -> dict[str, Any]:
    return get_location_outbreak_and_disease_data(postal_code=postal_code, city=city)


# ---------------------------------------------------------------------------
# 3. SaaS Zinc CSS Design System (DM Sans + JetBrains Mono, Minimal Chrome)
# ---------------------------------------------------------------------------
BG = "#09090b" if IS_DARK else "#ffffff"
BG_SUBTLE = "#0c0c0f" if IS_DARK else "#f8fafc"
CARD = "#0c0c0f" if IS_DARK else "#ffffff"
CARD_HOVER = "#141419" if IS_DARK else "#f4f4f5"
BORDER = "#1e1e24" if IS_DARK else "#e4e4e7"
BORDER_SUBTLE = "#16161a" if IS_DARK else "#f0f0f2"
TEXT = "#fafafa" if IS_DARK else "#09090b"
TEXT_MUTED = "#a1a1aa" if IS_DARK else "#71717a"
TEXT_DIM = "#71717a" if IS_DARK else "#a1a1aa"
GREEN = "#22c55e" if IS_DARK else "#16a34a"
GREEN_MUTED = "rgba(34,197,94,0.14)" if IS_DARK else "rgba(22,163,74,0.09)"
RED = "#ef4444" if IS_DARK else "#dc2626"
RED_MUTED = "rgba(239,68,68,0.14)" if IS_DARK else "rgba(220,38,38,0.09)"
AMBER = "#f59e0b" if IS_DARK else "#d97706"
AMBER_MUTED = "rgba(245,158,11,0.14)" if IS_DARK else "rgba(217,119,6,0.09)"
BLUE_MUTED = "rgba(37,99,235,0.16)" if IS_DARK else "rgba(37,99,235,0.09)"
SHADOW = "none" if IS_DARK else "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
    --bg: {BG};
    --bg-subtle: {BG_SUBTLE};
    --card: {CARD};
    --card-hover: {CARD_HOVER};
    --border: {BORDER};
    --border-subtle: {BORDER_SUBTLE};
    --text: {TEXT};
    --text-muted: {TEXT_MUTED};
    --text-dim: {TEXT_DIM};
    --accent: #2563eb;
    --accent-muted: #1d4ed8;
    --green: {GREEN};
    --green-muted: {GREEN_MUTED};
    --red: {RED};
    --red-muted: {RED_MUTED};
    --amber: {AMBER};
    --amber-muted: {AMBER_MUTED};
    --blue-muted: {BLUE_MUTED};
    --shadow: {SHADOW};
    --radius: 10px;
}}

header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
div[data-testid="stSidebarCollapsedControl"] {{
    display: none !important;
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

.block-container {{
    padding: 1.4rem 2.2rem 3rem !important;
    max-width: 1540px !important;
}}

[data-testid="stHorizontalBlock"] {{
    gap: 1.15rem !important;
}}

/* Dropdowns & Popovers z-index safety */
div[data-baseweb="select"], div[data-baseweb="popover"] {{
    position: relative;
    z-index: 30;
}}

/* Pill Tabs */
button[data-baseweb="tab"] {{
    background: transparent !important;
    color: var(--text-muted) !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    padding: 0.52rem 1rem !important;
    border: 1px solid transparent !important;
    border-radius: 7px !important;
    transition: all 0.18s ease !important;
}}
button[data-baseweb="tab"]:hover {{
    color: var(--text) !important;
    background: var(--card-hover) !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--text) !important;
    background: var(--card) !important;
    border-color: var(--border) !important;
    box-shadow: var(--shadow);
}}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
    display: none !important;
}}
[data-baseweb="tab-list"] {{
    gap: 4px !important;
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    margin-bottom: 1rem !important;
}}

/* Buttons & Popover Triggers */
.stButton button, .stPopover button {{
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    border: 1px solid var(--border) !important;
    background: var(--card) !important;
    color: var(--text) !important;
    transition: all 0.18s ease !important;
}}
.stButton button:hover:not(:disabled), .stPopover button:hover:not(:disabled) {{
    background: var(--card-hover) !important;
    border-color: var(--accent) !important;
    transform: translateY(-1px);
}}
.stButton button[kind="primary"], button[data-testid="stBaseButton-primary"] {{
    background: var(--accent) !important;
    color: #ffffff !important;
    border-color: var(--accent) !important;
}}
.stButton button[kind="primary"]:hover:not(:disabled), button[data-testid="stBaseButton-primary"]:hover:not(:disabled) {{
    background: var(--accent-muted) !important;
    color: #ffffff !important;
}}
.stButton button:disabled {{
    opacity: 0.45 !important;
    cursor: not-allowed !important;
}}

/* Cards, Badges & Tables */
.epa-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.35rem;
    box-shadow: var(--shadow);
    margin-bottom: 0.9rem;
    transition: border-color 0.18s ease, background-color 0.18s ease;
}}
.epa-card:hover {{
    border-color: rgba(37,99,235,0.45);
}}

.metric-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.25rem;
    box-shadow: var(--shadow);
    transition: transform 0.18s ease, border-color 0.18s ease;
    height: 100%;
}}
.metric-card:hover {{
    border-color: rgba(37,99,235,0.45);
    transform: translateY(-1px);
}}
.metric-label {{
    font-size: 0.75rem;
    color: var(--text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
.metric-value {{
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.03em;
    margin-top: 0.2rem;
    font-family: 'JetBrains Mono', monospace;
}}
.metric-delta {{
    font-size: 0.72rem;
    font-weight: 500;
    margin-top: 0.45rem;
    padding: 2px 8px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}}
.delta-up {{ color: var(--green); background: var(--green-muted); }}
.delta-down {{ color: var(--red); background: var(--red-muted); }}
.delta-warn {{ color: var(--amber); background: var(--amber-muted); }}
.delta-info {{ color: var(--accent); background: var(--blue-muted); }}

.chart-wrap {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.15rem 1.25rem 0.5rem;
    box-shadow: var(--shadow);
    margin-bottom: 0.95rem;
}}
.chart-title {{ font-size: 0.88rem; font-weight: 600; color: var(--text); }}
.chart-subtitle {{ font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.6rem; }}

.data-table-wrap {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow-x: auto;
    box-shadow: var(--shadow);
    margin-bottom: 0.95rem;
}}
.data-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.81rem;
}}
.data-table th {{
    text-align: left;
    padding: 0.68rem 0.9rem;
    color: var(--text-muted);
    font-weight: 600;
    font-size: 0.71rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: var(--bg-subtle);
    border-bottom: 1px solid var(--border);
}}
.data-table td {{
    padding: 0.7rem 0.9rem;
    color: var(--text);
    border-bottom: 1px solid var(--border-subtle);
    vertical-align: top;
}}
.data-table tbody tr {{
    transition: background-color 0.15s ease;
}}
.data-table tbody tr:hover {{
    background-color: var(--card-hover);
}}
.data-table tr:last-child td {{ border-bottom: none; }}

.badge {{
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.71rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    white-space: nowrap;
}}
.badge-green {{ color: var(--green); background: var(--green-muted); }}
.badge-red {{ color: var(--red); background: var(--red-muted); }}
.badge-amber {{ color: var(--amber); background: var(--amber-muted); }}
.badge-blue {{ color: #3b82f6; background: var(--blue-muted); }}
.badge-mono {{
    display: inline-block;
    white-space: nowrap;
    margin: 1px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    padding: 2px 7px;
    border-radius: 5px;
}}

.stepper-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.65rem;
    margin-bottom: 1rem;
}}
@media (max-width: 980px) {{
    .stepper-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}
.step-pill {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 0.72rem 0.9rem;
    transition: all 0.18s ease;
}}
.step-pill.active {{
    border-color: #2563eb;
    background: var(--blue-muted);
}}
.step-pill.locked {{
    opacity: 0.52;
}}
.step-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.69rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
}}
.step-name {{
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text);
    margin-top: 2px;
}}

.symptom-panel-shell {{
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.95rem 1.05rem;
    margin-bottom: 0.65rem;
}}
.hyp-coverage-track {{
    width: 100%;
    height: 8px;
    background: var(--border);
    border-radius: 999px;
    overflow: hidden;
    margin-top: 8px;
}}
.hyp-coverage-fill {{
    height: 100%;
    border-radius: 999px;
    transition: width 0.25s ease;
}}
.hyp-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 4px solid #2563eb;
    border-radius: 10px;
    padding: 0.88rem 0.98rem;
    margin-bottom: 0.62rem;
    box-shadow: var(--shadow);
    transition: border-color 0.18s ease, background-color 0.18s ease;
}}
.hyp-card-strong {{ border-left-color: var(--green); }}
.hyp-card-partial {{ border-left-color: var(--amber); }}
.hyp-card-outbreak {{ border-left-color: var(--red); }}
.hyp-card-baseline {{ border-left-color: #2563eb; }}
.hyp-card-locked {{
    border-left-color: var(--border);
    background: var(--bg-subtle);
    opacity: 0.78;
}}
.hyp-chip-wrap {{
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 5px;
}}
.chip-explain {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.71rem;
    font-weight: 600;
    color: var(--green);
    background: var(--green-muted);
    border: 1px solid rgba(34,197,94,0.25);
    cursor: help;
    white-space: nowrap;
}}
.chip-resolved {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.71rem;
    font-weight: 700;
    color: #2563eb;
    background: var(--blue-muted);
    border: 1px solid rgba(37,99,235,0.35);
    cursor: help;
    white-space: nowrap;
}}
.chip-gap {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.71rem;
    font-weight: 600;
    color: var(--red);
    background: var(--red-muted);
    border: 1px solid rgba(239,68,68,0.25);
    white-space: nowrap;
}}
.chip-gap-muted {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.71rem;
    font-weight: 500;
    color: var(--text-muted);
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    white-space: nowrap;
}}
</style>
""",
    unsafe_allow_html=True,
)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        family="DM Sans, sans-serif",
        color="#a1a1aa" if IS_DARK else "#52525b",
        size=11,
    ),
    margin=dict(l=10, r=14, t=18, b=10),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)" if IS_DARK else "rgba(0,0,0,0.05)",
        zerolinecolor="rgba(255,255,255,0.06)" if IS_DARK else "rgba(0,0,0,0.06)",
        tickfont=dict(size=10, color="#a1a1aa" if IS_DARK else "#71717a"),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)" if IS_DARK else "rgba(0,0,0,0.05)",
        zerolinecolor="rgba(255,255,255,0.06)" if IS_DARK else "rgba(0,0,0,0.06)",
        tickfont=dict(size=10, color="#a1a1aa" if IS_DARK else "#71717a"),
    ),
    hoverlabel=dict(
        bgcolor="#18181b" if IS_DARK else "#ffffff",
        bordercolor="#27272a" if IS_DARK else "#e4e4e7",
        font=dict(color="#fafafa" if IS_DARK else "#09090b", family="DM Sans, sans-serif", size=12),
    ),
)


def metric_card(label: str, value: str, delta: str | None = None, delta_type: str = "info") -> None:
    cls = f"delta-{delta_type}"
    delta_html = f'<div class="metric-delta {cls}">{esc(delta)}</div>' if delta else ""
    st.markdown(
        (
            '<div class="metric-card">'
            f'<div class="metric-label">{esc(label)}</div>'
            f'<div class="metric-value">{esc(value)}</div>'
            f'{delta_html}'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def flag_badge(flag: str) -> str:
    f = (flag or "N").upper()
    if f in ("HH", "LL", "CRITICAL"):
        return f'<span class="badge badge-red">{esc(f)}</span>'
    if f in ("H", "L", "HIGH", "MEDIUM"):
        return f'<span class="badge badge-amber">{esc(f)}</span>'
    return f'<span class="badge badge-green">{esc(f)}</span>'


# ---------------------------------------------------------------------------
# 4. Load Roster & Render Brand Header + Quick Patient Selector
# ---------------------------------------------------------------------------
roster_data = cached_ward_roster()
ward_patients: list[dict[str, Any]] = roster_data.get("patients", [])
fhir_source_label = roster_data.get("fhir_data_source", "GCP Cloud Healthcare FHIR Store")

patient_options = {
    p["patient_id"]: f"{p['patient_id']} · {p['full_name']} ({p['city']}, ZIP {p['postal_code']})"
    for p in ward_patients
}
if st.session_state.selected_patient_id not in patient_options and patient_options:
    st.session_state.selected_patient_id = next(iter(patient_options))

PUBLIC_API_REFERENCES: list[dict[str, str]] = [
    {
        "stage": "Step 3 · Pharmacology",
        "name": "NIH NLM RxNorm REST API",
        "endpoint": "https://rxnav.nlm.nih.gov/REST/",
        "docs_url": "https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html",
        "purpose": "Normalizes RxCUI codes, classifies Generic (IN/SCD) vs. Brand (BN/SBD), and detects duplicate active ingredients.",
    },
    {
        "stage": "Step 3 · Pharmacology",
        "name": "OpenFDA Drug Labeling (SPL) & DailyMed",
        "endpoint": "https://api.fda.gov/drug/label.json",
        "docs_url": "https://open.fda.gov/apis/drug/label/",
        "purpose": "Retrieves official clinical indications, FDA Boxed Warnings, contraindications, and labeled adverse reactions.",
    },
    {
        "stage": "Step 3 · Pharmacology",
        "name": "OpenFDA FAERS Adverse Events API",
        "endpoint": "https://api.fda.gov/drug/event.json",
        "docs_url": "https://open.fda.gov/apis/drug/event/",
        "purpose": "Queries post-marketing adverse event report counts across prescribed active ingredients.",
    },
    {
        "stage": "Step 4 · ZIP Surveillance",
        "name": "CDC NSSP Local Hospital ED Trajectories",
        "endpoint": "https://data.cdc.gov/resource/rdmq-nq56.json",
        "docs_url": "https://data.cdc.gov/d/rdmq-nq56",
        "purpose": "Weekly local Emergency Department respiratory visit % and outbreak trends (Influenza, COVID-19, RSV).",
    },
    {
        "stage": "Step 4 · ZIP Surveillance",
        "name": "CDC NWSS Sewershed Wastewater PCR",
        "endpoint": "https://data.cdc.gov/resource/ymmh-divb.json",
        "docs_url": "https://www.cdc.gov/nwss/index.html",
        "purpose": "Local sewershed viral PCR detections & surges (Flu A, SARS-CoV-2, RSV, Measles, Mpox on data.cdc.gov).",
    },
    {
        "stage": "Step 4 · ZIP Surveillance",
        "name": "CDC PLACES 5-Digit ZCTA5 Prevalence",
        "endpoint": "https://data.cdc.gov/resource/qnzd-25i4.json",
        "docs_url": "https://data.cdc.gov/d/qnzd-25i4",
        "purpose": "Chronic/endemic disease prevalence and population strictly scoped to the patient's 5-digit ZIP code.",
    },
    {
        "stage": "Step 4 · ZIP Surveillance",
        "name": "Open-Meteo Air Quality & Zippopotam.us",
        "endpoint": "https://air-quality-api.open-meteo.com/v1/air-quality",
        "docs_url": "https://open-meteo.com/en/docs/air-quality-api",
        "purpose": "Hyperlocal US AQI, PM2.5, PM10, Ozone, and NO2 respiratory triggers at the patient's ZIP coordinates.",
    },
]


hdr_left, hdr_mid, hdr_apis, hdr_right = st.columns([4.7, 3.1, 1.25, 0.95], vertical_alignment="center")
with hdr_left:
    render_html(
        f"""
        <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
            <div style="width:34px;height:34px;border-radius:9px;background:#2563eb;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;font-size:1rem;">◆</div>
            <div>
                <div style="font-size:1.25rem;font-weight:700;letter-spacing:-0.02em;color:var(--text);">
                    Emergency Physician Assist
                </div>
                <div style="font-size:0.76rem;color:var(--text-muted);">
                    Longitudinal EHR · 4-Track Swimlane · Drug &amp; Adverse-Effect Gate · 5-Digit ZIP Outbreaks · ADK Symptom Mapper
                </div>
            </div>
            <span class="badge badge-blue" style="margin-left:4px;">{esc(fhir_source_label)}</span>
        </div>
        """
    )

with hdr_mid:
    if (
        "header_patient_select" not in st.session_state
        or st.session_state.header_patient_select not in patient_options
    ):
        st.session_state.header_patient_select = st.session_state.selected_patient_id
    st.selectbox(
        "Active Emergency Ward Patient",
        options=list(patient_options.keys()),
        format_func=lambda k: patient_options[k],
        label_visibility="collapsed",
        key="header_patient_select",
        on_change=_on_header_patient_change,
    )

with hdr_apis:
    with st.popover("🔗 Public APIs", width="stretch"):
        st.markdown("**Live Public APIs & Clinical Data Sources**")
        st.caption(
            f"EHR data is pulled from **{fhir_source_label}** (cited by FHIR resource ID). "
            "External pharmacology & neighborhood surveillance pull live from:"
        )
        for ref in PUBLIC_API_REFERENCES:
            st.markdown(
                f"- **[{ref['name']}]({ref['docs_url']})** *({ref['stage']})*  \n"
                f"  `{ref['endpoint']}` — {ref['purpose']}"
            )

with hdr_right:
    theme_btn_label = "☀️ Light" if IS_DARK else "🌙 Dark"
    st.button(theme_btn_label, on_click=toggle_theme, width="stretch", key="theme_toggle_btn")

render_html('<div style="height:0.35rem;"></div>')

active_pid: str = st.session_state.selected_patient_id
pstate = get_pstate(active_pid)

# Preload Stage 1 longitudinal data for the selected patient
long_data = cached_longitudinal_bundle(active_pid)
intake_data = long_data["intake"]
labs_data = long_data["labs"]
notes_data = long_data["notes"]
loops_data = long_data["loops"]

demo = intake_data.get("demographics", {})
visit = intake_data.get("todays_visit", {})
vitals = visit.get("triage_vitals_today", {})
postal_code = demo.get("postal_code", "")

# Preload cached Stage 3 / Stage 4 results ONLY when unlocked so earlier stages stay fast and gated
drug_bundle: dict[str, Any] | None = None
outbreak_bundle: dict[str, Any] | None = None
if pstate["unlocked_stage"] >= 3:
    with st.spinner("Querying NIH RxNorm & OpenFDA Drug Adverse Events..."):
        drug_bundle = cached_drug_audit(active_pid)
if pstate["unlocked_stage"] >= 4 and postal_code:
    with st.spinner(f"Querying hyperlocal 5-digit ZIP {postal_code} outbreak surveillance..."):
        outbreak_bundle = cached_zip_outbreaks(postal_code, demo.get("city", ""))

symptom_map_data = compute_symptom_mappings(
    patient_id=active_pid,
    unlocked_stage=pstate["unlocked_stage"],
    precomputed_drug_checks=drug_bundle["rx_by_name"] if drug_bundle else None,
    precomputed_outbreaks=outbreak_bundle,
)


# ---------------------------------------------------------------------------
# 5. Competing Hypotheses Scorecard Renderer (Right Panel + Dedicated View 3)
# ---------------------------------------------------------------------------
def _truncate_text(raw: str, limit: int = 68) -> str:
    s = str(raw or "").strip()
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def build_hypothesis_card_html(
    card: dict[str, Any],
    rank_num: int | None = None,
) -> str:
    """Renders a single Root-Cause Hypothesis Card with ✓ Explains and ✗ Cannot Explain chips."""
    cid = card["id"]
    state = card["state"]
    fit_level = card["fit_level"]
    badge_txt = card["fit_badge"]

    # Reflect bedside physician confirmation / rule-out state directly on the Drug card
    if cid == "drug_ae" and state == "ACTIVE":
        if pstate.get("drug_gate_confirmed_match") and not pstate.get("drug_gate_validated_no_match"):
            badge_cls = "badge-green"
            badge_txt = f"✓ Confirmed · {card['explained_count']}/{card['total_count']}"
            card_mod = "hyp-card-strong"
        elif pstate.get("drug_gate_validated_no_match"):
            badge_cls = "badge-amber"
            badge_txt = f"Ruled Out Sole Cause · {card['explained_count']}/{card['total_count']}"
            card_mod = "hyp-card-partial"
        elif fit_level == "STRONG":
            badge_cls = "badge-green"
            card_mod = "hyp-card-strong"
        elif fit_level == "PARTIAL":
            badge_cls = "badge-amber"
            card_mod = "hyp-card-partial"
        else:
            badge_cls = "badge-red"
            card_mod = "hyp-card-partial"
    elif state in ("LOCKED", "GATED"):
        badge_cls = "badge-mono"
        card_mod = "hyp-card-locked"
    elif fit_level == "RESOLVES_GAPS":
        badge_cls = "badge-red"
        card_mod = "hyp-card-outbreak"
    elif fit_level == "STRONG":
        badge_cls = "badge-green"
        card_mod = "hyp-card-strong"
    elif cid == "ehr_baseline":
        badge_cls = "badge-blue"
        card_mod = "hyp-card-baseline"
    elif fit_level == "PARTIAL":
        badge_cls = "badge-amber"
        card_mod = "hyp-card-partial"
    else:
        badge_cls = "badge-mono"
        card_mod = "hyp-card-baseline"

    rank_prefix = (
        f'<span class="badge-mono" style="margin-right:6px;">#{rank_num}</span>'
        if rank_num is not None
        else ""
    )
    badge_span = (
        f'<span class="{badge_cls}">{esc(badge_txt)}</span>'
        if badge_cls == "badge-mono"
        else f'<span class="badge {badge_cls}">{esc(badge_txt)}</span>'
    )
    culprit_full = card.get("culprit_summary", "")
    culprit_disp = _truncate_text(culprit_full, 72)

    if state == "LOCKED":
        inner_html = (
            '<div style="font-size:0.74rem;color:var(--text-muted);margin-top:6px;line-height:1.45;">'
            'Audits prescribed, held &amp; OTC drugs via NIH RxNorm &amp; OpenFDA in <b>Step 3</b> to test whether medication toxicity or non-adherence explains this presentation.'
            '</div>'
        )
    elif state == "GATED":
        inner_html = (
            '<div style="font-size:0.74rem;color:var(--text-muted);margin-top:6px;line-height:1.45;">'
            f'Gated behind <b>Step 3 Medication Check</b> — unlocks hyperlocal <b>ZIP {esc(postal_code)}</b> ED, wastewater PCR &amp; AQI correlation if drugs leave residual symptoms.'
            '</div>'
        )
    else:
        explains = card.get("explains", [])
        gaps = card.get("cannot_explain", [])

        if explains:
            exp_chips = "".join(
                (
                    f'<span class="chip-resolved" title="{esc("Resolves Drug Gap — " + item["detail"] + " [" + item["citation"] + "]")}">'
                    f'★ {esc(item["short_label"])}</span>'
                    if item.get("resolves_drug_gap")
                    else f'<span class="chip-explain" title="{esc(item["detail"] + " [" + item["citation"] + "]")}">'
                    f'✓ {esc(item["short_label"])}</span>'
                )
                for item in explains
            )
            exp_block = (
                f'<div style="font-size:0.7rem;font-weight:700;color:var(--text-muted);margin-top:7px;text-transform:uppercase;letter-spacing:0.03em;">'
                f'Explains ({len(explains)}/{card["total_count"]})'
                f'</div>'
                f'<div class="hyp-chip-wrap">{exp_chips}</div>'
            )
        else:
            exp_block = (
                '<div style="font-size:0.74rem;color:var(--text-muted);margin-top:7px;">'
                'Does not match any extracted presenting findings.'
                '</div>'
            )

        if not gaps and card["total_count"] > 0:
            gap_block = (
                '<div style="font-size:0.73rem;font-weight:600;color:var(--green);margin-top:7px;">'
                '✓ Accounts for 100% of presenting symptoms, vitals &amp; acute labs'
                '</div>'
            )
        elif gaps:
            is_zip_card = cid == "zip_outbreak"
            gap_hdr = "Non-Infectious / Baseline Signs" if is_zip_card else f"Cannot Explain ({len(gaps)})"
            gap_chips = "".join(
                (
                    f'<span class="chip-gap-muted" title="{esc(g["label"])}">— {esc(g["short_label"])}</span>'
                    if is_zip_card
                    else f'<span class="chip-gap" title="{esc(g["label"])}">✗ {esc(g["short_label"])}</span>'
                )
                for g in gaps
            )
            gap_block = (
                f'<div style="font-size:0.7rem;font-weight:700;color:var(--text-muted);margin-top:7px;text-transform:uppercase;letter-spacing:0.03em;">'
                f'{esc(gap_hdr)}'
                f'</div>'
                f'<div class="hyp-chip-wrap">{gap_chips}</div>'
            )
        else:
            gap_block = ""

        inner_html = exp_block + gap_block

    return (
        f'<div class="hyp-card {card_mod}">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:wrap;">'
        f'<div style="font-size:0.73rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.03em;">'
        f'{rank_prefix}{esc(card["domain_title"])}'
        f'</div>'
        f'{badge_span}'
        f'</div>'
        f'<div style="font-size:0.86rem;font-weight:700;color:var(--text);margin-top:4px;" title="{esc(culprit_full)}">'
        f'{esc(culprit_disp)}'
        f'</div>'
        f'<div style="font-size:0.7rem;color:var(--text-muted);margin-top:1px;">{esc(card["source_tag"])}</div>'
        f'{inner_html}'
        f'</div>'
    )


def _get_live_competing_hypotheses() -> tuple[dict[str, Any], set[str]]:
    """Computes competing hypotheses dynamically incorporating any ruled-out drugs in Step 3."""
    ruled_out_meds = {
        med
        for med, dec in pstate["drug_decisions"].items()
        if dec in ("Ruled Out (No Match)", "Rule Out (No Match)")
    }
    competing = build_competing_hypotheses(
        mapped_symptoms=symptom_map_data.get("mapped_symptoms", []),
        unlocked_stage=pstate["unlocked_stage"],
        postal_code=postal_code,
        ruled_out_medications=ruled_out_meds,
    )
    return competing, ruled_out_meds


def render_symptom_mapping_ui(compact_side_panel: bool = True) -> None:
    """Renders the Option 1 Competing Hypotheses Scorecard UI with 1-click bedside actions."""
    mapped_list: list[dict[str, Any]] = symptom_map_data.get("mapped_symptoms", [])
    unlocked = pstate["unlocked_stage"]
    competing, ruled_out_meds = _get_live_competing_hypotheses()

    total_f = competing["total_findings"]
    overall_exp = competing["overall_explained"]
    cov_pct = competing["coverage_pct"]
    drug_exp = competing["drug_explained_count"]
    drug_gaps_cnt = competing["drug_unexplained_count"]
    resolves_cnt = competing["resolves_drug_gaps_count"]

    scope_labels = {
        1: "Stage 1 · EHR Baseline",
        2: "Stage 2 · EHR + Swimlane",
        3: "Stage 3 · Medication Gate",
        4: f"Stage 4 · Full + ZIP {postal_code}",
    }

    # Determine coverage bar color & concise clinical status takeaway
    if unlocked == 3:
        bar_pct = competing["drug_coverage_pct"]
        if drug_gaps_cnt == 0 and total_f > 0:
            bar_color = "var(--green)"
            status_line = f"✓ Medications account for all {drug_exp}/{total_f} acute findings"
        else:
            bar_color = "var(--amber)"
            gap_preview = ", ".join(competing["drug_unexplained_chips"][:3])
            status_line = f"⚠ Meds explain {drug_exp}/{total_f} · Leaves {drug_gaps_cnt} unexplained ({gap_preview})"
        cov_headline = f"Medication Fit: {drug_exp}/{total_f} ({bar_pct}%)"
    elif unlocked >= 4:
        bar_pct = cov_pct
        bar_color = "var(--green)" if overall_exp == total_f else "#2563eb"
        if resolves_cnt > 0:
            status_line = f"★ ZIP {postal_code} outbreak data resolves {resolves_cnt} finding(s) unexplained by medications"
        else:
            status_line = f"✓ {overall_exp}/{total_f} findings mapped across EHR, Medications & ZIP {postal_code}"
        cov_headline = f"Diagnostic Coverage: {overall_exp}/{total_f} ({cov_pct}%)"
    else:
        bar_pct = cov_pct
        bar_color = "#2563eb"
        unm_cnt = competing["overall_unexplained_count"]
        status_line = (
            f"{overall_exp}/{total_f} matched to baseline EHR · Audit medications in Step 3 to test acute cause"
            if unm_cnt == 0
            else f"{overall_exp}/{total_f} in baseline EHR · {unm_cnt} acute sign(s) need Step 3–4 check"
        )
        cov_headline = f"Baseline Coverage: {overall_exp}/{total_f} ({cov_pct}%)"

    render_html(
        f"""
        <div class="symptom-panel-shell">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;">
                <div>
                    <div style="font-size:0.95rem;font-weight:700;color:var(--text);">
                        Competing Hypotheses Scorecard
                    </div>
                    <div style="font-size:0.73rem;color:var(--text-muted);margin-top:1px;">
                        Root-cause fit across clinical stages · Hover <span style="color:var(--green);font-weight:600;">✓</span> chips for proof
                    </div>
                </div>
                <span class="badge badge-blue">{esc(scope_labels.get(unlocked, "Stage 1"))}</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-top:10px;font-size:0.76rem;font-weight:700;color:var(--text);">
                <span>{esc(cov_headline)}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--text-muted);">{total_f} Extracted Signs</span>
            </div>
            <div class="hyp-coverage-track">
                <div class="hyp-coverage-fill" style="width:{max(4, bar_pct)}%;background:{bar_color};"></div>
            </div>
            <div style="font-size:0.73rem;color:var(--text-muted);margin-top:6px;line-height:1.4;">
                {esc(status_line)}
            </div>
        </div>
        """
    )

    # Render Ranked Root-Cause Hypothesis Cards (2-3 cards instead of 10+ symptom dropdown cards)
    btn_suffix = "side" if compact_side_panel else "full"
    for idx, hcard in enumerate(competing["hypotheses"], start=1):
        render_html(build_hypothesis_card_html(hcard, rank_num=idx))

        # Contextual 1-Click Bedside Action Buttons directly beneath the Drug Hypothesis Card
        if hcard["id"] == "drug_ae":
            if unlocked < 3:
                if st.button(
                    "Audit Medications & Adverse Effects (Step 3) →",
                    key=f"hyp_jump_step3_{active_pid}_{btn_suffix}",
                    width="stretch",
                ):
                    pstate["unlocked_stage"] = max(unlocked, 3)
                    pstate["active_stage"] = 3
                    st.rerun()
            elif unlocked == 3:
                act_c1, act_c2 = st.columns(2)
                if drug_gaps_cnt > 0:
                    with act_c1:
                        if st.button(
                            f"→ {drug_gaps_cnt} Gap{'s' if drug_gaps_cnt != 1 else ''} · Unlock ZIP {postal_code}",
                            key=f"hyp_unlock_zip_{active_pid}_{btn_suffix}",
                            type="primary",
                            width="stretch",
                            help="Mark medications insufficient as sole cause and unlock Step 4 ZIP Outbreak Surveillance.",
                        ):
                            pstate["drug_gate_validated_no_match"] = True
                            pstate["drug_gate_timestamp"] = datetime.now().strftime("%b %d, %Y %H:%M")
                            pstate["unlocked_stage"] = 4
                            pstate["active_stage"] = 4
                            st.rerun()
                    with act_c2:
                        if st.button(
                            "✓ Confirm Drug Cause",
                            key=f"hyp_confirm_drug_{active_pid}_{btn_suffix}",
                            width="stretch",
                        ):
                            pstate["drug_gate_confirmed_match"] = True
                            pstate["drug_gate_confirmed_timestamp"] = datetime.now().strftime("%b %d, %Y %H:%M")
                            st.rerun()
                else:
                    with act_c1:
                        if st.button(
                            "✓ Confirm Drug Cause (100% Fit)",
                            key=f"hyp_confirm_drug_full_{active_pid}_{btn_suffix}",
                            type="primary",
                            width="stretch",
                        ):
                            pstate["drug_gate_confirmed_match"] = True
                            pstate["drug_gate_confirmed_timestamp"] = datetime.now().strftime("%b %d, %Y %H:%M")
                            st.rerun()
                    with act_c2:
                        if st.button(
                            f"Check ZIP {postal_code} Anyway →",
                            key=f"hyp_unlock_zip_anyway_{active_pid}_{btn_suffix}",
                            width="stretch",
                        ):
                            pstate["drug_gate_validated_no_match"] = True
                            pstate["drug_gate_timestamp"] = datetime.now().strftime("%b %d, %Y %H:%M")
                            pstate["unlocked_stage"] = 4
                            pstate["active_stage"] = 4
                            st.rerun()

    if unlocked >= 4 and pstate["active_stage"] != 4 and compact_side_panel:
        if st.button(
            f"Open Step 4: ZIP {postal_code} Outbreak Dashboard →",
            key=f"hyp_open_step4_{active_pid}_{btn_suffix}",
            width="stretch",
        ):
            pstate["active_stage"] = 4
            st.rerun()

    # Live Google ADK Hypothesis Verdict (Concise 3-Part Clinical Synthesis)
    run_adk_clicked = st.button(
        f"✦ Generate AI Clinical Verdict (Stage {unlocked})",
        key=f"run_adk_sym_{active_pid}_{unlocked}_{btn_suffix}",
        type="primary" if unlocked >= 4 else "secondary",
        width="stretch",
    )
    if run_adk_clicked:
        bedside_context_parts: list[str] = []
        if ruled_out_meds:
            bedside_context_parts.append(f"Physician ruled out medications: {', '.join(sorted(ruled_out_meds))}")
        if pstate.get("drug_gate_note"):
            bedside_context_parts.append(pstate["drug_gate_note"])
        with st.spinner("Synthesizing concise Competing Hypotheses Verdict via Google ADK..."):
            try:
                ai_md = run_adk_symptom_mapper(
                    patient_id=active_pid,
                    unlocked_stage=unlocked,
                    physician_validation_note="; ".join(bedside_context_parts),
                )
                pstate["adk_ai_mapping_md"] = clean_agent_md(ai_md)
                pstate["adk_ai_stage"] = unlocked
            except Exception as exc:
                pstate["adk_ai_mapping_md"] = (
                    f"**ADK Agent Notice:** Could not reach Vertex AI (`{type(exc).__name__}`). "
                    "Deterministic Competing Hypotheses Scorecard above remains fully active."
                )
                pstate["adk_ai_stage"] = unlocked

    if pstate.get("adk_ai_mapping_md"):
        with st.expander(
            f"✦ AI Competing Hypotheses Verdict (Stage {pstate.get('adk_ai_stage', unlocked)})",
            expanded=True,
        ):
            st.markdown(clean_agent_md(pstate["adk_ai_mapping_md"]))

    # Optional clean collapsed evidence drawer for citations (zero clutter by default)
    if mapped_list:
        with st.expander("🔍 Inspect Underlying Citations & Finding Rationales", expanded=False):
            ev_rows = ""
            for sym in mapped_list:
                sources: list[str] = []
                for m in sym.get("ehr_matches", [])[:1]:
                    sources.append(f'<span class="badge badge-blue">EHR</span> {esc(m["finding"])} <span class="badge-mono">{esc(m["citation"])}</span>')
                if unlocked >= 3:
                    for m in sym.get("drug_matches", [])[:1]:
                        if m["medication"] not in ruled_out_meds:
                            sources.append(f'<span class="badge badge-amber">DRUG</span> {esc(m["medication"])}')
                if unlocked >= 4:
                    for m in sym.get("outbreak_matches", [])[:1]:
                        sources.append(f'<span class="badge badge-red">ZIP</span> {esc(m["pathogen_or_factor"])}')
                src_html = "<br>".join(sources) or '<span style="color:var(--text-muted);">Unexplained at current stage</span>'
                ev_rows += (
                    f'<tr>'
                    f'<td style="font-weight:600;">{esc(sym.get("short_label") or sym["label"])}<br>'
                    f'<span class="badge-mono">{esc(sym["fhir_citation"])}</span></td>'
                    f'<td style="font-size:0.75rem;">{src_html}</td>'
                    f'</tr>'
                )
            render_html(
                f"""
                <div class="data-table-wrap" style="margin-bottom:0.1rem;">
                    <table class="data-table">
                        <thead><tr><th>Finding</th><th>Matched Evidence</th></tr></thead>
                        <tbody>{ev_rows}</tbody>
                    </table>
                </div>
                """
            )


# ---------------------------------------------------------------------------
# 6. Top-Level Workspace Navigation (Supports 1-Click Auto-Switch from Roster)
# ---------------------------------------------------------------------------
_VIEW_KEYS = ["roster", "workspace", "sym_matrix"]
_VIEW_LABELS = {
    "roster": "1 · Emergency Ward Patient Roster",
    "workspace": f"2 · Guided Clinical Investigation ({demo.get('name', active_pid)})",
    "sym_matrix": "3 · Competing Hypotheses Scorecard",
}
active_view = st.radio(
    "Workspace View",
    options=_VIEW_KEYS,
    format_func=lambda k: _VIEW_LABELS[k],
    horizontal=True,
    label_visibility="collapsed",
    key="active_view",
)

# ===========================================================================
# VIEW 1: EMERGENCY WARD PATIENT ROSTER (SELECT PATIENT)
# ===========================================================================
if active_view == "roster":
    roster_hdr_col, roster_cta_col = st.columns([6.6, 3.4], vertical_alignment="center")
    with roster_hdr_col:
        render_html(
            f"""
            <div class="epa-card" style="margin-bottom:0.55rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
                    <div>
                        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                            <span class="badge badge-blue">STEP 1 · CHOOSE PATIENT</span>
                            <span style="font-size:1.02rem;font-weight:700;color:var(--text);">
                                Emergency Ward Active Patient Roster ({len(ward_patients)} Patients)
                            </span>
                        </div>
                        <div style="font-size:0.79rem;color:var(--text-muted);margin-top:4px;">
                            Click <b>Open Workspace →</b> on any patient below (or continue with the selected patient on the right) to proceed to <b>Page 2: Guided Clinical Investigation</b>.
                        </div>
                    </div>
                    <span class="badge badge-green">Live HL7 FHIR R4 Connected</span>
                </div>
            </div>
            """
        )
    with roster_cta_col:
        st.button(
            f"Open Clinical Workspace: {demo.get('name', active_pid)} ({active_pid}) →",
            key="roster_top_continue_cta",
            type="primary",
            width="stretch",
            on_click=select_patient,
            args=(active_pid,),
            help="Advance immediately to Page 2 (Guided Clinical Investigation) for the selected patient.",
        )

    search_q = st.text_input(
        "Filter Emergency Ward Roster by Name, Patient ID, Chief Complaint, or 5-Digit ZIP Code",
        placeholder="e.g., Alex Jordan Conn, pat-001, 02118, dyspnea, rhabdomyolysis...",
    ).strip().lower()

    filtered_patients = [
        p
        for p in ward_patients
        if not search_q
        or search_q in p["patient_id"].lower()
        or search_q in p["full_name"].lower()
        or search_q in p["chief_complaint_today"].lower()
        or search_q in p.get("postal_code", "").lower()
        or search_q in p.get("location", "").lower()
    ]

    # Styled HTML summary table + unmistakable one-click workspace launch buttons
    roster_rows = ""
    for p in filtered_patients:
        is_sel = p["patient_id"] == active_pid
        ps = get_pstate(p["patient_id"])
        stage_badge = f'<span class="badge badge-blue">Step {ps["active_stage"]}/4</span>'
        if ps["drug_gate_validated_no_match"]:
            stage_badge = '<span class="badge badge-green">ZIP Unlocked</span>'
        elif ps.get("drug_gate_confirmed_match"):
            stage_badge = '<span class="badge badge-green">Drug Etiology Confirmed</span>'
        next_badge = (
            '<span class="badge badge-blue">Selected · Ready to Open →</span>'
            if is_sel
            else '<span class="badge-mono">Click Button Below →</span>'
        )
        sel_style = "background:var(--blue-muted);" if is_sel else ""
        roster_rows += (
            f'<tr style="{sel_style}">'
            f'<td><span class="badge-mono">{esc(p["patient_id"])}</span></td>'
            f'<td style="font-weight:600;">{esc(p["full_name"])}</td>'
            f'<td>{esc((p.get("gender") or "").capitalize())} · {fmt_date(p.get("birth_date"))}</td>'
            f'<td>{esc(p.get("city"))}, {esc(p.get("state"))} <span class="badge-mono">{esc(p.get("postal_code"))}</span></td>'
            f'<td>{esc(p.get("chief_complaint_today"))}</td>'
            f'<td>{stage_badge}</td>'
            f'<td>{next_badge}</td>'
            '</tr>'
        )

    render_html(
        f"""
        <div class="data-table-wrap">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Patient ID</th>
                        <th>Patient Name</th>
                        <th>Sex &amp; DOB</th>
                        <th>Location &amp; 5-Digit ZIP</th>
                        <th>Today's Chief Complaint</th>
                        <th>Workflow Status</th>
                        <th>Next Step</th>
                    </tr>
                </thead>
                <tbody>{roster_rows}</tbody>
            </table>
        </div>
        """
    )

    render_html(
        '<div style="font-size:0.84rem;font-weight:700;color:var(--text);margin:0.2rem 0 0.45rem;">'
        'Click Any Patient Below to Open Their Guided Clinical Workspace (Page 2) &rarr;'
        '</div>'
    )

    cols_per_row = 4
    for i in range(0, len(filtered_patients), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, p in enumerate(filtered_patients[i : i + cols_per_row]):
            with row_cols[j]:
                is_current = p["patient_id"] == active_pid
                btn_lbl = f"Open {p['patient_id']} · {p['full_name']} →"
                st.button(
                    btn_lbl,
                    key=f"roster_btn_{p['patient_id']}",
                    type="primary" if is_current else "secondary",
                    width="stretch",
                    on_click=select_patient,
                    args=(p["patient_id"],),
                )

    render_html('<div style="height:0.35rem;"></div>')
    st.button(
        f"Proceed to Page 2: Guided Clinical Investigation for {demo.get('name', active_pid)} ({active_pid}) →",
        key="roster_bottom_continue_cta",
        type="primary",
        width="stretch",
        on_click=select_patient,
        args=(active_pid,),
    )

# ===========================================================================
# VIEW 2: SPLIT CLINICAL WORKSPACE (LEFT: 4-STAGE INVESTIGATION | RIGHT: SYMPTOM MAPPER UI)
# ===========================================================================
elif active_view == "workspace":
    # Patient Identity & High-Criticality Allergy Header Strip
    allergies = demo.get("allergies", [])
    allergy_badges = " ".join(
        f'<span class="badge {"badge-red" if "high" in str(a.get("criticality", "")).lower() or "severe" in str(a.get("severity", "")).lower() else "badge-amber"}">'
        f'Allergy: {esc(a.get("substance"))} ({esc(a.get("reaction"))})</span>'
        for a in allergies
    )
    if pstate.get("drug_gate_confirmed_match") and not pstate.get("drug_gate_validated_no_match"):
        allergy_badges += ' <span class="badge badge-green">Drug Etiology Confirmed</span>'

    hdr_col, rst_col = st.columns([6.5, 1.0])
    with hdr_col:
        st.markdown(
            f"""
            <div class="epa-card" style="padding:1rem 1.25rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
                    <div>
                        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                            <span style="font-size:1.2rem;font-weight:700;color:var(--text);">{esc(demo.get('name'))}</span>
                            <span class="badge-mono">{esc(active_pid)}</span>
                            <span class="badge badge-blue">{esc(demo.get('age'))}yo {esc(demo.get('sex'))} · DOB {fmt_date(demo.get('dob'))}</span>
                            <span class="badge-mono">ZIP {esc(postal_code)} · {esc(demo.get('city'))}, {esc(demo.get('state'))}</span>
                        </div>
                        <div style="font-size:0.82rem;color:var(--text-muted);margin-top:6px;">
                            <b>Today's Triage ({fmt_date(visit.get('visit_date'))}):</b> "{esc(visit.get('chief_complaint'))}"
                            · PCP: {esc(demo.get('primary_care_physician'))}
                        </div>
                    </div>
                    <div style="display:flex;gap:6px;flex-wrap:wrap;">{allergy_badges}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with rst_col:
        if st.button("↺ Reset Case", key=f"reset_case_{active_pid}", width="stretch", help="Reset this patient back to Step 1"):
            reset_pstate(active_pid)
            st.rerun()

    # 4-Step Progressive Clinical Stepper Navigation
    s_cols = st.columns(4)
    step_defs = [
        (1, "Step 1 · Longitudinal Charts", "Labs, Imaging, Notes & Loops"),
        (2, "Step 2 · Swimlane Chart", "4-Track Temporal Causality"),
        (3, "Step 3 · Drugs & Adverse Effects", "RxNorm + OpenFDA Validation Gate"),
        (4, f"Step 4 · ZIP {postal_code} Outbreaks", "CDC ED, Wastewater & ZCTA5"),
    ]
    for col_obj, (s_num, s_title, s_sub) in zip(s_cols, step_defs):
        with col_obj:
            is_unlocked = pstate["unlocked_stage"] >= s_num
            is_active = pstate["active_stage"] == s_num
            lock_icon = "● " if is_active else ("✓ " if is_unlocked else "🔒 ")
            if st.button(
                f"{lock_icon}{s_title}",
                key=f"nav_step_{active_pid}_{s_num}",
                type="primary" if is_active else "secondary",
                disabled=not is_unlocked,
                width="stretch",
            ):
                pstate["active_stage"] = s_num
                st.rerun()

    st.markdown('<div style="height:0.35rem;"></div>', unsafe_allow_html=True)

    # Split Layout: Main Clinical Investigation (Left 64%) | Separate Symptom Mapping UI (Right 36%)
    col_main, col_sym = st.columns([6.4, 3.6], gap="large")

    with col_main:
        current_stage = pstate["active_stage"]

        # -------------------------------------------------------------------
        # STEP 1: LONGITUDINAL MEDICAL CHARTS
        # -------------------------------------------------------------------
        if current_stage == 1:
            v1, v2, v3, v4 = st.columns(4)
            with v1:
                bp_str = str(vitals.get("bp", "—"))
                metric_card("Blood Pressure", bp_str, "Triage Vital", "down" if "15" in bp_str or "16" in bp_str or "98/" in bp_str else "info")
            with v2:
                hr_str = str(vitals.get("hr", "—"))
                metric_card("Heart Rate", hr_str, "Triage Vital", "warn" if "10" in hr_str or "11" in hr_str else "up")
            with v3:
                spo2_str = str(vitals.get("spo2", "—"))
                metric_card("SpO2 Saturation", spo2_str, "Room Air", "down" if "91" in spo2_str or "92" in spo2_str else "up")
            with v4:
                temp_or_wt = vitals.get("temp") or f"{vitals.get('weight_lbs', '—')} lbs"
                metric_card("Temp / Weight", str(temp_or_wt), f"Wt: {vitals.get('weight_lbs', '—')} lbs", "warn")

            st.markdown(
                f"""
                <div class="epa-card" style="margin-top:0.75rem;">
                    <div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;color:var(--text-muted);letter-spacing:0.04em;">
                        MA / Nurse Triage Intake Note · <span class="badge-mono">{esc(visit.get('fhir_resource'))}</span>
                    </div>
                    <div style="font-size:0.86rem;color:var(--text);margin-top:6px;line-height:1.55;">
                        {esc(vitals.get('ma_intake_note', 'No intake note recorded.'))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Interactive Plotly Longitudinal Biomarker Charts (1 chart per row per design rules)
            long_labs = labs_data.get("labs_and_diagnostics", {}).get("longitudinal_labs", [])
            st.markdown(
                '<div style="font-size:0.96rem;font-weight:700;margin:0.6rem 0 0.45rem;">Longitudinal Laboratory Trajectories</div>',
                unsafe_allow_html=True,
            )
            chart_colors = ["#3b82f6", "#ef4444", "#f59e0b", "#10b981", "#8b5cf6"]
            for l_idx, lab in enumerate(long_labs):
                hist = lab.get("history", [])
                numeric_pts = []
                for h in hist:
                    try:
                        numeric_pts.append((fmt_date(h["date"]), float(h["value"]), h.get("flag", "N"), h.get("context", "")))
                    except (ValueError, TypeError):
                        pass
                if len(numeric_pts) >= 2:
                    st.markdown(
                        (
                            '<div class="chart-wrap" style="margin-bottom:0.35rem;">'
                            f'<div class="chart-title">{esc(lab["test"])}</div>'
                            f'<div class="chart-subtitle">Reference Range: {esc(lab["ref_range"])} · LOINC {esc(lab.get("loinc_code"))}</div>'
                            '</div>'
                        ),
                        unsafe_allow_html=True,
                    )
                    fig = go.Figure()
                    line_color = chart_colors[l_idx % len(chart_colors)]
                    marker_colors = [
                        "#ef4444" if fl in ("HH", "LL") else ("#f59e0b" if fl in ("H", "L") else "#22c55e")
                        for _, _, fl, _ in numeric_pts
                    ]
                    fig.add_trace(
                        go.Scatter(
                            x=[p[0] for p in numeric_pts],
                            y=[p[1] for p in numeric_pts],
                            mode="lines+markers+text",
                            text=[f"{p[1]} [{p[2]}]" for p in numeric_pts],
                            textposition="top center",
                            line=dict(color=line_color, width=3),
                            marker=dict(size=10, color=marker_colors, line=dict(width=1.5, color="#ffffff")),
                            hovertemplate="<b>%{x}</b><br>Value: %{y}<br>%{customdata}<extra></extra>",
                            customdata=[p[3] for p in numeric_pts],
                        )
                    )
                    fig.update_layout(**PLOT_LAYOUT, height=210, showlegend=False)
                    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            # Lab Summary HTML Table
            lab_rows = ""
            for lab in long_labs:
                hist = lab.get("history", [])
                traj_html = " &rarr; ".join(
                    f"<span style='font-family:JetBrains Mono,monospace;'>{esc(h['value'])}</span> {flag_badge(h['flag'])} "
                    f"<span style='color:var(--text-muted);font-size:0.72rem;'>({fmt_date(h['date'])})</span>"
                    for h in hist
                )
                latest_res = hist[-1]["fhir_resource"] if hist else "Observation"
                lab_rows += f"""
                <tr>
                    <td style="font-weight:600;">{esc(lab['test'])}</td>
                    <td><span class="badge-mono">{esc(lab['ref_range'])}</span></td>
                    <td>{traj_html}</td>
                    <td><span class="badge-mono">{esc(latest_res)}</span></td>
                </tr>
                """
            st.markdown(
                f"""
                <div class="data-table-wrap">
                    <table class="data-table">
                        <thead><tr><th>Biomarker</th><th>Ref Range</th><th>Baseline &rarr; Peak &rarr; Latest Trajectory</th><th>FHIR Resource</th></tr></thead>
                        <tbody>{lab_rows.replace("\n", "")}</tbody>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Diagnostic Imaging + Hospitalization/Consult Notes + Open Loops
            imaging_list = labs_data.get("labs_and_diagnostics", {}).get("imaging_and_procedures", [])
            if imaging_list:
                st.markdown('<div style="font-size:0.94rem;font-weight:700;margin:0.4rem 0;">Diagnostic Imaging &amp; Procedures</div>', unsafe_allow_html=True)
                for img in imaging_list:
                    st.markdown(
                        f"""
                        <div class="epa-card" style="padding:0.95rem 1.15rem;">
                            <div style="display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;">
                                <div style="font-weight:700;font-size:0.85rem;">{esc(img['modality'])} · {fmt_date(img['date'])}</div>
                                <span class="badge-mono">{esc(img['fhir_resource'])}</span>
                            </div>
                            <div style="font-size:0.8rem;color:var(--text-muted);margin-top:5px;">{esc(img['findings'])}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            enc_notes = notes_data.get("encounters_and_notes", [])
            if enc_notes:
                st.markdown('<div style="font-size:0.94rem;font-weight:700;margin:0.4rem 0;">Hospitalization &amp; Specialist Consult Notes</div>', unsafe_allow_html=True)
                for en in enc_notes:
                    st.markdown(
                        f"""
                        <div class="epa-card" style="padding:0.95rem 1.15rem;">
                            <div style="display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;">
                                <div style="font-weight:700;font-size:0.85rem;">{fmt_date(en['date'])} · {esc(en['type'])} ({esc(en['facility'])})</div>
                                <span class="badge-mono">{esc(", ".join(en.get("fhir_resources", [])))}</span>
                            </div>
                            <div style="font-size:0.78rem;font-weight:600;color:#3b82f6;margin-top:3px;">
                                Primary Diagnosis: {esc(en['primary_diagnosis'])} · {esc(en['author'])}
                            </div>
                            <div style="font-size:0.8rem;color:var(--text-muted);margin-top:5px;line-height:1.5;">{esc(en['summary_note'])}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            open_loops = loops_data.get("unresolved_items_and_care_gaps", [])
            if open_loops:
                st.markdown('<div style="font-size:0.94rem;font-weight:700;margin:0.4rem 0;">Unresolved Open Clinical Loops &amp; Care Gaps</div>', unsafe_allow_html=True)
                for lp in open_loops:
                    st.markdown(
                        f"""
                        <div class="epa-card" style="padding:0.9rem 1.1rem;">
                            <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;">
                                <div style="font-weight:700;font-size:0.84rem;">{flag_badge(lp['priority'])} &nbsp;{esc(lp['category'])}</div>
                                <span class="badge-mono">{esc(lp['fhir_resource'])}</span>
                            </div>
                            <div style="font-size:0.8rem;color:var(--text-muted);margin-top:5px;">{esc(lp['details'])}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if st.button(
                f"Proceed to Step 2: Review 4-Track Clinical Swimlane Chart for {demo.get('name')} →",
                type="primary",
                width="stretch",
                key=f"advance_to_step2_{active_pid}",
            ):
                pstate["unlocked_stage"] = max(pstate["unlocked_stage"], 2)
                pstate["active_stage"] = 2
                st.rerun()

        # -------------------------------------------------------------------
        # STEP 2: INTERACTIVE 4-TRACK CLINICAL SWIMLANE CHART
        # -------------------------------------------------------------------
        elif current_stage == 2:
            with st.spinner("Synchronizing 4-Track HL7 FHIR R4 Clinical Swimlane Timeline..."):
                swim_data = cached_swimlane(active_pid)

            st.markdown(
                f"""
                <div class="epa-card">
                    <div style="font-size:0.98rem;font-weight:700;color:var(--text);">{esc(swim_data.get('headline'))}</div>
                    <div style="font-size:0.79rem;color:var(--text-muted);margin-top:4px;">
                        Synchronizes Encounters, Medications, Labs/Imaging, and Open Clinical Loops along a shared horizontal timeline before initiating formal Drug &amp; OpenFDA Adverse-Effect verification.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Embedded interactive HTML/SVG Swimlane viewer (tall enough for all 4 tracks + inspector)
            if swim_data.get("html_content"):
                components.html(swim_data["html_content"], height=1200, scrolling=True)

            # Synchronized 4-Track Temporal Matrix HTML Table (collapsible printable view)
            tps = swim_data.get("synchronized_timepoints", [])
            tracks_meta = swim_data.get("tracks", [])
            if tps and tracks_meta:
                th_cols = "".join(
                    f"<th>{fmt_date(tp.get('date'))}<br><span style='font-weight:400;text-transform:none;'>{esc(tp.get('phase'))}</span></th>"
                    for tp in tps
                )
                tr_rows = ""
                for trk in tracks_meta:
                    tid = trk["track_id"]
                    cells = ""
                    for tp in tps:
                        cell = tp.get("tracks", {}).get(tid, {})
                        sev = str(cell.get("severity", "info")).lower()
                        b_cls = "badge-red" if sev in ("critical", "high") else ("badge-amber" if sev in ("warning", "moderate") else "badge-blue")
                        cells += (
                            f"<td>"
                            f"<span class='badge {b_cls}'>{esc(cell.get('badge', 'EVENT'))}</span>"
                            f"<div style='font-weight:600;margin-top:4px;'>{esc(cell.get('title', '—'))}</div>"
                            f"<div style='font-size:0.75rem;color:var(--text-muted);margin-top:2px;'>{esc(cell.get('detail', ''))}</div>"
                            f"</td>"
                        )
                    tr_rows += f"<tr><td style='font-weight:700;'>{esc(trk['label'])}<br><span style='font-size:0.72rem;font-weight:400;color:var(--text-muted);'>{esc(trk['subtitle'])}</span></td>{cells}</tr>"

                with st.expander("View Printable 4-Track Matrix Table", expanded=False):
                    st.markdown(
                        f"""
                        <div class="data-table-wrap" style="margin-top:0.2rem;">
                            <table class="data-table">
                                <thead><tr><th>Parallel Clinical Track</th>{th_cols}</tr></thead>
                                <tbody>{tr_rows.replace("\n", "")}</tbody>
                            </table>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            causal_links = swim_data.get("causal_links", [])
            if causal_links:
                st.markdown('<div style="font-size:0.92rem;font-weight:700;margin:0.5rem 0;">Cross-Track Temporal Causal Links</div>', unsafe_allow_html=True)
                for cl in causal_links:
                    st.markdown(
                        f"""
                        <div class="epa-card" style="padding:0.78rem 1rem;margin-bottom:0.5rem;">
                            <span class="badge-mono">{esc(cl.get('from'))} &rarr; {esc(cl.get('to'))}</span>
                            <span style="font-size:0.81rem;color:var(--text);margin-left:8px;">{esc(cl.get('label'))}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if st.button(
                f"Swimlane Reviewed → Initiate Step 3: Check Drugs & Adverse Effects for {demo.get('name')} →",
                type="primary",
                width="stretch",
                key=f"advance_to_step3_{active_pid}",
            ):
                pstate["unlocked_stage"] = max(pstate["unlocked_stage"], 3)
                pstate["active_stage"] = 3
                st.rerun()

        # -------------------------------------------------------------------
        # STEP 3: DRUGS USED & ADVERSE EFFECTS + PHYSICIAN VALIDATION GATE
        # -------------------------------------------------------------------
        elif current_stage == 3:
            if not drug_bundle:
                drug_bundle = cached_drug_audit(active_pid)

            med_hist = drug_bundle["med_hist"]
            rx_map = drug_bundle["rx_by_name"]
            active_meds = med_hist.get("medications", {}).get("active_prescriptions", [])
            held_meds = med_hist.get("medications", {}).get("held_or_discontinued_medications", [])

            st.markdown(
                """
                <div class="epa-card">
                    <div style="font-size:0.98rem;font-weight:700;color:var(--text);">
                        Medication Reconciliation, NIH RxNorm Generic/Brand &amp; OpenFDA Adverse Effects Audit
                    </div>
                    <div style="font-size:0.79rem;color:var(--text-muted);margin-top:4px;">
                        Review every active, held, and stopped medication alongside OpenFDA boxed warnings and FAERS adverse event reports. Once you validate whether drugs explain all presenting symptoms, unlock Step 4 (5-Digit ZIP Outbreak Surveillance).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for dup_alert in drug_bundle.get("duplicate_alerts", []):
                st.markdown(
                    f"""
                    <div class="epa-card" style="padding:0.85rem 1.1rem;border-color:var(--red);background:var(--red-muted);">
                        <div style="font-size:0.83rem;font-weight:700;color:var(--red);">
                            DUPLICATE ACTIVE INGREDIENT SAFETY HAZARD · {esc(dup_alert.get('shared_active_ingredient', '')).upper()}
                        </div>
                        <div style="font-size:0.78rem;color:var(--text);margin-top:3px;">
                            {esc(dup_alert.get('warning', ''))}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            faers_chart_rows: list[dict[str, Any]] = []

            for m in active_meds:
                m_name = m["medication"]
                rx = rx_map.get(m_name, {})
                is_brand = rx.get("is_brand_name", False)
                cls_badge = (
                    '<span class="badge badge-amber">BRAND NAME (RxNorm)</span>'
                    if is_brand
                    else '<span class="badge badge-green">ORIGINAL GENERIC (RxNorm)</span>'
                )
                uses = rx.get("openfda_drug_uses", {})
                side_fx = rx.get("openfda_known_side_effects", {})
                boxed = side_fx.get("boxed_warning_excerpt")
                faers = side_fx.get("top_faers_reported_adverse_events", {})
                if isinstance(faers, dict):
                    for term, cnt in list(faers.items())[:5]:
                        faers_chart_rows.append({"Medication": m_name.split()[0], "Reaction": f"{m_name.split()[0]}: {term}", "Reports": cnt})

                boxed_html = (
                    f'<div style="margin-top:8px;padding:0.65rem 0.85rem;border-radius:7px;background:var(--red-muted);border:1px solid var(--red);font-size:0.77rem;color:var(--text);">'
                    f'<b style="color:var(--red);">FDA BOXED WARNING:</b> {esc(boxed)}</div>'
                    if boxed
                    else ""
                )
                faers_pills = " ".join(
                    f'<span class="badge-mono">{esc(k)} ({esc(v)})</span>'
                    for k, v in (list(faers.items())[:6] if isinstance(faers, dict) else [])
                )

                faers_row_html = (
                    '<div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;align-items:center;">'
                    '<span style="font-size:0.73rem;font-weight:600;color:var(--text-muted);">Top FAERS Reports:</span>'
                    f'{faers_pills or "<span style=\\\"font-size:0.73rem;color:var(--text-muted);\\\">None returned</span>"}'
                    '</div>'
                )
                st.markdown(
                    (
                        '<div class="epa-card">'
                        '<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap;">'
                        '<div>'
                        f'<span style="font-size:0.95rem;font-weight:700;color:var(--text);">{esc(m_name)}</span>'
                        f'&nbsp;{cls_badge}'
                        f'&nbsp;<span class="badge-mono">RxCUI {esc(rx.get("rxcui") or m.get("rxcui_code") or "—")}</span>'
                        f'&nbsp;<span class="badge-mono">{esc(m.get("fhir_resource"))}</span>'
                        '</div>'
                        f'<span class="badge badge-blue">{esc(m.get("change_type"))}</span>'
                        '</div>'
                        '<div style="font-size:0.8rem;color:var(--text);margin-top:7px;">'
                        f'<b>Prescribed Dose:</b> {esc(m.get("prescribed_dose"))} &nbsp;|&nbsp; '
                        f'<b>Adherence / Reconciliation:</b> {esc(m.get("actual_patient_adherence"))}'
                        '</div>'
                        '<div style="font-size:0.78rem;color:var(--text-muted);margin-top:5px;">'
                        f'<b>Active Ingredient(s):</b> {esc(", ".join(rx.get("original_generic_ingredients", [])) or rx.get("rxnorm_normalized_name") or "See RxNorm")} '
                        f'&nbsp;|&nbsp; <b>Clinical Uses (OpenFDA):</b> {esc((uses.get("indications_and_usage") or uses.get("purpose") or "Standard labeled indication")[:220])}'
                        '</div>'
                        '<div style="font-size:0.78rem;color:var(--text-muted);margin-top:5px;">'
                        f'<b>Known Labeled Adverse Reactions:</b> {esc((side_fx.get("labeled_adverse_reactions") or side_fx.get("warnings_and_cautions") or "See OpenFDA SPL")[:260])}'
                        '</div>'
                        f'{boxed_html}{faers_row_html}'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )

                cur_dec = pstate["drug_decisions"].get(m_name, "Undecided")
                if cur_dec == "Rule Out (No Match)":
                    cur_dec = "Ruled Out (No Match)"
                dec_opts = ["Undecided", "Ruled Out (No Match)", "Matches Symptoms"]
                chosen_dec = st.radio(
                    f"Physician Assessment for {m_name}",
                    options=dec_opts,
                    index=dec_opts.index(cur_dec),
                    horizontal=True,
                    key=f"drug_dec_{active_pid}_{m['medication_request_id']}",
                )
                if chosen_dec != cur_dec:
                    pstate["drug_decisions"][m_name] = chosen_dec
                    st.rerun()

            if held_meds:
                st.markdown('<div style="font-size:0.92rem;font-weight:700;margin:0.6rem 0 0.4rem;">Held &amp; Previously Discontinued Medications</div>', unsafe_allow_html=True)
                held_rows = ""
                for hm in held_meds:
                    rx_h = rx_map.get(hm["medication"], {})
                    cls_txt = "BRAND" if rx_h.get("is_brand_name") else "GENERIC"
                    held_rows += f"""
                    <tr>
                        <td style="font-weight:600;">{esc(hm['medication'])} <span class="badge-mono">{cls_txt}</span></td>
                        <td><span class="badge badge-amber">{esc(hm['status'])}</span></td>
                        <td>{esc(hm['reason'])}</td>
                        <td>{esc(hm['reinitiation_criteria'])}</td>
                        <td><span class="badge-mono">{esc(hm['fhir_resource'])}</span></td>
                    </tr>
                    """
                st.markdown(
                    f"""
                    <div class="data-table-wrap">
                        <table class="data-table">
                            <thead><tr><th>Medication</th><th>Status</th><th>Hold / Stop Reason</th><th>Restart Criteria</th><th>FHIR ID</th></tr></thead>
                            <tbody>{held_rows.replace("\n", "")}</tbody>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Interactive Plotly FAERS Chart (1 chart per row)
            if faers_chart_rows:
                df_faers = pd.DataFrame(faers_chart_rows).sort_values("Reports", ascending=True).tail(10)
                st.markdown(
                    (
                        '<div class="chart-wrap" style="margin-bottom:0.35rem;">'
                        '<div class="chart-title">OpenFDA FAERS Top Reported Adverse Event Counts</div>'
                        '<div class="chart-subtitle">Post-marketing adverse event frequency across active prescribed medications (OpenFDA /drug/event.json)</div>'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )
                fig_faers = go.Figure(
                    go.Bar(
                        x=df_faers["Reports"],
                        y=df_faers["Reaction"],
                        orientation="h",
                        marker=dict(color="#f59e0b", cornerradius=4),
                        hovertemplate="<b>%{y}</b><br>FAERS Reports: %{x:,}<extra></extra>",
                    )
                )
                fig_faers.update_layout(**PLOT_LAYOUT, height=280)
                st.plotly_chart(fig_faers, width="stretch", config={"displayModeBar": False})

            # ---------------------------------------------------------------
            # PHYSICIAN VALIDATION GATE -> UNLOCKS STEP 4 (ZIP OUTBREAKS)
            # ---------------------------------------------------------------
            unmatched_labels = symptom_map_data.get("unmatched_by_drugs_labels", [])
            ruled_out_drugs = [k for k, v in pstate["drug_decisions"].items() if v in ("Ruled Out (No Match)", "Rule Out (No Match)")]
            st.markdown(
                f"""
                <div class="epa-card" style="border:1.5px solid #2563eb;background:var(--blue-muted);">
                    <div style="font-size:0.96rem;font-weight:700;color:var(--text);">
                        Physician Drug &amp; Adverse-Effect Validation Gate
                    </div>
                    <div style="font-size:0.8rem;color:var(--text);margin-top:5px;line-height:1.5;">
                        • Drugs explicitly ruled out by physician: <b>{len(ruled_out_drugs)}/{len(active_meds)}</b><br>
                        • Presenting signs unmatched by drugs alone: <b>{esc(", ".join(unmatched_labels) if unmatched_labels else "None auto-flagged — verify clinically")}</b><br>
                        Validate below if medications do not match (or leave residual infectious symptoms) to proceed to <b>Step 4: Hyperlocal ZIP {esc(postal_code)} Outbreak Surveillance</b>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            gate_note = st.text_input(
                "Optional Physician Validation Note (recorded with case & passed to ADK Symptom Mapper)",
                value=pstate.get("drug_gate_note", ""),
                placeholder="e.g., Drugs ruled out as sole cause (fever 100.2F & acute bronchospasm not explained by meds) — check ZIP outbreaks.",
                key=f"gate_note_input_{active_pid}",
            )
            pstate["drug_gate_note"] = gate_note

            g1, g2 = st.columns(2)
            with g1:
                if st.button(
                    f"✓ Validate: Nothing Matches / Unexplained Symptoms → Check ZIP {postal_code} Outbreaks →",
                    type="primary",
                    width="stretch",
                    key=f"validate_unlock_step4_{active_pid}",
                ):
                    pstate["drug_gate_validated_no_match"] = True
                    pstate["drug_gate_timestamp"] = datetime.now().strftime("%b %d, %Y %H:%M")
                    pstate["unlocked_stage"] = 4
                    pstate["active_stage"] = 4
                    st.rerun()
            with g2:
                if st.button(
                    "Log Confirmed Drug Adverse-Effect Match Only",
                    width="stretch",
                    key=f"log_drug_match_only_{active_pid}",
                ):
                    pstate["drug_gate_confirmed_match"] = True
                    pstate["drug_gate_confirmed_timestamp"] = datetime.now().strftime("%b %d, %Y %H:%M")
                    st.rerun()

            if pstate.get("drug_gate_confirmed_match"):
                st.success(
                    f"Confirmed drug/adverse-effect etiology logged ({pstate.get('drug_gate_confirmed_timestamp', 'session')}). "
                    "You can still validate above anytime to check concurrent ZIP outbreaks."
                )

        # -------------------------------------------------------------------
        # STEP 4: HYPERLOCAL 5-DIGIT ZIP-CODE OUTBREAK SURVEILLANCE
        # -------------------------------------------------------------------
        elif current_stage == 4:
            if not pstate["drug_gate_validated_no_match"]:
                st.warning("Please complete the Physician Drug Validation Gate in Step 3 before viewing Step 4.")
            else:
                if not outbreak_bundle and postal_code:
                    outbreak_bundle = cached_zip_outbreaks(postal_code, demo.get("city", ""))

                outbreak_bundle = outbreak_bundle or {}
                zloc = outbreak_bundle.get("patient_zip_location", {})
                aq = outbreak_bundle.get("zip_code_air_quality", {})
                ed_traj = (
                    outbreak_bundle.get("active_outbreaks_in_zip_catchment", {})
                    .get("local_emergency_department_trajectories", {})
                )
                ww_list = (
                    outbreak_bundle.get("active_outbreaks_in_zip_catchment", {})
                    .get("local_sewershed_wastewater_pathogens", [])
                )
                places_list = outbreak_bundle.get("persistent_diseases_in_zip_code", [])

                st.markdown(
                    f"""
                    <div class="epa-card">
                        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                            <div>
                                <div style="font-size:0.98rem;font-weight:700;color:var(--text);">
                                    Hyperlocal Infection &amp; Outbreak Surveillance — Strictly 5-Digit ZIP <span class="badge-mono">{esc(postal_code)}</span>
                                </div>
                                <div style="font-size:0.78rem;color:var(--text-muted);margin-top:3px;">
                                    Physician Drug Validation Gate passed ({esc(pstate.get('drug_gate_timestamp') or 'Validated')}) · Locality: {esc(zloc.get('locality') or demo.get('city'))}
                                </div>
                            </div>
                            <span class="badge badge-green">Strictly Scoped to ZIP {esc(postal_code)}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                z1, z2, z3, z4 = st.columns(4)
                with z1:
                    metric_card("Patient ZIP Code", str(postal_code), str(zloc.get("locality") or demo.get("city")), "info")
                with z2:
                    pop = zloc.get("zip_total_population")
                    metric_card("ZIP Population", f"{pop:,}" if isinstance(pop, int) else "—", "CDC ZCTA5", "info")
                with z3:
                    aqi_val = aq.get("us_aqi")
                    aqi_str = str(aqi_val) if aqi_val is not None else "—"
                    aqi_type = "warn" if (isinstance(aqi_val, (int, float)) and aqi_val >= 50) else "up"
                    metric_card("US Air Quality (AQI)", aqi_str, "Open-Meteo ZIP", aqi_type)
                with z4:
                    pm_val = aq.get("pm2_5_ug_m3")
                    metric_card("PM2.5 Particulate", f"{pm_val} ug/m3" if pm_val is not None else "—", f"O3: {aq.get('ozone_ug_m3', '—')}", "info")

                # Chart 1: Local Hospital ED Respiratory Outbreak Trajectories
                path_trajs = ed_traj.get("pathogen_ed_trajectories", [])
                if path_trajs:
                    st.markdown(
                        (
                            '<div class="chart-wrap" style="margin-top:0.85rem;margin-bottom:0.35rem;">'
                            f'<div class="chart-title">Local Emergency Department Respiratory Pathogen Visits — ZIP {esc(postal_code)} Catchment</div>'
                            f'<div class="chart-subtitle">CDC NSSP Local Hospital ED Surveillance (rdmq-nq56) · Week Ending {fmt_date(ed_traj.get("week_ending"))}</div>'
                            '</div>'
                        ),
                        unsafe_allow_html=True,
                    )
                    bar_colors = [
                        "#ef4444" if "increas" in str(p.get("weekly_outbreak_trend", "")).lower() else "#3b82f6"
                        for p in path_trajs
                    ]
                    fig_ed = go.Figure(
                        go.Bar(
                            x=[p["pathogen"] for p in path_trajs],
                            y=[p["percent_of_local_ed_visits"] for p in path_trajs],
                            text=[f"{p['percent_of_local_ed_visits']}% ({p['weekly_outbreak_trend']})" for p in path_trajs],
                            textposition="outside",
                            marker=dict(color=bar_colors, cornerradius=6),
                            hovertemplate="<b>%{x}</b><br>% of Local ED Visits: %{y}%<extra></extra>",
                        )
                    )
                    max_y = max((float(p["percent_of_local_ed_visits"] or 0) for p in path_trajs), default=1.0)
                    fig_ed.update_layout(**PLOT_LAYOUT, height=250, yaxis_range=[0, max(max_y * 1.35, 1.0)])
                    st.plotly_chart(fig_ed, width="stretch", config={"displayModeBar": False})

                # Local Sewershed Wastewater PCR Pathogen Table
                if ww_list:
                    ww_rows = ""
                    for w in ww_list:
                        det = w.get("detected_in_local_wastewater")
                        det_badge = '<span class="badge badge-red">DETECTED (PCR+)</span>' if det else '<span class="badge badge-green">Below LOD</span>'
                        ww_rows += (
                            f'<tr><td style="font-weight:600;">{esc(w["pathogen"])}</td>'
                            f'<td>{det_badge}</td>'
                            f'<td><span class="badge-mono">{esc(w.get("viral_concentration"))} {esc(w.get("concentration_units"))}</span></td>'
                            f'<td>{esc(w.get("short_term_trend"))}</td>'
                            f'<td>{fmt_date(w.get("sample_collect_date"))}</td></tr>'
                        )
                    st.markdown(
                        (
                            '<div style="font-size:0.92rem;font-weight:700;margin:0.5rem 0 0.4rem;">Local Sewershed Wastewater Pathogen Surveillance (CDC NWSS)</div>'
                            '<div class="data-table-wrap"><table class="data-table">'
                            '<thead><tr><th>Pathogen</th><th>PCR Status</th><th>Concentration</th><th>Short-Term Trend</th><th>Sample Date</th></tr></thead>'
                            f'<tbody>{ww_rows}</tbody></table></div>'
                        ),
                        unsafe_allow_html=True,
                    )

                # Chart 2: Persistent Chronic / Endemic Disease Prevalence Strictly Inside ZIP
                if places_list:
                    top_places = places_list[:8]
                    st.markdown(
                        (
                            '<div class="chart-wrap" style="margin-bottom:0.35rem;">'
                            f'<div class="chart-title">Persistent Disease Prevalence Strictly Inside ZIP {esc(postal_code)} (CDC PLACES ZCTA5)</div>'
                            f'<div class="chart-subtitle">Source: data.cdc.gov/resource/qnzd-25i4.json?locationname={esc(postal_code)}</div>'
                            '</div>'
                        ),
                        unsafe_allow_html=True,
                    )
                    fig_pl = go.Figure(
                        go.Bar(
                            x=[r["prevalence_in_zip_percent"] for r in reversed(top_places)],
                            y=[r["short_name"] or r["condition"] for r in reversed(top_places)],
                            orientation="h",
                            marker=dict(color="#3b82f6", cornerradius=4),
                            text=[f"{r['prevalence_in_zip_percent']}%" for r in reversed(top_places)],
                            textposition="outside",
                            hovertemplate="<b>%{y}</b><br>ZIP Prevalence: %{x}%<extra></extra>",
                        )
                    )
                    max_prev = max((float(r["prevalence_in_zip_percent"] or 0) for r in top_places), default=10.0)
                    fig_pl.update_layout(**PLOT_LAYOUT, height=280, xaxis_range=[0, max_prev * 1.22])
                    st.plotly_chart(fig_pl, width="stretch", config={"displayModeBar": False})

    # -----------------------------------------------------------------------
    # RIGHT COLUMN: SEPARATE LIVE SYMPTOM MAPPING ASSISTANT UI
    # -----------------------------------------------------------------------
    with col_sym:
        render_symptom_mapping_ui(compact_side_panel=True)

# ===========================================================================
# VIEW 3: FULL-SCREEN COMPETING HYPOTHESES SCORECARD & COVERAGE GRID
# ===========================================================================
elif active_view == "sym_matrix":
    unlocked_now = pstate["unlocked_stage"]
    competing_full, ruled_out_full = _get_live_competing_hypotheses()
    total_f = competing_full["total_findings"]
    overall_exp = competing_full["overall_explained"]
    cov_pct = competing_full["coverage_pct"]
    drug_gaps_cnt = competing_full["drug_unexplained_count"]

    render_html(
        f"""
        <div class="epa-card" style="padding:1.05rem 1.25rem;margin-bottom:0.75rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
                <div>
                    <div style="font-size:1.05rem;font-weight:700;color:var(--text);">
                        Competing Hypotheses Scorecard — {esc(demo.get('name'))} (<span class="badge-mono">{esc(active_pid)}</span>)
                    </div>
                    <div style="font-size:0.79rem;color:var(--text-muted);margin-top:3px;">
                        Side-by-side root-cause comparison across Chronic EHR Baseline, Medication/Adverse-Effect Pharmacology, and Hyperlocal 5-Digit ZIP ({esc(postal_code)}) Outbreaks.
                    </div>
                </div>
                <div style="text-align:right;">
                    <span class="badge badge-blue">Diagnostic Coverage: {overall_exp}/{total_f} ({cov_pct}%)</span>
                </div>
            </div>
            <div class="hyp-coverage-track">
                <div class="hyp-coverage-fill" style="width:{max(4, cov_pct)}%;background:{'var(--green)' if overall_exp == total_f else '#2563eb'};"></div>
            </div>
        </div>
        """
    )

    # 3-Column Side-by-Side Competing Hypotheses Board
    hc1, hc2, hc3 = st.columns(3)
    for col_widget, hcard in zip((hc1, hc2, hc3), competing_full["hypotheses"]):
        with col_widget:
            render_html(build_hypothesis_card_html(hcard, rank_num=None))

    # 1-Click Bedside Action Bar + Live ADK Verdict Trigger
    bar_c1, bar_c2, bar_c3 = st.columns([1.2, 1.2, 1.4])
    with bar_c1:
        if unlocked_now < 3:
            if st.button(
                "Audit Medications in Step 3 →",
                key=f"full_jump_step3_{active_pid}",
                type="primary",
                width="stretch",
            ):
                pstate["unlocked_stage"] = max(unlocked_now, 3)
                pstate["active_stage"] = 3
                st.session_state.active_view = "workspace"
                st.rerun()
        else:
            if st.button(
                "✓ Confirm Medication Etiology",
                key=f"full_confirm_drug_{active_pid}",
                type="primary" if drug_gaps_cnt == 0 else "secondary",
                width="stretch",
            ):
                pstate["drug_gate_confirmed_match"] = True
                pstate["drug_gate_confirmed_timestamp"] = datetime.now().strftime("%b %d, %Y %H:%M")
                st.rerun()
    with bar_c2:
        if unlocked_now >= 3:
            btn_zip_lbl = (
                f"→ {drug_gaps_cnt} Drug Gap{'s' if drug_gaps_cnt != 1 else ''} · Unlock ZIP {postal_code}"
                if unlocked_now == 3 and drug_gaps_cnt > 0
                else (f"Unlock ZIP {postal_code} Outbreaks →" if unlocked_now == 3 else f"Open Step 4: ZIP {postal_code} Details →")
            )
            if st.button(
                btn_zip_lbl,
                key=f"full_unlock_zip_{active_pid}",
                type="primary" if unlocked_now == 3 and drug_gaps_cnt > 0 else "secondary",
                width="stretch",
            ):
                pstate["drug_gate_validated_no_match"] = True
                if not pstate.get("drug_gate_timestamp"):
                    pstate["drug_gate_timestamp"] = datetime.now().strftime("%b %d, %Y %H:%M")
                pstate["unlocked_stage"] = 4
                pstate["active_stage"] = 4
                st.session_state.active_view = "workspace"
                st.rerun()
        else:
            st.button(
                f"🔒 ZIP {postal_code} Outbreaks (Complete Step 3 First)",
                key=f"full_locked_zip_{active_pid}",
                disabled=True,
                width="stretch",
            )
    with bar_c3:
        if st.button(
            f"✦ Generate AI Clinical Verdict (Stage {unlocked_now})",
            key=f"full_run_adk_{active_pid}_{unlocked_now}",
            width="stretch",
        ):
            ctx_parts: list[str] = []
            if ruled_out_full:
                ctx_parts.append(f"Physician ruled out medications: {', '.join(sorted(ruled_out_full))}")
            if pstate.get("drug_gate_note"):
                ctx_parts.append(pstate["drug_gate_note"])
            with st.spinner("Synthesizing concise Competing Hypotheses Verdict via Google ADK..."):
                try:
                    ai_md = run_adk_symptom_mapper(
                        patient_id=active_pid,
                        unlocked_stage=unlocked_now,
                        physician_validation_note="; ".join(ctx_parts),
                    )
                    pstate["adk_ai_mapping_md"] = clean_agent_md(ai_md)
                    pstate["adk_ai_stage"] = unlocked_now
                except Exception as exc:
                    pstate["adk_ai_mapping_md"] = (
                        f"**ADK Agent Notice:** Could not reach Vertex AI (`{type(exc).__name__}`). "
                        "Deterministic Competing Hypotheses Scorecard remains active above."
                    )
                    pstate["adk_ai_stage"] = unlocked_now

    if pstate.get("adk_ai_mapping_md"):
        with st.expander(
            f"✦ AI Competing Hypotheses Verdict (Stage {pstate.get('adk_ai_stage', unlocked_now)})",
            expanded=True,
        ):
            st.markdown(clean_agent_md(pstate["adk_ai_mapping_md"]))

    # Compact At-a-Glance Finding-by-Hypothesis Checkmark Grid (2-second scan, zero paragraphs)
    matrix_rows = ""
    for sym in symptom_map_data.get("mapped_symptoms", []):
        chip_lbl = sym.get("short_label") or sym["label"]
        ehr_m = sym.get("ehr_matches", [])
        eff_drug_m = [
            m for m in (sym.get("drug_matches", []) if unlocked_now >= 3 else [])
            if m["medication"] not in ruled_out_full
        ]
        out_m = sym.get("outbreak_matches", []) if unlocked_now >= 4 else []

        if ehr_m:
            ehr_cell = (
                f'<span class="chip-explain" title="{esc(ehr_m[0]["rationale"] + " [" + ehr_m[0]["citation"] + "]")}">'
                f'✓ {esc(_truncate_text(ehr_m[0]["finding"], 38))}</span>'
            )
        else:
            ehr_cell = '<span style="color:var(--text-muted);">—</span>'

        if unlocked_now < 3:
            drug_cell = '<span class="badge-mono">Step 3</span>'
        elif eff_drug_m:
            drug_cell = (
                f'<span class="chip-explain" title="{esc(eff_drug_m[0]["rationale"])}">'
                f'✓ {esc(_truncate_text(eff_drug_m[0]["medication"], 34))}</span>'
            )
        else:
            drug_cell = '<span class="chip-gap">✗ No Drug Match</span>'

        if unlocked_now < 4:
            zip_cell = '<span class="badge-mono">Gated</span>'
        elif out_m:
            cls_z = "chip-resolved" if not eff_drug_m else "chip-explain"
            pfx_z = "★" if not eff_drug_m else "✓"
            zip_cell = (
                f'<span class="{cls_z}" title="{esc(out_m[0]["rationale"])}">'
                f'{pfx_z} {esc(_truncate_text(out_m[0]["pathogen_or_factor"], 32))}</span>'
            )
        else:
            zip_cell = '<span style="color:var(--text-muted);">—</span>'

        explained_any = bool(ehr_m) or bool(eff_drug_m) or bool(out_m)
        fit_pill = (
            '<span class="badge badge-green">Explained</span>'
            if explained_any
            else '<span class="badge badge-red">Unexplained</span>'
        )

        matrix_rows += (
            f'<tr>'
            f'<td style="font-weight:600;">{esc(chip_lbl)}'
            f'<div style="font-size:0.71rem;font-weight:400;color:var(--text-muted);">{esc(sym["observed_value"])}</div></td>'
            f'<td><span class="badge-mono">{esc(sym["category"].replace(" Abnormality", "").replace("Presenting ", ""))}</span></td>'
            f'<td>{ehr_cell}</td>'
            f'<td>{drug_cell}</td>'
            f'<td>{zip_cell}</td>'
            f'<td>{fit_pill}</td>'
            f'</tr>'
        )

    render_html(
        f"""
        <div style="font-size:0.88rem;font-weight:700;color:var(--text);margin:0.65rem 0 0.4rem;">
            At-a-Glance Finding Coverage Grid (Hover Any Pill for Clinical Rationale)
        </div>
        <div class="data-table-wrap">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Clinical Finding</th>
                        <th>Type</th>
                        <th>Chronic EHR Baseline</th>
                        <th>Medication / Adverse Effect</th>
                        <th>ZIP {esc(postal_code)} Outbreak / AQI</th>
                        <th>Coverage</th>
                    </tr>
                </thead>
                <tbody>{matrix_rows}</tbody>
            </table>
        </div>
        """
    )

# ===========================================================================
# PERSISTENT NON-DISRUPTIVE FOOTER: PUBLIC API & CLINICAL DATA REFERENCES
# ===========================================================================
render_html('<div style="height:1.1rem;"></div>')
api_pills_html = " ".join(
    f'<a href="{esc(r["docs_url"])}" target="_blank" rel="noopener noreferrer" '
    f'style="text-decoration:none;"><span class="badge-mono" style="cursor:pointer;">{esc(r["name"])} ↗</span></a>'
    for r in PUBLIC_API_REFERENCES
)
render_html(
    f"""
    <div style="border-top:1px solid var(--border);padding-top:0.85rem;margin-top:0.25rem;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <div style="font-size:0.76rem;color:var(--text-muted);">
            <b style="color:var(--text);">Live Public APIs Pulled:</b>&nbsp; {api_pills_html}
        </div>
        <span style="font-size:0.73rem;color:var(--text-muted);">EHR Source: {esc(fhir_source_label)}</span>
    </div>
    """
)
with st.expander("📚 Public API Endpoints & Clinical Data Reference Details", expanded=False):
    ref_rows = "".join(
        f'<tr>'
        f'<td><span class="badge badge-blue">{esc(r["stage"])}</span></td>'
        f'<td style="font-weight:600;"><a href="{esc(r["docs_url"])}" target="_blank" rel="noopener noreferrer" style="color:var(--accent);text-decoration:none;">{esc(r["name"])} ↗</a></td>'
        f'<td><span class="badge-mono">{esc(r["endpoint"])}</span></td>'
        f'<td>{esc(r["purpose"])}</td>'
        f'</tr>'
        for r in PUBLIC_API_REFERENCES
    )
    render_html(
        f"""
        <div class="data-table-wrap" style="margin-bottom:0.2rem;">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Workflow Stage</th>
                        <th>Public API / Official Documentation</th>
                        <th>Live REST Endpoint Pulled</th>
                        <th>Clinical Role in Workspace</th>
                    </tr>
                </thead>
                <tbody>{ref_rows}</tbody>
            </table>
        </div>
        """
    )

