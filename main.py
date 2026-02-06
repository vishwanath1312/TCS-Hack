import streamlit as st
import os
import requests
import json
import time
from datetime import datetime

# =====================================
# CONFIG
# =====================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
APP_API_KEY = os.getenv("APP_API_KEY")

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4.1-mini"

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not set in Streamlit secrets")
    st.stop()

if not APP_API_KEY:
    st.error("❌ APP_API_KEY not set in Streamlit secrets")
    st.stop()

# =====================================
# PAGE SETUP
# =====================================
st.set_page_config(
    page_title="Manufacturing Diagnostic Agent",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Manufacturing Diagnostic Agent")
st.caption("Streamlit + OpenAI (Responses API) | Production-safe")

# =====================================
# USER API-KEY PROTECTION
# =====================================
with st.sidebar:
    st.header("🔐 Access Control")
    user_key = st.text_input("Enter App API Key", type="password")

    if user_key != APP_API_KEY:
        st.warning("Unauthorized access")
        st.stop()

    st.success("Authorized ✅")

# =====================================
# SESSION STATE INIT
# =====================================
if "llm_result" not in st.session_state:
    st.session_state.llm_result = None

if "last_call_time" not in st.session_state:
    st.session_state.last_call_time = 0.0

# =====================================
# INPUTS
# =====================================
st.subheader("📝 Issue Description")
issue_description = st.text_area(
    "Describe the manufacturing issue",
    placeholder="Sudden temperature rise and vibration in spindle motor",
    height=120
)

st.subheader("👷 Operator Notes")
operator_notes = st.text_area(
    "Optional notes",
    placeholder="Noise observed before emergency shutdown",
    height=100
)

st.subheader("📊 Sensor Logs (JSON)")
sample_logs = [
    {
        "timestamp": "2025-02-01T10:00:00",
        "temperature_c": 72,
        "pressure_bar": 5.2,
        "vibration_mm_s": 1.1,
        "downtime_min": 0
    },
    {
        "timestamp": "2025-02-01T10:10:00",
        "temperature_c": 94,
        "pressure_bar": 4.1,
        "vibration_mm_s": 3.8,
        "downtime_min": 12
    }
]

logs_text = st.text_area(
    "Paste sensor logs",
    value=json.dumps(sample_logs, indent=2),
    height=240
)

# =====================================
# LLM CALL WITH RATE-LIMIT SAFETY
# =====================================
def call_llm(prompt: str, retries: int = 3) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENAI_MODEL,
        "input": prompt
    }

    for attempt in range(retries):
        response = requests.post(
            f"{OPENAI_BASE_URL}/responses",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 429:
            wait_time = 2 ** attempt
            time.sleep(wait_time)
            continue

        response.raise_for_status()
        data = response.json()
        return data["output"][0]["content"][0]["text"]

    raise RuntimeError("Rate limit exceeded. Please try again later.")

# =====================================
# RUN DIAGNOSTIC
# =====================================
if st.button("🧪 Run Diagnostic", type="primary"):

    # ---- Cooldown protection (10 seconds)
    now = time.time()
    if now - st.session_state.last_call_time < 10:
        st.warning("⏳ Please wait 10 seconds before retrying")
        st.stop()

    st.session_state.last_call_time = now
    st.session_state.llm_result = None

    try:
        logs = json.loads(logs_text)

        prompt = f"""
Return ONLY valid JSON with:
issue_summary
probable_root_causes
confidence_score
recommended_actions

Issue: {issue_description}

Operator Notes: {operator_notes}

Sensor Logs:
{json.dumps(logs)}
"""

        with st.spinner("Analyzing manufacturing data..."):
            st.session_state.llm_result = call_llm(prompt)

    except json.JSONDecodeError:
        st.error("❌ Sensor logs must be valid JSON")

    except requests.HTTPError as e:
        st.error(f"❌ LLM Request Error: {e}")

    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")

# =====================================
# OUTPUT
# =====================================
if st.session_state.llm_result:
    st.subheader("🧾 Diagnostic Report")
    try:
        st.json(json.loads(st.session_state.llm_result))
    except Exception:
        st.text(st.session_state.llm_result)
