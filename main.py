import streamlit as st
import os
import requests
import json
from datetime import datetime

# ===============================
# CONFIG
# ===============================
APP_API_KEY = os.getenv("APP_API_KEY")          # Your app-level auth
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")    # OpenAI key

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4.1-mini"

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not set")
    st.stop()

if not APP_API_KEY:
    st.error("❌ APP_API_KEY not set")
    st.stop()

# ===============================
# PAGE SETUP
# ===============================
st.set_page_config(
    page_title="Manufacturing Diagnostic Agent",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Manufacturing Diagnostic Agent")
st.caption("Streamlit + OpenAI (Responses API)")

# ===============================
# SIMPLE API-KEY PROTECTION
# ===============================
with st.sidebar:
    st.header("🔐 Access Control")
    user_key = st.text_input("Enter App API Key", type="password")

    if user_key != APP_API_KEY:
        st.warning("Unauthorized")
        st.stop()

    st.success("Authorized ✅")

# ===============================
# INPUTS
# ===============================
st.subheader("📝 Issue Description")
issue_description = st.text_area(
    "Describe the machine issue",
    placeholder="Sudden temperature rise and vibration in spindle motor",
    height=120
)

st.subheader("👷 Operator Notes")
operator_notes = st.text_area(
    "Optional notes",
    placeholder="Noise observed before shutdown",
    height=100
)

st.subheader("📊 Sample Sensor Logs (JSON)")
default_logs = [
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
    "Logs",
    value=json.dumps(default_logs, indent=2),
    height=220
)

# ===============================
# LLM CALL (NEW RESPONSES API)
# ===============================
def call_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENAI_MODEL,
        "input": prompt
    }

    response = requests.post(
        f"{OPENAI_BASE_URL}/responses",
        headers=headers,
        json=payload,
        timeout=60
    )

    response.raise_for_status()
    data = response.json()

    return data["output"][0]["content"][0]["text"]

# ===============================
# DIAGNOSE
# ===============================
if st.button("🧪 Run Diagnostic", type="primary"):
    try:
        logs = json.loads(logs_text)

        prompt = f"""
You are an industrial diagnostics AI.

Analyze the following manufacturing issue and return VALID JSON with:
- issue_summary
- probable_root_causes
- confidence_score (0–1)
- recommended_actions

ISSUE:
{issue_description}

OPERATOR NOTES:
{operator_notes}

SENSOR LOGS:
{json.dumps(logs, indent=2)}

Return only JSON.
"""

        with st.spinner("Analyzing..."):
            result = call_llm(prompt)

        st.subheader("🧾 Diagnostic Report")
        st.json(json.loads(result))

    except json.JSONDecodeError:
        st.error("❌ Logs or LLM output is not valid JSON")
        st.text(result)

    except requests.HTTPError as e:
        st.error(f"❌ LLM Request Error: {e}")

    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")
