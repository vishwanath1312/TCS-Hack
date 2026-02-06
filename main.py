import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv

# --------------------------------------------------
# Load environment variables (local dev only)
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# App Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="LLM Diagnostic Assistant",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Manufacturing Diagnostic Assistant")
st.caption("Streamlit + LLM | GitHub & Streamlit Cloud Ready")

# --------------------------------------------------
# Environment Variables
# --------------------------------------------------
API_KEY = os.getenv("API_KEY")

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")

# --------------------------------------------------
# Basic validation
# --------------------------------------------------
if not API_KEY:
    st.error("❌ API_KEY is missing. Set it in environment variables.")
    st.stop()

if not LLM_API_KEY:
    st.error("❌ LLM_API_KEY is missing. Set it in environment variables.")
    st.stop()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"**Model:** `{LLM_MODEL}`")
    st.write(f"**Endpoint:** `{LLM_BASE_URL}`")
    st.markdown("---")
    st.info("Keys are read securely from environment variables.")

# --------------------------------------------------
# LLM Call Function (Provider-agnostic)
# --------------------------------------------------
def call_llm(prompt: str) -> str:
    """
    Works with:
    - OpenAI compatible APIs
    - Azure / TCS GenAI Lab compatible APIs
    """

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert manufacturing diagnostic assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        return f"❌ Request error: {e}"
    except (KeyError, IndexError):
        return "❌ Unexpected LLM response format."

# --------------------------------------------------
# UI Input
# --------------------------------------------------
st.subheader("🔍 Describe the Problem")

user_input = st.text_area(
    "Enter machine issue, symptoms, or logs:",
    height=160,
    placeholder="Example: CNC machine shows spindle vibration and overheating after 2 hours of operation..."
)

# --------------------------------------------------
# Action Button
# --------------------------------------------------
if st.button("Analyze with LLM 🚀"):
    if not user_input.strip():
        st.warning("Please enter a problem description.")
    else:
        with st.spinner("Analyzing..."):
            prompt = f"""
Analyze the following manufacturing issue and provide:
1. Probable root causes
2. Diagnostic steps
3. Recommended corrective actions

Problem Description:
{user_input}
"""
            result = call_llm(prompt)

        st.subheader("🧾 Diagnostic Report")
        st.markdown(result)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("© 2026 | Streamlit + LLM Diagnostic System")
