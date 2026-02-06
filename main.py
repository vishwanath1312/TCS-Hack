import streamlit as st
import os
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import faiss
from sentence_transformers import SentenceTransformer

# =====================================================
# CONFIG
# =====================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
APP_API_KEY = os.getenv("APP_API_KEY")

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4.1-mini"

FAISS_INDEX_PATH = "faiss.index"
FAISS_META_PATH = "metadata.json"

EMBED_MODEL = SentenceTransformer("thenlper/gte-large")
EMBED_DIM = 1024

if not OPENAI_API_KEY or not APP_API_KEY:
    st.error("❌ Missing OPENAI_API_KEY or APP_API_KEY in Streamlit Secrets")
    st.stop()

# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(
    page_title="Manufacturing Diagnostic Agent",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Manufacturing Diagnostic Agent")
st.caption("FAISS + RAG + Anomaly Detection + LLM")

# =====================================================
# API KEY PROTECTION
# =====================================================
with st.sidebar:
    st.header("🔐 Access Control")
    user_key = st.text_input("Enter App API Key", type="password")

    if user_key != APP_API_KEY:
        st.warning("Unauthorized access")
        st.stop()

    st.success("Authorized")

# =====================================================
# FAISS UTILITIES
# =====================================================
def load_faiss():
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        metadata = json.load(open(FAISS_META_PATH))
    else:
        index = faiss.IndexFlatL2(EMBED_DIM)
        metadata = []
    return index, metadata


def save_faiss(index, metadata):
    faiss.write_index(index, FAISS_INDEX_PATH)
    json.dump(metadata, open(FAISS_META_PATH, "w"), indent=2)


def add_documents(texts, source):
    index, meta = load_faiss()
    embeddings = EMBED_MODEL.encode(texts, normalize_embeddings=True)
    index.add(np.array(embeddings).astype("float32"))

    for t in texts:
        meta.append({"source": source, "content": t})

    save_faiss(index, meta)


def retrieve_context(query, k=3):
    index, meta = load_faiss()
    if index.ntotal == 0:
        return []

    q_emb = EMBED_MODEL.encode([query], normalize_embeddings=True)
    _, ids = index.search(np.array(q_emb).astype("float32"), k)

    return [meta[i]["content"] for i in ids[0]]

# =====================================================
# SAMPLE KNOWLEDGE INGEST (ONE TIME)
# =====================================================
if not os.path.exists(FAISS_INDEX_PATH):
    seed_docs = [
        "High spindle vibration often indicates bearing wear or imbalance.",
        "Rapid temperature rise suggests lubrication failure or overload.",
        "Pressure drop combined with vibration may indicate hydraulic leaks."
    ]
    add_documents(seed_docs, source="seed_knowledge")

# =====================================================
# INPUTS
# =====================================================
st.subheader("📝 Issue Description")
issue_description = st.text_area(
    "Describe the issue",
    height=120,
    placeholder="Sudden vibration and temperature rise in CNC spindle"
)

st.subheader("👷 Operator Notes")
operator_notes = st.text_area(
    "Optional notes",
    height=100,
    placeholder="Noise before shutdown"
)

st.subheader("📊 Sensor Logs (JSON with machine_id)")
default_logs = [
    {
        "machine_id": "CNC_01",
        "timestamp": "2025-02-01T10:00:00",
        "temperature_c": 70,
        "pressure_bar": 5.2,
        "vibration_mm_s": 1.0,
        "downtime_min": 0
    },
    {
        "machine_id": "CNC_01",
        "timestamp": "2025-02-01T10:10:00",
        "temperature_c": 95,
        "pressure_bar": 4.0,
        "vibration_mm_s": 3.9,
        "downtime_min": 15
    }
]

logs_text = st.text_area(
    "Paste sensor logs",
    value=json.dumps(default_logs, indent=2),
    height=260
)

# =====================================================
# LLM CALL (RATE SAFE)
# =====================================================
def call_llm(prompt, retries=3):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENAI_MODEL,
        "input": prompt
    }

    for i in range(retries):
        r = requests.post(
            f"{OPENAI_BASE_URL}/responses",
            headers=headers,
            json=payload,
            timeout=60
        )

        if r.status_code == 429:
            time.sleep(2 ** i)
            continue

        r.raise_for_status()
        data = r.json()
        return data["output"][0]["content"][0]["text"]

    raise RuntimeError("Rate limit exceeded")

# =====================================================
# RUN DIAGNOSTIC
# =====================================================
if st.button("🧠 Run Diagnostic", type="primary"):
    try:
        logs = json.loads(logs_text)
        df = pd.DataFrame(logs)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        machine_ids = df["machine_id"].unique().tolist()
        selected_machine = st.selectbox("Select Machine", machine_ids)
        machine_df = df[df["machine_id"] == selected_machine]

        # ---------------- CHARTS ----------------
        st.subheader("📊 Sensor Trends")
        st.line_chart(
            machine_df.set_index("timestamp")[["temperature_c", "vibration_mm_s"]]
        )

        # ---------------- ANOMALIES ----------------
        machine_df["temp_z"] = (
            machine_df["temperature_c"] - machine_df["temperature_c"].mean()
        ) / machine_df["temperature_c"].std()

        anomalies = machine_df[machine_df["temp_z"] > 2.5]

        st.subheader("🚨 Detected Anomalies")
        st.dataframe(anomalies)

        # ---------------- RAG ----------------
        context = retrieve_context(issue_description)

        prompt = f"""
You are an industrial diagnostics AI.

Use the following historical knowledge:
{context}

Machine ID: {selected_machine}

Issue Description:
{issue_description}

Operator Notes:
{operator_notes}

Sensor Logs:
{machine_df.to_dict(orient="records")}

Return ONLY valid JSON with:
issue_summary
root_causes
confidence_score
recommended_actions
"""

        with st.spinner("Analyzing data with LLM..."):
            llm_output = call_llm(prompt)

        st.subheader("🧾 Diagnostic Report")
        st.json(json.loads(llm_output))

    except Exception as e:
        st.error(f"❌ Error: {e}")
