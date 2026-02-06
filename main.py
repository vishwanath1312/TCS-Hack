import os
import time
import json
import faiss
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from openai import OpenAI

# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="Manufacturing Diagnostics AI", layout="wide")

LLM_MODEL = "gpt-4.1-mini"
client = OpenAI(api_key=os.getenv("LLM_API_KEY"))

FEATURES = ["temperature", "vibration", "pressure", "rpm"]

# ==========================
# UTILITIES
# ==========================
def anomaly_confidence(score):
    return min(1.0, abs(score))

def create_pdf(report, filename="diagnostic_report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    story = []

    for k, v in report.items():
        story.append(Paragraph(f"<b>{k}</b>: {v}", styles["Normal"]))

    doc.build(story)
    return filename

# ==========================
# UI
# ==========================
st.title("🏭 Manufacturing Diagnostics AI (RAG + LLM)")

uploaded_file = st.file_uploader("📂 Upload Sensor CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV Loaded")

    st.dataframe(df.head())

    # ==========================
    # MODEL TRAINING
    # ==========================
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])

    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(X)

    df["anomaly_score"] = iso_forest.decision_function(X)
    df["anomaly"] = iso_forest.predict(X)

    # ==========================
    # FAISS INDEX
    # ==========================
    faiss_index = faiss.IndexFlatL2(X.shape[1])
    faiss_index.add(X.astype("float32"))

    st.subheader("📊 Anomaly Overview")
    st.line_chart(df.groupby("machine_id")["anomaly_score"].mean())

    # ==========================
    # REAL-TIME USER INPUT
    # ==========================
    st.sidebar.header("🔮 Live Prediction")

    with st.sidebar.form("predict_form"):
        machine_id = st.text_input("Machine ID", "M-001")
        temperature = st.number_input("Temperature", 0.0, 200.0, 85.0)
        vibration = st.number_input("Vibration", 0.0, 1.0, 0.05)
        pressure = st.number_input("Pressure", 0.0, 20.0, 5.2)
        rpm = st.number_input("RPM", 0, 5000, 1450)

        simulate_stream = st.checkbox("Simulate real-time stream")
        submit = st.form_submit_button("Predict")

    if submit:
        user_df = pd.DataFrame([{
            "machine_id": machine_id,
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "rpm": rpm
        }])

        # ==========================
        # STREAMING SIMULATION
        # ==========================
        if simulate_stream:
            st.info("Streaming input...")
            for _ in range(3):
                time.sleep(0.5)
                st.write("Receiving sensor packet...")

        # ==========================
        # PREDICTION
        # ==========================
        X_user = scaler.transform(user_df[FEATURES])
        score = iso_forest.decision_function(X_user)[0]
        is_anomaly = iso_forest.predict(X_user)[0] == -1
        confidence = anomaly_confidence(score)

        # ==========================
        # FAISS RAG
        # ==========================
        D, I = faiss_index.search(X_user.astype("float32"), 3)
        context = df.iloc[I[0]].to_dict(orient="records")

        # ==========================
        # LLM DIAGNOSTICS
        # ==========================
        prompt = f"""
You are an industrial diagnostics expert.

Sensor Input:
{user_df.to_json(orient="records")}

Anomaly: {is_anomaly}
Score: {score}
Confidence: {confidence}

Similar cases:
{json.dumps(context)}

Return STRICT JSON with:
status, severity, root_cause, recommended_action
"""

        response = client.responses.create(
            model=LLM_MODEL,
            input=prompt
        )

        diagnosis = response.output_text

        # ==========================
        # FINAL RESULT
        # ==========================
        result = {
            "machine_id": machine_id,
            "anomaly": bool(is_anomaly),
            "anomaly_score": round(float(score), 4),
            "confidence": round(confidence, 3),
            "diagnosis": diagnosis
        }

        st.subheader("🧾 Prediction Result")
        st.json(result)

        # ==========================
        # EXPORT
        # ==========================
        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(result, indent=2),
            file_name="prediction.json"
        )

        pdf_file = create_pdf(result)
        with open(pdf_file, "rb") as f:
            st.download_button(
                "⬇️ Download PDF",
                data=f,
                file_name="prediction.pdf"
            )
