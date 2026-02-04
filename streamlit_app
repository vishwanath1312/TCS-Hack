import streamlit as st
import requests
import pandas as pd
import json

API_URL = "http://localhost:8000"
API_KEY = st.secrets.get("API_KEY", "")

headers = {
    "x-api-key": API_KEY
}

st.set_page_config(page_title="Manufacturing Diagnostic Agent", layout="wide")
st.title("🏭 Manufacturing Production Issue Diagnostic Agent")

st.sidebar.header("📤 Upload Knowledge Base")

def upload_file(endpoint, label):
    file = st.sidebar.file_uploader(label)
    if file and st.sidebar.button(f"Upload {label}"):
        files = {"file": file}
        res = requests.post(f"{API_URL}{endpoint}", files=files, headers=headers)
        st.sidebar.write(res.json())

upload_file("/upload/logs", "Logs")
upload_file("/upload/notes", "Operator Notes")
upload_file("/upload/manuals", "Maintenance Manuals")
upload_file("/upload/incidents", "Historical Incidents")

st.divider()

st.header("🔍 Diagnose Production Issue")

issue_description = st.text_area("Describe the issue:")
operator_notes = st.text_area("Operator Notes:")

uploaded_log_file = st.file_uploader("Upload sensor logs CSV for analysis")

if uploaded_log_file:
    df = pd.read_csv(uploaded_log_file)
    st.dataframe(df.head())

    if st.button("Run Diagnosis"):
        payload = {
            "logs": df.to_dict(orient="records"),
            "operator_notes": operator_notes,
            "issue_description": issue_description
        }

        res = requests.post(f"{API_URL}/diagnose", json=payload, headers=headers)

        if res.status_code == 200:
            output = res.json()
            st.subheader("🧠 Diagnosis Result")
            st.json(output)
        else:
            st.error(res.text)
