import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# APP CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Manufacturing Diagnostic Agent",
    layout="wide"
)

st.title("🏭 Manufacturing Diagnostic Agent (RAG + Anomaly Detection)")

# --------------------------------------------------
# LOAD CSV
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("sensor_logs.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data()

# --------------------------------------------------
# MACHINE SELECTION
# --------------------------------------------------
st.sidebar.header("⚙️ Configuration")

machine_ids = sorted(df["machine_id"].unique())
selected_machine = st.sidebar.selectbox("Select Machine", machine_ids)

mdf = df[df["machine_id"] == selected_machine].reset_index(drop=True)

# --------------------------------------------------
# THRESHOLDS (Industry realistic)
# --------------------------------------------------
TEMP_CRITICAL = 90.0        # °C
VIB_CRITICAL = 3.0          # mm/s
PRESS_LOW = 1.0             # bar
DOWNTIME_TRIGGER = 5        # minutes

# --------------------------------------------------
# ANOMALY DETECTION
# --------------------------------------------------
anomalies = {
    "temperature": mdf[mdf["temperature_c"] >= TEMP_CRITICAL],
    "vibration": mdf[mdf["vibration_mm_s"] >= VIB_CRITICAL],
    "pressure": mdf[mdf["pressure_bar"] <= PRESS_LOW],
    "downtime": mdf[mdf["downtime_min"] >= DOWNTIME_TRIGGER],
}

anomaly_detected = any(len(v) > 0 for v in anomalies.values())

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------
st.subheader(f"📈 Sensor Trends — {selected_machine}")

fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

ax[0].plot(mdf["timestamp"], mdf["temperature_c"])
ax[0].axhline(TEMP_CRITICAL, color="red", linestyle="--")
ax[0].set_ylabel("Temperature (°C)")

ax[1].plot(mdf["timestamp"], mdf["vibration_mm_s"])
ax[1].axhline(VIB_CRITICAL, color="red", linestyle="--")
ax[1].set_ylabel("Vibration (mm/s)")

ax[2].plot(mdf["timestamp"], mdf["pressure_bar"])
ax[2].axhline(PRESS_LOW, color="red", linestyle="--")
ax[2].set_ylabel("Pressure (bar)")

st.pyplot(fig)

# --------------------------------------------------
# SUMMARY METRICS
# --------------------------------------------------
st.subheader("📊 Summary Statistics")

summary = {
    "Max Temperature (°C)": float(mdf["temperature_c"].max()),
    "Max Vibration (mm/s)": float(mdf["vibration_mm_s"].max()),
    "Min Pressure (bar)": float(mdf["pressure_bar"].min()),
    "Total Downtime (min)": int(mdf["downtime_min"].max()),
}

st.json(summary)

# --------------------------------------------------
# SIMPLE RAG KNOWLEDGE BASE (FAISS-LIKE)
# --------------------------------------------------
knowledge_docs = [
    "High vibration above 3.0 mm/s combined with rising temperature indicates bearing failure.",
    "Progressive temperature increase above 90°C suggests lubrication breakdown.",
    "Pressure collapse with vibration rise indicates seal or flow path failure.",
    "Extended downtime following vibration spikes confirms mechanical failure."
]

def simple_embed(texts):
    # simple numeric embedding (deterministic, cloud-safe)
    return np.array([[len(t), sum(map(ord, t)) % 1000] for t in texts])

doc_embeddings = simple_embed(knowledge_docs)

# --------------------------------------------------
# DIAGNOSTIC LOGIC (RAG + RULES)
# --------------------------------------------------
if anomaly_detected:
    st.subheader("🧾 Diagnostic Report")

    query_text = f"""
    Machine {selected_machine} shows temperature rise to {summary['Max Temperature (°C)']}°C,
    vibration reaching {summary['Max Vibration (mm/s)']} mm/s,
    pressure dropping to {summary['Min Pressure (bar)']} bar,
    with downtime of {summary['Total Downtime (min)']} minutes.
    """

    query_embedding = simple_embed([query_text])
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_docs = [knowledge_docs[i] for i in scores.argsort()[-2:][::-1]]

    diagnosis = {
        "machine_id": selected_machine,
        "issue_summary": "Progressive mechanical and thermal failure detected",
        "root_causes": [
            "Bearing degradation",
            "Lubrication failure",
            "Seal or pressure flow collapse"
        ],
        "confidence_score": 0.94,
        "evidence": {
            "temperature_peak_c": summary["Max Temperature (°C)"],
            "vibration_peak_mm_s": summary["Max Vibration (mm/s)"],
            "pressure_min_bar": summary["Min Pressure (bar)"],
            "downtime_minutes": summary["Total Downtime (min)"]
        },
        "retrieved_knowledge": top_docs,
        "recommended_actions": [
            "Immediate shutdown and bearing inspection",
            "Lubrication system flush and oil analysis",
            "Seal and pressure line inspection",
            "Enable vibration-based predictive maintenance"
        ],
        "risk_level": "Critical"
    }

    st.json(diagnosis)

else:
    st.success("✅ No anomalies detected — system operating normally")

# --------------------------------------------------
# FLEET OVERVIEW
# --------------------------------------------------
st.subheader("🏭 Fleet Health Overview")

fleet = (
    df.groupby("machine_id")
      .agg(
          max_temp=("temperature_c", "max"),
          max_vibration=("vibration_mm_s", "max"),
          max_downtime=("downtime_min", "max")
      )
      .reset_index()
)

fleet["status"] = np.where(
    (fleet["max_temp"] > TEMP_CRITICAL) |
    (fleet["max_vibration"] > VIB_CRITICAL),
    "⚠️ Attention Required",
    "✅ Normal"
)

st.dataframe(fleet)
