from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import os
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import json
import requests
import faiss
from sentence_transformers import SentenceTransformer
import uuid
import PyPDF2
import re

# -------------------- CONFIG --------------------
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable not set")

LLM_API_KEY = os.getenv("LLM_API_KEY")
if not LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY environment variable not set")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://genailab.tcs.in/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "azure_ai/genailab-maas-DeepSeek-V3-0324")

EMBEDDING_MODEL_NAME = "thenlper/gte-large"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
EMBEDDING_DIM = 1024
MAX_FILE_SIZE_MB = 10

# -------------------- APP SETUP --------------------
app = FastAPI(title="Manufacturing Diagnostic Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("data")
LOGS_DIR = DATA_DIR / "logs"
NOTES_DIR = DATA_DIR / "notes"
MANUALS_DIR = DATA_DIR / "manuals"
INCIDENTS_DIR = DATA_DIR / "incidents"
VECTOR_DIR = DATA_DIR / "vector_store"

for d in [LOGS_DIR, NOTES_DIR, MANUALS_DIR, INCIDENTS_DIR, VECTOR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

INDEX_PATH = VECTOR_DIR / "faiss.index"
META_PATH = VECTOR_DIR / "metadata.json"

# -------------------- API KEY MIDDLEWARE --------------------
@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if request.url.path.startswith("/docs") or request.url.path.startswith("/openapi.json"):
        return await call_next(request)

    api_key = request.headers.get("x-api-key")
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return await call_next(request)

# -------------------- FILE UTILS --------------------
def validate_file(file: UploadFile, allowed_extensions: list):
    ext = file.filename.split(".")[-1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid file type")

def validate_file_size(file: UploadFile):
    file.file.seek(0, os.SEEK_END)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

def save_file(upload_file: UploadFile, destination: Path):
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

# -------------------- TEXT EXTRACTION --------------------
def extract_text_from_file(path: Path):
    if path.suffix.lower() == ".pdf":
        reader = PyPDF2.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif path.suffix.lower() in [".txt", ".json"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return df.to_csv(index=False)
    else:
        return ""

# -------------------- METADATA EXTRACTION --------------------
def extract_machine_id(text: str):
    match = re.search(r"(machine[_\-\s]?id|machine)\s*[:=]?\s*([A-Za-z0-9\-]+)", text, re.IGNORECASE)
    if match:
        return match.group(2)
    return None

# -------------------- CHUNKING --------------------
def chunk_text(text: str, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

# -------------------- VECTOR STORE --------------------
def load_faiss_index():
    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "r") as f:
            metadata = json.load(f)
    else:
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        metadata = []
    return index, metadata

def save_faiss_index(index, metadata):
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

def add_embeddings(chunks, metadata_records):
    index, metadata = load_faiss_index()
    embeddings = embedding_model.encode(chunks, normalize_embeddings=True)
    index.add(np.array(embeddings).astype("float32"))
    metadata.extend(metadata_records)
    save_faiss_index(index, metadata)

# -------------------- EMBEDDING PIPELINE --------------------
def process_and_embed(file_path: Path, source_type: str):
    text = extract_text_from_file(file_path)
    machine_id = extract_machine_id(text)
    chunks = chunk_text(text)

    metadata_records = []
    timestamp = datetime.utcnow().isoformat()

    for chunk in chunks:
        metadata_records.append({
            "id": str(uuid.uuid4()),
            "source_type": source_type,
            "machine_id": machine_id,
            "file_name": file_path.name,
            "timestamp": timestamp,
            "content": chunk
        })

    add_embeddings(chunks, metadata_records)

# -------------------- UPLOAD ENDPOINTS --------------------
@app.post("/upload/logs")
async def upload_logs(file: UploadFile = File(...)):
    validate_file(file, ["csv", "json"])
    validate_file_size(file)
    dest = LOGS_DIR / file.filename
    save_file(file, dest)
    process_and_embed(dest, source_type="logs")
    return {"status": "success", "file": file.filename, "category": "logs"}

@app.post("/upload/notes")
async def upload_notes(file: UploadFile = File(...)):
    validate_file(file, ["txt", "pdf"])
    validate_file_size(file)
    dest = NOTES_DIR / file.filename
    save_file(file, dest)
    process_and_embed(dest, source_type="notes")
    return {"status": "success", "file": file.filename, "category": "notes"}

@app.post("/upload/manuals")
async def upload_manuals(file: UploadFile = File(...)):
    validate_file(file, ["pdf", "txt"])
    validate_file_size(file)
    dest = MANUALS_DIR / file.filename
    save_file(file, dest)
    process_and_embed(dest, source_type="manuals")
    return {"status": "success", "file": file.filename, "category": "manuals"}

@app.post("/upload/incidents")
async def upload_incidents(file: UploadFile = File(...)):
    validate_file(file, ["csv", "pdf", "txt", "json"])
    validate_file_size(file)
    dest = INCIDENTS_DIR / file.filename
    save_file(file, dest)
    process_and_embed(dest, source_type="incidents")
    return {"status": "success", "file": file.filename, "category": "incidents"}

# -------------------- RETRIEVAL API --------------------
@app.post("/retrieve")
async def retrieve(payload: dict):
    issue_description = payload.get("issue_description")
    if not issue_description:
        raise HTTPException(status_code=400, detail="issue_description is required")

    index, metadata = load_faiss_index()
    if index.ntotal == 0:
        return {"results": []}

    query_embedding = embedding_model.encode([issue_description], normalize_embeddings=True)
    distances, indices = index.search(np.array(query_embedding).astype("float32"), k=5)

    results = []
    for rank, idx in enumerate(indices[0]):
        meta = metadata[idx]
        results.append({
            "source_type": meta.get("source_type"),
            "machine_id": meta.get("machine_id"),
            "file_name": meta.get("file_name"),
            "timestamp": meta.get("timestamp"),
            "content": meta.get("content"),
            "score": float(distances[0][rank])
        })

    return {"results": results}

# -------------------- LOG ANALYSIS ENGINE --------------------
class LogAnalysisEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.df = self.df.sort_values("timestamp")

    def detect_temperature_spikes(self, z_threshold: float = 2.5):
        temps = self.df["temperature_c"]
        mean = temps.mean()
        std = temps.std()
        self.df["temp_z"] = (temps - mean) / std
        return self.df[self.df["temp_z"] > z_threshold]

    def detect_pressure_drops(self, drop_threshold_pct: float = 10.0):
        self.df["pressure_pct_change"] = self.df["pressure_bar"].pct_change() * 100
        return self.df[self.df["pressure_pct_change"] < -drop_threshold_pct]

    def detect_vibration_anomalies(self, vibration_threshold: float = None):
        vib = self.df["vibration_mm_s"]
        if vibration_threshold is None:
            vibration_threshold = vib.mean() + 2.5 * vib.std()
        return self.df[self.df["vibration_mm_s"] > vibration_threshold]

    def detect_downtime_clusters(self, min_cluster_size: int = 2, window_minutes: int = 60):
        downtime_events = self.df[self.df["downtime_min"] > 0]
        clusters = []
        current_cluster = []

        for _, row in downtime_events.iterrows():
            if not current_cluster:
                current_cluster.append(row)
            else:
                last_time = current_cluster[-1]["timestamp"]
                if row["timestamp"] - last_time <= timedelta(minutes=window_minutes):
                    current_cluster.append(row)
                else:
                    if len(current_cluster) >= min_cluster_size:
                        clusters.append(current_cluster)
                    current_cluster = [row]

        if len(current_cluster) >= min_cluster_size:
            clusters.append(current_cluster)

        return clusters

    def run_all(self):
        return {
            "temperature_spikes": self.detect_temperature_spikes().to_dict(orient="records"),
            "pressure_drops": self.detect_pressure_drops().to_dict(orient="records"),
            "vibration_anomalies": self.detect_vibration_anomalies().to_dict(orient="records"),
            "downtime_clusters": [
                [row.to_dict() for row in cluster] for cluster in self.detect_downtime_clusters()
            ],
        }

# -------------------- PROMPT BUILDER --------------------
def build_diagnostic_prompt(anomalies, operator_notes, retrieved_docs):
    prompt = f"""
You are an industrial diagnostics AI specializing in manufacturing production failures.

Analyze the following data and return a structured JSON with:
- issue_summary
- root_causes
- confidence_score
- evidence
- recommended_actions

--- SENSOR ANOMALIES ---
Temperature Spikes:
{json.dumps(anomalies.get("temperature_spikes", []), indent=2)}

Pressure Drops:
{json.dumps(anomalies.get("pressure_drops", []), indent=2)}

Vibration Anomalies:
{json.dumps(anomalies.get("vibration_anomalies", []), indent=2)}

Downtime Clusters:
{json.dumps(anomalies.get("downtime_clusters", []), indent=2)}

--- OPERATOR NOTES ---
{operator_notes}

--- HISTORICAL INCIDENTS & MANUAL EXCERPTS ---
"""
    for doc in retrieved_docs:
        prompt += f"\nSource: {doc.get('source_type')} | File: {doc.get('file_name')} | Machine: {doc.get('machine_id')}\n"
        prompt += doc.get("content", "")[:1200] + "\n"

    prompt += "\nReturn only valid JSON."
    return prompt.strip()

# -------------------- LLM CLIENT --------------------
def call_llm(prompt: str):
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
    }
    response = requests.post(f"{LLM_BASE_URL}/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# -------------------- DIAGNOSE ENDPOINT --------------------
@app.post("/diagnose")
async def diagnose(payload: dict):
    df = pd.DataFrame(payload["logs"])
    analyzer = LogAnalysisEngine(df)
    anomalies = analyzer.run_all()

    retrieval_results = await retrieve({"issue_description": payload.get("issue_description", "")})
    retrieved_docs = retrieval_results.get("results", [])

    prompt = build_diagnostic_prompt(
        anomalies=anomalies,
        operator_notes=payload.get("operator_notes", ""),
        retrieved_docs=retrieved_docs,
    )

    llm_output = call_llm(prompt)

    try:
        return json.loads(llm_output)
    except Exception:
        return {"raw_output": llm_output, "error": "LLM did not return valid JSON"}
