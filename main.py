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

# ============================================================
# CONFIG
# ============================================================

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable not set")

LLM_API_KEY = os.getenv("LLM_API_KEY")
if not LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY environment variable not set")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")

EMBEDDING_MODEL_NAME = "thenlper/gte-large"
EMBEDDING_DIM = 1024
MAX_FILE_SIZE_MB = 10

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(title="Manufacturing Diagnostic Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DIRECTORIES
# ============================================================

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

# ============================================================
# API KEY MIDDLEWARE
# ============================================================

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if request.url.path.startswith("/docs") or request.url.path.startswith("/openapi.json"):
        return await call_next(request)

    api_key = request.headers.get("x-api-key")
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return await call_next(request)

# ============================================================
# FILE UTILITIES
# ============================================================

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

# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text_from_file(path: Path):
    if path.suffix.lower() == ".pdf":
        reader = PyPDF2.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif path.suffix.lower() in [".txt", ".json"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return df.to_csv(index=False)
    return ""

# ============================================================
# METADATA EXTRACTION
# ============================================================

def extract_machine_id(text: str):
    match = re.search(
        r"(machine[_\-\s]?id|machine)\s*[:=]?\s*([A-Za-z0-9\-]+)",
        text,
        re.IGNORECASE,
    )
    return match.group(2) if match else None

# ============================================================
# CHUNKING
# ============================================================

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

# ============================================================
# VECTOR STORE
# ============================================================

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

# ============================================================
# EMBEDDING PIPELINE
# ============================================================

def process_and_embed(file_path: Path, source_type: str):
    text = extract_text_from_file(file_path)
    machine_id = extract_machine_id(text)
    chunks = chunk_text(text)

    timestamp = datetime.utcnow().isoformat()
    metadata_records = [
        {
            "id": str(uuid.uuid4()),
            "source_type": source_type,
            "machine_id": machine_id,
            "file_name": file_path.name,
            "timestamp": timestamp,
            "content": chunk,
        }
        for chunk in chunks
    ]

    add_embeddings(chunks, metadata_records)

# ============================================================
# UPLOAD ENDPOINTS
# ============================================================

@app.post("/upload/logs")
async def upload_logs(file: UploadFile = File(...)):
    validate_file(file, ["csv", "json"])
    validate_file_size(file)
    dest = LOGS_DIR / file.filename
    save_file(file, dest)
    process_and_embed(dest, "logs")
    return {"status": "success", "file": file.filename}

@app.post("/upload/notes")
async def upload_notes(file: UploadFile = File(...)):
    validate_file(file, ["txt", "pdf"])
    validate_file_size(file)
    dest = NOTES_DIR / file.filename
    save_file(file, dest)
    process_and_embed(dest, "notes")
    return {"status": "success", "file": file.filename}

@app.post("/upload/manuals")
async def upload_manuals(file: UploadFile = File(...)):
    validate_file(file, ["pdf", "txt"])
    validate_file_size(file)
    dest = MANUALS_DIR / file.filename
    save_file(file, dest)
    process_and_embed(dest, "manuals")
    return {"status": "success", "file": file.filename}

@app.post("/upload/incidents")
async def upload_incidents(file: UploadFile = File(...)):
    validate_file(file, ["csv", "pdf", "txt", "json"])
    validate_file_size(file)
    dest = INCIDENTS_DIR / file.filename
    save_file(file, dest)
    process_and_embed(dest, "incidents")
    return {"status": "success", "file": file.filename}

# ============================================================
# RETRIEVAL
# ============================================================

@app.post("/retrieve")
async def retrieve(payload: dict):
    issue_description = payload.get("issue_description")
    if not issue_description:
        raise HTTPException(status_code=400, detail="issue_description is required")

    index, metadata = load_faiss_index()
    if index.ntotal == 0:
        return {"results": []}

    query_embedding = embedding_model.encode(
        [issue_description], normalize_embeddings=True
    )
    distances, indices = index.search(
        np.array(query_embedding).astype("float32"), k=5
    )

    results = []
    for rank, idx in enumerate(indices[0]):
        meta = metadata[idx]
        results.append(
            {
                "source_type": meta["source_type"],
                "machine_id": meta["machine_id"],
                "file_name": meta["file_name"],
                "timestamp": meta["timestamp"],
                "content": meta["content"],
                "score": float(distances[0][rank]),
            }
        )

    return {"results": results}

# ============================================================
# LOG ANALYSIS ENGINE
# ============================================================

class LogAnalysisEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.df.sort_values("timestamp", inplace=True)

    def detect_temperature_spikes(self, z_threshold=2.5):
        temps = self.df["temperature_c"]
        self.df["temp_z"] = (temps - temps.mean()) / temps.std()
        return self.df[self.df["temp_z"] > z_threshold]

    def detect_pressure_drops(self, drop_threshold_pct=10.0):
        self.df["pressure_pct_change"] = (
            self.df["pressure_bar"].pct_change() * 100
        )
        return self.df[self.df["pressure_pct_change"] < -drop_threshold_pct]

    def detect_vibration_anomalies(self):
        vib = self.df["vibration_mm_s"]
        threshold = vib.mean() + 2.5 * vib.std()
        return self.df[vib > threshold]

    def run_all(self):
        return {
            "temperature_spikes": self.detect_temperature_spikes().to_dict("records"),
            "pressure_drops": self.detect_pressure_drops().to_dict("records"),
            "vibration_anomalies": self.detect_vibration_anomalies().to_dict("records"),
        }

# ============================================================
# LLM CLIENT
# ============================================================

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

    response = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ============================================================
# DIAGNOSE ENDPOINT
# ============================================================

@app.post("/diagnose")
async def diagnose(payload: dict):
    df = pd.DataFrame(payload["logs"])
    anomalies = LogAnalysisEngine(df).run_all()

    retrieved_docs = (
        await retrieve({"issue_description": payload.get("issue_description", "")})
    ).get("results", [])

    prompt = f"""
You are an industrial diagnostics AI.

Analyze anomalies and return JSON with:
issue_summary, root_causes, confidence_score, evidence, recommended_actions.

Anomalies:
{json.dumps(anomalies, indent=2)}

Operator Notes:
{payload.get("operator_notes", "")}

Documents:
{json.dumps(retrieved_docs[:3], indent=2)}

Return ONLY valid JSON.
"""

    output = call_llm(prompt)
    try:
        return json.loads(output)
    except Exception:
        return {"raw_output": output, "error": "Invalid JSON from LLM"}
