from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ingest import ingest
from rag import query
from evaluate import run_evaluation
from guardrails import check_input, check_output, GuardrailViolation

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.csv'}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

app = FastAPI(title="RAG Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Ingest an uploaded file into the vector store."""
    # Validate extension (server-side, not just frontend)
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Sanitize filename to prevent path traversal
    safe_name = Path(file.filename).name
    path = UPLOAD_DIR / safe_name

    # Enforce file size limit
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 20 MB limit")

    path.write_bytes(contents)
    try:
        count = ingest(str(path))
        return {"message": f"Ingested {count} chunks from {safe_name}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        path.unlink(missing_ok=True)  # Clean up after ingestion


class URLRequest(BaseModel):
    url: str


@app.post("/ingest-url")
def ingest_url(req: URLRequest):
    """Ingest content from a web URL."""
    parsed = urlparse(req.url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Invalid URL")
    try:
        count = ingest(req.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": f"Ingested {count} chunks from {req.url}"}


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def ask(req: QueryRequest):
    """Query the RAG pipeline."""
    try:
        safe_question = check_input(req.question)
    except GuardrailViolation as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        result = query(safe_question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # Output guardrail — check for hallucination signals and conflicting sources
    output_flags = check_output(result["answer"], result.get("sources", []))
    return {**result, **output_flags}


class EvalSample(BaseModel):
    question: str
    ground_truth: str


@app.post("/evaluate")
def evaluate_rag(samples: list[EvalSample]):
    """Run RAGAS evaluation on provided Q&A pairs."""
    valid = [s.model_dump() for s in samples if s.question.strip() and s.ground_truth.strip()]
    if not valid:
        raise HTTPException(status_code=400, detail="Provide at least one valid question + ground truth pair")
    try:
        results = run_evaluation(valid)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"results": results}
