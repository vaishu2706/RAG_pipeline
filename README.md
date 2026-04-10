# RAG Full Pipeline

## Stack
- **Backend**: Python, FastAPI, LangChain, OpenAI, ChromaDB, RAGAS
- **Frontend**: React

## Project Structure
```
RAG-full-pipeline/
├── backend/
│   ├── ingest.py       # Steps 1-5: load → split → embed → ChromaDB
│   ├── rag.py          # Steps 6-7: retriever + generator chain
│   ├── evaluate.py     # Step 8: RAGAS evaluation
│   ├── main.py         # FastAPI server
│   ├── requirements.txt
│   └── .env            # Add your OPENAI_API_KEY here
└── frontend/           # React chat UI
```

## Setup

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Add your key to .env
# OPENAI_API_KEY=sk-...

uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm start
```

Open http://localhost:3000

## Usage
1. **Upload** a PDF, DOCX, TXT, or CSV — or paste a URL to ingest web content
2. **Chat** — ask questions; answers include source snippets
3. **Evaluate** — add question + ground truth pairs, run RAGAS to get:
   - `faithfulness` — is the answer grounded in retrieved context?
   - `answer_relevancy` — is the answer relevant to the question?
   - `context_recall` — did retrieval find the right chunks?

## RAG Pipeline Steps
| Step | Component | Detail |
|------|-----------|--------|
| 1 | Preprocessing | File type detection |
| 2 | Document Loaders | PDF, DOCX, TXT, CSV, Web URL |
| 3 | Text Splitter | 300 char chunks, 50 char overlap |
| 4 | Embeddings | `text-embedding-3-small` |
| 5 | Vector Store | ChromaDB (persisted to `./chroma_db`) |
| 6 | Retriever | Similarity search, top-4 chunks |
| 7 | Generator | GPT-4o with custom prompt |
| 8 | Evaluation | RAGAS: faithfulness, relevancy, recall |
