"""
Step 1: Preprocessing
Step 2: Document Loaders  - PDF, DOCX, TXT, CSV, web URLs
Step 3: Text Splitters    - 300 char chunks, 50 char overlap
Step 4: Embeddings        - OpenAI text-embedding-3-small
Step 5: Vector Store      - ChromaDB (persisted)
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = "./chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


def load_document(source: str):
    """Step 2: Load document based on file extension or URL."""
    if source.startswith("http://") or source.startswith("https://"):
        return WebBaseLoader(source).load()
    ext = os.path.splitext(source)[1].lower()
    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
    }
    loader_cls = loaders.get(ext)
    if not loader_cls:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader_cls(source).load()


def split_documents(docs):
    """Step 3: Split into 800-char chunks with 200-char overlap (25% overlap for better boundary coverage)."""
    if not docs or all(not d.page_content.strip() for d in docs):
        raise ValueError("Document appears to be empty or contains no extractable text (possible OCR/scan issue).")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    # Filter out garbled/whitespace-only chunks (malformed PDFs, broken tables)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 20]
    if not chunks:
        raise ValueError("No usable text chunks extracted. The document may be image-based or malformed.")
    return chunks


def get_vector_store():
    """Step 4 + 5: Return ChromaDB vector store with OpenAI embeddings."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


def ingest(source: str):
    """Full ingestion pipeline: load → split → embed → store."""
    docs = load_document(source)
    chunks = split_documents(docs)
    vs = get_vector_store()
    vs.add_documents(chunks)
    return len(chunks)
