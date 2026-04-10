"""
Step 6: Retriever  - Hybrid search (BM25 + semantic) + Cohere Rerank
Step 7: Generator  - LangChain RAG chain with OpenAI GPT-4o

Hybrid search addresses:
- Needle-in-a-haystack: BM25 catches exact keyword matches embeddings miss
- Ambiguous queries: ensemble scoring reduces single-model bias
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from ingest import get_vector_store

load_dotenv()

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use ONLY the context below to answer.
If the answer is not in the context, say exactly: "I don't have enough information in the provided documents."
Do NOT use your training knowledge. Do NOT make up facts.
Be concise and direct. If the question has multiple parts, address each part separately.

Context:
{context}

Question: {question}
Answer:""",
)


def build_rag_chain():
    """Step 6 + 7: Hybrid retrieval → Cohere rerank → GPT-4o."""
    vs = get_vector_store()
    collection_size = vs._collection.count()

    k = max(1, min(20, collection_size))
    print(f"Collection has {collection_size} chunks. Using k={k} for retrieval and top_n={max(1, min(6, k))} for reranking.")
    top_n = max(1, min(6, k))

    # Semantic retriever
    semantic_retriever = vs.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )

    # BM25 (keyword) retriever — built from all stored chunks
    all_docs = vs.get()
    if all_docs and all_docs.get("documents"):
        from langchain_core.documents import Document
        docs_for_bm25 = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
        ]
        bm25_retriever = BM25Retriever.from_documents(docs_for_bm25, k=k)
        # Ensemble: 50% semantic + 50% BM25
        base_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )
    else:
        base_retriever = semantic_retriever

    reranker = CohereRerank(model="rerank-english-v3.0", top_n=top_n)
    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )


def query(question: str) -> dict:
    """Run the RAG pipeline. Guardrail checks are handled in main.py."""
    vs = get_vector_store()
    if vs._collection.count() == 0:
        return {
            "answer": "No documents have been ingested yet. Please upload a file first.",
            "sources": [],
        }

    chain = build_rag_chain()
    result = chain.invoke({"query": question})

    sources = [
        {"source": doc.metadata.get("source", ""), "content": doc.page_content}
        for doc in result["source_documents"]
    ]

    # Flag when retrieval returned nothing useful (hallucination risk)
    if not result["source_documents"]:
        return {
            "answer": "I don't have enough information in the provided documents.",
            "sources": [],
            "warning": "No relevant chunks were retrieved for this query.",
        }

    return {"answer": result["result"], "sources": sources}
