"""
rag.py - Answer questions from the BioRAG knowledge base, with citations.

How it works:
  1. Retrieve the most relevant abstract chunks from the FAISS index.
  2. Ask the LLM (Google Gemini) to answer ONLY from those chunks and cite PubMed IDs.
  3. If the chunks don't contain the answer, return a refusal instead of guessing.

Try it:  python rag.py "What drives resistance to enzalutamide?"
"""

import os
import sys

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

INDEX_DIR = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # must match ingest.py
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")   # free-tier model; check AI Studio for current names
TOP_K = 4                                                 # number of chunks to retrieve
REFUSAL = "I don't have enough evidence in the indexed abstracts to answer that."

SYSTEM_PROMPT = f"""You are a careful biomedical research assistant.
Answer the question using ONLY the numbered sources provided.
Rules:
- Cite every claim with the PubMed ID in square brackets, e.g. [PMID: 12345678].
- Do not use outside knowledge, and do not invent PubMed IDs.
- If the sources do not contain the answer, reply exactly: "{REFUSAL}"
- Be concise: 3-6 sentences."""

# Load the index and model once, so repeated questions are fast.
_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
# allow_dangerous_deserialization is safe here because we load an index we built ourselves.
_index = FAISS.load_local(INDEX_DIR, _embeddings, allow_dangerous_deserialization=True)
_llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=os.getenv("LLM_API_KEY"),
    temperature=0,      # deterministic answers make evaluation repeatable
)

def answer(question: str) -> dict:
    """Return {'answer': str, 'sources': [{'pmid', 'title', 'year'}, ...]}."""
    chunks = _index.similarity_search(question, k=TOP_K)

    context = "\n\n".join(
        f"Source {i + 1} [PMID: {c.metadata['pmid']}] ({c.metadata['year']})\n{c.page_content}"
        for i, c in enumerate(chunks)
    )
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Sources:\n{context}\n\nQuestion: {question}"),
    ]
    reply = _llm.invoke(messages).content
    if isinstance(reply, list):  # some models return a list of content parts
        reply = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in reply)
    reply = reply.strip()

    # Report each retrieved paper once, in retrieval order.
    seen, sources = set(), []
    for c in chunks:
        pmid = c.metadata["pmid"]
        if pmid not in seen:
            seen.add(pmid)
            sources.append({"pmid": pmid, "title": c.metadata["title"], "year": c.metadata["year"]})

    return {"answer": reply, "refused": reply.startswith(REFUSAL[:30]), "sources": sources}

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What are the main mechanisms of resistance to androgen receptor-targeted therapy?"
    result = answer(q)
    print(f"\nQ: {q}\n\nA: {result['answer']}\n\nRetrieved sources:")
    for s in result["sources"]:
        print(f"  - PMID {s['pmid']} ({s['year']}): {s['title']}")
