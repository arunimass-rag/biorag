"""
ingest.py - Build the BioRAG knowledge base.

Steps:
  1. Get PubMed IDs (from data/pmids.txt, or by searching PubMed if that file doesn't exist).
  2. Download each paper's title and abstract from PubMed (NCBI Entrez).
  3. Split abstracts into chunks, embed them with a free local model, and save a FAISS index.

Run:  python ingest.py
"""

import json
import os
import time
from pathlib import Path

from Bio import Entrez, Medline
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- Settings ----------
QUERY = (
    '("androgen receptor"[Title/Abstract]) AND ("prostate cancer"[Title/Abstract]) '
    "AND hasabstract AND 2018:2026[dp]"
)
MAX_PAPERS = 200
PMID_FILE = Path("data/pmids.txt")             # committed to GitHub (IDs only)
ABSTRACT_FILE = Path("data/abstracts/abstracts.jsonl")  # ignored by .gitignore
INDEX_DIR = "faiss_index"                       # ignored by .gitignore
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

load_dotenv()
Entrez.email = os.getenv("NCBI_EMAIL")          # NCBI requires an email address
if not Entrez.email:
    raise SystemExit("Add NCBI_EMAIL=your@email.com to your .env file first.")

def get_pmids() -> list[str]:
    """Read PubMed IDs from data/pmids.txt, or search PubMed and save them there."""
    if PMID_FILE.exists():
        pmids = [line.strip() for line in PMID_FILE.read_text().splitlines() if line.strip()]
        print(f"Loaded {len(pmids)} PubMed IDs from {PMID_FILE}")
        return pmids

    print("Searching PubMed...")
    with Entrez.esearch(db="pubmed", term=QUERY, retmax=MAX_PAPERS, sort="relevance") as handle:
        pmids = Entrez.read(handle)["IdList"]
    PMID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PMID_FILE.write_text("\n".join(pmids) + "\n")
    print(f"Saved {len(pmids)} PubMed IDs to {PMID_FILE}")
    return pmids

def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """Download title + abstract for each PubMed ID, in batches of 50."""
    records = []
    for start in range(0, len(pmids), 50):
        batch = pmids[start:start + 50]
        with Entrez.efetch(db="pubmed", id=",".join(batch), rettype="medline", retmode="text") as handle:
            for rec in Medline.parse(handle):
                if rec.get("AB"):  # skip papers without an abstract
                    records.append({
                        "pmid": rec.get("PMID", ""),
                        "title": rec.get("TI", ""),
                        "journal": rec.get("TA", ""),
                        "year": rec.get("DP", "")[:4],
                        "abstract": rec["AB"],
                    })
        print(f"  fetched {min(start + 50, len(pmids))}/{len(pmids)}")
        time.sleep(0.4)  # stay under NCBI's limit of 3 requests per second
    ABSTRACT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with ABSTRACT_FILE.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} abstracts to {ABSTRACT_FILE}")
    return records

def build_index(records: list[dict]) -> None:
    """Chunk the abstracts, embed them, and save a FAISS index."""
    docs = [
        Document(
            page_content=f"{r['title']}\n\n{r['abstract']}",
            metadata={"pmid": r["pmid"], "title": r["title"], "journal": r["journal"], "year": r["year"]},
        )
        for r in records
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"Split {len(docs)} abstracts into {len(chunks)} chunks")

    print("Embedding chunks (first run downloads the model, about 90 MB)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(INDEX_DIR)
    print(f"Saved FAISS index to {INDEX_DIR}/")

if __name__ == "__main__":
    pmids = get_pmids()
    records = fetch_abstracts(pmids)
    build_index(records)
    print("Done.")
