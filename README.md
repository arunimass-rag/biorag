# biorag
Retrieval-augmented question answering over PubMed abstracts using LangChain, FAISS and FastAPI, with cited sources and grounding evaluation.
BioRAG

Grounded question answering over biomedical literature. BioRAG is a retrieval-augmented generation (RAG) service that answers biology questions using a curated set of PubMed abstracts. Every answer cites the PubMed IDs it relied on, and the service declines to answer when the retrieved evidence is insufficient.

Status: 🚧 In progress. See the roadmap below.

Why this project

Large language models can produce fluent but unsupported answers. In earlier work designing LLM evaluation tasks from my own experimental data, I found that a general model could reach correct answers by recalling a publication rather than reasoning from the data. BioRAG explores the opposite approach: answers must be grounded in retrieved evidence, and that grounding is measured.

How it works
Question ──▶ FastAPI /ask ──▶ FAISS retrieval (top-k abstract chunks)
                                     │
                                     ▼
                     LLM answers ONLY from retrieved text
                                     │
                                     ▼
              Answer + cited PubMed IDs  (or "not enough evidence")
Ingestion: abstracts are split into chunks and embedded with a sentence-transformers model, then stored in a FAISS index.
Retrieval and generation: LangChain retrieves the most relevant chunks and prompts the LLM to answer only from them, with citations.
API: a FastAPI microservice exposes POST /ask.
Tech stack

Python · LangChain · FAISS · sentence-transformers · FastAPI · Docker · AWS (planned)

Getting started
bash
git clone https://github.com/<your-username>/biorag.git
cd biorag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then add your own LLM API key and service API key
python ingest.py            # downloads abstracts by PubMed ID and builds the FAISS index
uvicorn app:app --reload    # open http://localhost:8000/docs to try it

Data note: abstracts are not stored in this repository. ingest.py fetches them from PubMed using the IDs listed in data/pmids.txt.

Evaluation (responsible AI)

A 20-question test set (eval.json) checks:

Metric	What it measures
Accuracy	Correct answers on 15 questions covered by the abstracts
Grounding	Share of answers that cite a retrieved source
Refusal	Share of 5 out-of-scope questions answered with "not enough evidence"

Results will be added here once the evaluation runs.

Roadmap
[ ] Ingestion and FAISS index
[ ] RAG pipeline with citations and refusal rule
[ ] FastAPI /ask endpoint with API-key authentication
[ ] Evaluation script and results
[ ] Request logging to DynamoDB
[ ] Secrets in AWS Secrets Manager; least-privilege IAM role
[ ] Docker image and deployment on AWS App Runner
Security
No secrets are committed; configuration is loaded from environment variables (see .env.example).
The API requires an x-api-key header.
Author

Arunima Sikdar, Ph.D. · LinkedIn · Structural biologist and data scientist working on LLM evaluation and responsible AI.
