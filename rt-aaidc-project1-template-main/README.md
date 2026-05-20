# OmniRAG: A Modular Multi-Provider RAG Assistant with ChromaDB

OmniRAG is a production-grade, local Retrieval-Augmented Generation (RAG) framework designed to ingest, chunk, embed, and query multi-domain document corpuses. Built with a highly modular architecture, the system isolates orchestration logic to allow seamless hot-swapping between prominent LLM providers (OpenAI, Groq, Google Gemini) while utilizing a persistent local vector database for semantic context grounding.

---

## 🚀 Core Features

* **Multi-Provider LLM Orchestration:** Dynamic client switching across major model providers via standardized environment interfaces.
* **Persistent Vector Semantics:** Document grounding utilizing local ChromaDB instances and optimized vector indexing.
* **Deterministic Chunking:** Explicit token-constrained structural parsing to maximize semantic density and minimize context dilution.
* **Zero-Hallucination Guardrails:** Explicit prompt engineering and source attribution to guarantee strict factual grounding.
* **Robust OS Execution:** Optimized for smooth file path management and terminal interaction on Windows/DOS environments.

---

## 📂 Project Architecture

The core codebase is structured to preserve isolation between the ingestion pipeline, data store, and runtime execution loops:

* [`src/`](./rt-aaidc-project1-template-main/src) — Contains core execution mechanics.
  * `app.py` — Main CLI runtime, environment configuration loading (`python-dotenv`), prompt engineering pipelines, and interactive loop logic.
  * `vectordb.py` — Database initialization, collection configurations, and context retrieval algorithms.
* [`data/`](./rt-aaidc-project1-template-main/data) — Local target folder housing text corpus partitions including:
  * Domain datasets: `quantum_computing.txt`, `biotechnology.txt`, `climate_science.txt`, `artificial_intelligence.txt`.
  * Experimental data: `my_test_files.txt` (used for pipeline synchronization and factual retrieval verification).
* `requirements.txt` — Frozen dependency declarations for strict environment replication.
* `LICENSE` — Open-source distribution permissions under the MIT Framework.

---

## 🛠️ Technology Stack & Dependencies

* **Language Engine:** Python 3.11+
* **Framework:** LangChain (LCEL) & Sentence-Transformers
* **Vector Store:** ChromaDB
* **API Providers:** OpenAI API, Groq, Google Gemini API

---

## ⚡ Quick Start & Deployment

### 1. Environment Setup
Clone the repository and initialize your local virtual environment:

```cmd
git clone [https://github.com/karmachandran-repository/RAG-Assistant.git](https://github.com/karmachandran-repository/RAG-Assistant.git)
cd RAG-Assistant/rt-aaidc-project1-template-main
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt