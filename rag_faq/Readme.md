# RAG_FAQ – Technical Documentation

## Overview
**RAG_FAQ** is a Retrieval‑Augmented Generation (RAG) pipeline specialized for answering FAQs. It creates FAQs from raw texts using an LLM, embeds the FAQ questions with **Sentence‑Transformers**, retrieves the most similar entries by **cosine similarity**, and generates final answers with an LLM over the retrieved context.

## Architecture
- **FAQ generation (indexing):** `rag_faq/indexer.py` uses `langchain_openai.ChatOpenAI` with prompts to turn raw texts into *k* question–answer pairs → saved as **`faq.csv`**.
- **Embeddings:** `rag_faq/embedder.py` encodes the **question** strings with `SentenceTransformer(model)` and saves **`embeddings.npy`** and **`faq_with_embeddings.csv`**.
- **Retrieval:** `rag_faq/retriever.py` loads `embeddings.npy` and computes cosine similarity (via `sklearn.metrics.pairwise.cosine_similarity`) between the user question and FAQ vectors; returns the top‑k entries from `faq.csv`.
- **Generation:** `rag_faq/generator.py` calls an LLM with a prompt template and the retrieved context to compose the final answer.
- **CLI:** `rag_faq/main.py` orchestrates **`--mode index`** (generate_faqs → embed_faqs) and **`--mode query`** (interactive RAG).
- **Server (HTTP):** `rag_faq/server.py` exposes a minimal Flask app with a browser form and a `/api/ask` endpoint that wraps `generate_rag_answer`.
- **Driver script:** `run_index.py` demonstrates building one project from `data/dataset_sample.csv`.

## File Tree (key files)
```
run_index.py
config.yaml
faq_gen.ipynb          # Manually generate FAQs and build embeddings
load_pdf.ipynb         # PDF preprocessing: split to pages/chunks and export CSVs
prompts/
  ├─ persona_aluno.txt
  ├─ persona_pesquisador.txt
  ├─ persona_professor.txt
  ├─ response.txt
  └─ rules.txt
data/
  ├─ dataset_sample.csv
  └─ ppp_<course>/
     ├─ ppp_<course>.pdf
     ├─ ppp_<course>_pages.csv
     └─ ppp_<course>_chunks.csv
projects/
  └─ myproj/
     ├─ ppp_all_courses/
     │     ├─ individual/   # faq.csv, faq_with_embeddings.csv, embeddings.npy
     │     └─ unificado/    # faq.csv, faq_with_embeddings.csv, embeddings.npy
     └─ ppp_<course>/
       ├─ individual/       # faq.csv, faq_with_embeddings.csv, embeddings.npy
       └─ unificado/        # faq.csv, faq_with_embeddings.csv, embeddings.npy (+ role splits)
rag_faq/
  ├─ main.py            # CLI (index/query)
  ├─ server.py          # Flask server + /api/ask
  ├─ generator.py       # Final answer via LLM over retrieved context
  ├─ retriever.py       # cosine similarity over Sentence-Transformers embeddings
  ├─ indexer.py         # LLM-based FAQ generation (CSV)
  ├─ embedder.py        # builds embeddings.npy (+ csv w/ vectors)
  ├─ config.py          # load_config()
  └─ utils.py           # prompt templates and JSON parsing helpers
```

## Configuration (`config.yaml`)
The effective config (secrets redacted) used in this repository is:
```yaml
llm:
  faq_generator:
    provider: openrouter
    model: meta-llama/llama-4-scout
    temperature: 0.0
    api_key: ***API_KEY***

  rag_answer:
    provider: openrouter
    model: meta-llama/llama-4-scout
    temperature: 0.0
    api_key: ***API_KEY***

embedding:
  model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

retrieval:
  top_k: 5

indexing:
  questions_per_text: 10

paths:
  projects_dir: ./projects
  prompts_dir: ./prompts

```
**Important parameters:**
- `llm.faq_generator` and `llm.rag_answer`: provider/model/temperature/api_key (use env vars; do not commit secrets).
- `embedding.model`: e.g., `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- `retrieval.top_k`: number of FAQs to pass as context.
- `indexing.questions_per_text`: how many Q/A pairs per input text.
- `paths.projects_dir`: where artifacts are created per project.
- `paths.prompts_dir`: directory with prompt templates used by `utils.py`.

## Module Details

### `rag_faq/indexer.py`
- **Function:** `generate_faqs(config, project_dir, texts)`
- **LLM:** `langchain_openai.ChatOpenAI` (OpenRouter) with a system+user prompt built via `utils.load_prompt_template/format_prompt`.
- **Output:** `project_dir / 'faq.csv'` with columns: `source_text`, `question`, `answer`.

### `rag_faq/embedder.py`
- **Function:** `embed_faqs(config, project_dir)`
- **Backend:** `SentenceTransformer(config['embedding']['model'])`.
- **Inputs:** `faq.csv`.
- **Outputs:** `embeddings.npy` (NumPy array aligned with `faq.csv`) and `faq_with_embeddings.csv`.

### `rag_faq/retriever.py`
- **Function:** `retrieve_similar_faqs(config, project_dir, user_question)`
- **Steps:** encode question → cosine similarity (`sklearn`) vs. `embeddings.npy` → rank → select `top_k` from `faq.csv`.
- **Returns:** list with keys `source_text`, `question`, `answer`, `score`.

### `rag_faq/generator.py`
- **Function:** `generate_rag_answer(config, project_dir, user_question, debug=False)`
- **Pipeline:** calls `retrieve_similar_faqs` → builds context → prompts LLM to produce final answer → returns keys `answer`, `context`, `raw_response`.

### `rag_faq/server.py`
- **Flask** single‑file app. CLI entry `start_server()` reads `--project` and `--config`, resolves `project_dir`, and serves:
  - `GET /` HTML form.
  - `POST /api/ask` → JSON `{ question: str }` → `{ answer, context }`.

### `rag_faq/main.py`
- **CLI:** `--mode index|query`, `--project`, `--config`.
  - `index`: `generate_faqs` then `embed_faqs`.
  - `query`: interactive console via `run_rag()`.

### `run_index.py`
- Simple example that loads `data/dataset_sample.csv`, sets `project_name="llm_test_project"`, and runs **FAQ generation** then **embeddings**.

## Artifacts per Project
- `individual/` (Single persona, per course or all-courses aggregate)
  - `faq.csv`
  - `faq_with_embeddings.csv` (includes vectors)
  - `embeddings.npy`
- `unificado/` (Multi-persona, per course or all-courses aggregate)
  - `faq.csv`
  - `faq_with_embeddings.csv` (includes vectors)
  - `embeddings.npy`
  - `faq_aluno.csv`        # persona-specific
  - `faq_professor.csv`    # persona-specific
  - `faq_pesquisador.csv`  # persona-specific
- Retrieval is in‑memory cosine similarity; no Chroma/FAISS store is required.

## How to Install
```bash
pip install -e .
# or
pip install .
```

## How to Run

### A) Index + Query via CLI
```bash
# Build artifacts
python -m rag_faq.main --mode index --project myproj --config config.yaml

# Interactive querying
python -m rag_faq.main --mode query --project myproj --config config.yaml
```

### B) Run the Flask Server
```bash
python -m rag_faq.server --project myproj --config config.yaml --port 8000
# open http://localhost:8000/
```

