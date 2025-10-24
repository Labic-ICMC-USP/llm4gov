# 🤖 RAG_FAQ – Documentação Técnica

## 📋 Visão Geral
**RAG_FAQ** é um pipeline de Retrieval-Augmented Generation (RAG) que utiliza Perguntas Frequentes **(FAQs)** como estratégia de indexação para aprimorar a recuperação e geração de respostas a partir de textos extensos.

O sistema gera FAQs automaticamente a partir de documentos brutos com o auxílio de um Modelo de Linguagem (LLM), converte as perguntas em vetores semânticos por meio de **Sentence-Transformers**, realiza a busca por **similaridade de cosseno** para identificar os trechos mais relevantes e, por fim, gera respostas fundamentadas com um LLM, condicionadas ao contexto recuperado.

## 🏗️ Arquitetura
- **Geração de FAQs (indexação):** `rag_faq/indexer.py` usa `langchain_openai.ChatOpenAI` com prompts para transformar textos brutos em *k* pares pergunta-resposta → salvo como **`faq.csv`**.
- **Embeddings:** `rag_faq/embedder.py` codifica as strings de **pergunta** com `SentenceTransformer(model)` e salva **`embeddings.npy`** e **`faq_with_embeddings.csv`**.
- **Recuperação:** `rag_faq/retriever.py` carrega `embeddings.npy` e calcula similaridade de cosseno (via `sklearn.metrics.pairwise.cosine_similarity`) entre a pergunta do usuário e os vetores das FAQs; retorna as entradas top‑k de `faq.csv`.
- **Geração:** `rag_faq/generator.py` chama um LLM com um template de prompt e o contexto recuperado para compor a resposta final.
- **CLI:** `rag_faq/main.py` orquestra **`--mode index`** (generate_faqs → embed_faqs) e **`--mode query`** (RAG interativo).
- **Servidor (HTTP):** `rag_faq/server.py` expõe uma aplicação Flask mínima com um formulário no navegador e um endpoint `/api/ask` que encapsula `generate_rag_answer`.
- **Script driver:** `run_index.py` é utilizado quando o modo `--mode index` é executado, demonstrando a construção de um projeto a partir de configurações desejadas.

## 📁 Estrutura de Arquivos (arquivos principais)
```
run_index.py                # Script principal para indexação
config.yaml                 # Configurações do sistema
README.md                   # Documentação técnica
USAGE_GUIDE.md              # Guia de uso detalhado
notebooks/
  ├─ faq_gen.ipynb          # Geração manual de FAQs e construção de embeddings
  ├─ load_pdf.ipynb         # Pré-processamento de PDF: dividir em páginas/chunks e exportar CSVs
  └─ rag_faq_demo.ipynb     # Demo RAG completo: construir FAQs, incorporar e consultar interativamente
prompts/
  ├─ persona_aluno.txt
  ├─ persona_pesquisador.txt
  ├─ persona_professor.txt
  ├─ response.txt
  └─ rules.txt
data/
  ├─ dataset_sample.csv
  └─ ppp_<curso>/           # Dados de cada curso
     ├─ ppp_<curso>.pdf
     ├─ ppp_<curso>_pages.csv
     └─ ppp_<curso>_chunks.csv
projects/
  └─ <projeto>/
     ├─ ppp_all_courses/    # Agregado de todos os cursos
     │   ├─ individual/     # faq.csv, faq_with_embeddings.csv, embeddings.npy
     │   └─ unificado/      # faq.csv, faq_with_embeddings.csv, embeddings.npy
     └─ ppp_<curso>/        # Pasta específica do curso
       ├─ individual/       # faq.csv, faq_with_embeddings.csv, embeddings.npy
       └─ unificado/        # faq.csv, faq_with_embeddings.csv, embeddings.npy (+ divisões por persona)
rag_faq/
  ├─ main.py                # CLI (index/query)
  ├─ server.py              # Servidor Flask + /api/ask
  ├─ generator.py           # Resposta final via LLM sobre contexto recuperado
  ├─ retriever.py           # similaridade de cosseno sobre embeddings Sentence-Transformers
  ├─ indexer.py             # Geração de FAQs baseada em LLM (CSV)
  ├─ embedder.py            # constrói embeddings.npy (+ csv com vetores)
  ├─ config.py              # load_config()
  └─ utils.py               # templates de prompt e helpers de parsing JSON
```

## ⚙️ Configuração (`config.yaml`)
A configuração completa (segredos omitidos) usada neste repositório é:
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

# CONFIGURAÇÃO DO PIPELINE DE GERAÇÃO DE FAQ
pipeline:
  # Personas disponíveis
  personas: ["aluno", "professor", "pesquisador"]
  
  # Fontes de dados disponíveis
  data_sources:
    - name: "ppp_bcc"
      csv_file: "data/ppp_bcc/ppp_bcc_chunks.csv"
      course_name: "Bacharelado em Ciência da Computação"
    - name: "ppp_bcd"
      csv_file: "data/ppp_bcd/ppp_bcd_chunks.csv"
      course_name: "Bacharelado em Ciência de Dados"
    - name: "ppp_best"
      csv_file: "data/ppp_best/ppp_best_chunks.csv"
      course_name: "Bacharelado em Estatística"
    - name: "ppp_bmacc"
      csv_file: "data/ppp_bmacc/ppp_bmacc_chunks.csv"
      course_name: "Bacharelado em Matemática Aplicada e Computação Científica"
    - name: "ppp_bmat"
      csv_file: "data/ppp_bmat/ppp_bmat_chunks.csv"
      course_name: "Bacharelado em Matemática"
    - name: "ppp_bsi"
      csv_file: "data/ppp_bsi/ppp_bsi_chunks.csv"
      course_name: "Bacharelado em Sistemas de Informação"
    - name: "ppp_engcomp"
      csv_file: "data/ppp_engcomp/ppp_engcomp_chunks.csv"
      course_name: "Engenharia da Computação"
    - name: "ppp_lce"
      csv_file: "data/ppp_lce/ppp_lce_chunks.csv"
      course_name: "Licenciatura em Ciências Exatas"
    - name: "ppp_lmat"
      csv_file: "data/ppp_lmat/ppp_lmat_chunks.csv"
      course_name: "Licenciatura em Matemática"
    - name: "dataset_sample"
      csv_file: "data/dataset_sample.csv"
      course_name: "Sample Dataset"

paths:
  projects_dir: ./projects
  prompts_dir: ./prompts
```

**Parâmetros importantes:**
- `llm.faq_generator` e `llm.rag_answer`: provider/model/temperature/api_key (use variáveis de ambiente; não commite segredos).
- `embedding.model`: ex., `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- `retrieval.top_k`: número de FAQs para passar como contexto.
- `indexing.questions_per_text`: quantos pares Q/A por texto de entrada.
- `pipeline.personas`: personas disponíveis para geração de FAQs.
- `pipeline.data_sources`: fontes de dados configuradas com nomes, arquivos CSV e nomes dos cursos.
- `paths.projects_dir`: onde os artefatos são criados por projeto.
- `paths.prompts_dir`: diretório com templates de prompt usados por `utils.py`.

## 🔧 Detalhes dos Módulos

### 📝 `rag_faq/indexer.py`
- **Função:** `generate_faqs(config, project_dir, texts)`
- **LLM:** `langchain_openai.ChatOpenAI` (OpenRouter) com um prompt system+user construído via `utils.load_prompt_template/format_prompt`.
- **Saída:** `project_dir / 'faq.csv'` com colunas: `source_text`, `question`, `answer`.

### 🔢 `rag_faq/embedder.py`
- **Função:** `embed_faqs(config, project_dir)`
- **Backend:** `SentenceTransformer(config['embedding']['model'])`.
- **Entradas:** `faq.csv`.
- **Saídas:** `embeddings.npy` (array NumPy alinhado com `faq.csv`) e `faq_with_embeddings.csv`.

### 🔍 `rag_faq/retriever.py`
- **Função:** `retrieve_similar_faqs(config, project_dir, user_question)`
- **Passos:** codificar pergunta → similaridade de cosseno (`sklearn`) vs. `embeddings.npy` → ranquear → selecionar `top_k` de `faq.csv`.
- **Retorna:** lista com chaves `source_text`, `question`, `answer`, `score`.

### 💬 `rag_faq/generator.py`
- **Função:** `generate_rag_answer(config, project_dir, user_question, debug=False)`
- **Pipeline:** chama `retrieve_similar_faqs` → constrói contexto → solicita LLM para produzir resposta final → retorna chaves `answer`, `context`, `raw_response`.

### 🌐 `rag_faq/server.py`
- **Flask** aplicação de arquivo único. Entrada CLI `start_server()` lê `--project` e `--config`, resolve `project_dir`, e serve:
  - `GET /` formulário HTML.
  - `POST /api/ask` → JSON `{ question: str }` → `{ answer, context }`.

### ⚡ `rag_faq/main.py`
- **CLI:** `--mode index|query`, `--project`, `--config`.
  - `index`: `generate_faqs` depois `embed_faqs`.
  - `query`: console interativo via `run_rag()`.

### 🚀 `run_index.py`
- Módulo principal de indexação que implementa a lógica de geração de FAQs e embeddings. Contém as funções `run_index()` e `run_batch_indexing()` que são chamadas quando o modo `--mode index` é executado. Suporta processamento individual (persona única) e unificado (multi-persona), além de processamento em lote para múltiplas fontes de dados.

## 📦 Artefatos por Projeto
- `ppp_<curso>/individual/` (Persona única, por curso ou agregado de todos os cursos)
  - `faq.csv`
  - `faq_with_embeddings.csv` (inclui vetores)
  - `embeddings.npy`
- `ppp_<curso>/unificado/` (Multi-persona, por curso ou agregado de todos os cursos)
  - `faq.csv`
  - `faq_with_embeddings.csv` (inclui vetores)
  - `embeddings.npy`
  - `faq_aluno.csv`        # específico da persona
  - `faq_professor.csv`    # específico da persona
  - `faq_pesquisador.csv`  # específico da persona
- A recuperação é similaridade de cosseno em memória; não é necessário armazenamento Chroma/FAISS.

## 📥 Como Instalar
```bash
pip install -e .
# ou
pip install .
```

> **📖 Para instruções detalhadas de uso, consulte o [USAGE_GUIDE.md](USAGE_GUIDE.md)**

