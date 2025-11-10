# RAG_FAQ – Documentação Técnica

## Visão Geral
**RAG_FAQ** é um pipeline de Retrieval-Augmented Generation (RAG) que utiliza Perguntas Frequentes **(FAQs)** como estratégia de indexação para aprimorar a recuperação e geração de respostas a partir de textos extensos.

O sistema gera FAQs automaticamente a partir de documentos brutos com o auxílio de um Modelo de Linguagem (LLM), converte as perguntas em vetores semânticos por meio de **Sentence-Transformers**, realiza a busca por **similaridade de cosseno** para identificar os trechos mais relevantes e, por fim, gera respostas fundamentadas com um LLM, condicionadas ao contexto recuperado.

## Arquitetura
- **Geração de FAQs (indexação):** `rag_faq/indexer.py` usa `langchain_openai.ChatOpenAI` com prompts para transformar textos brutos em *k* pares pergunta-resposta → salvo como **`faq.csv`**.
- **Embeddings:** `rag_faq/embedder.py` codifica as strings de **pergunta** com `SentenceTransformer(model)` e salva **`embeddings.npy`** e **`faq_with_embeddings.csv`**.
- **Recuperação:** `rag_faq/retriever.py` carrega `embeddings.npy` e calcula similaridade de cosseno (via `sklearn.metrics.pairwise.cosine_similarity`) entre a pergunta do usuário e os vetores das FAQs; retorna as entradas top‑k de `faq.csv`.
- **Geração:** `rag_faq/generator.py` chama um LLM com um template de prompt e o contexto recuperado para compor a resposta final.
- **CLI:** `rag_faq/main.py` orquestra **`--mode index`** (generate_faqs → embed_faqs) e **`--mode query`** (RAG interativo).
- **Servidor (HTTP):** `rag_faq/server.py` expõe uma aplicação Flask mínima com um formulário no navegador e um endpoint `/api/ask` que encapsula `generate_rag_answer`.
- **Script driver:** `run_index.py` é utilizado quando o modo `--mode index` é executado, demonstrando a construção de um projeto a partir de configurações desejadas.

## Estrutura de Arquivos (arquivos principais)
```
config.yaml                 # Configurações do sistema
README.md                   # Documentação técnica
USAGE_GUIDE.md              # Guia de uso do sistema detalhado
run_index.py                # Script principal para indexação
run_evaluation.py           # Script principal para avaliação (eval)

prompts/
  ├─ persona_aluno.txt
  ├─ persona_pesquisador.txt
  ├─ persona_professor.txt
  ├─ generation_judge.txt  
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
     ├─ all_courses/        # Dataset agregado de todos os cursos
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
  ├─ evaluator.py           # lógica de avaliação do sistema
  ├─ config.py              # load_config()
  └─ utils.py               # templates de prompt e helpers de parsing JSON
notebooks/
  ├─ faq_gen.ipynb               # Geração manual de FAQs e construção de embeddings
  ├─ load_pdf.ipynb              # Pré-processamento de PDF: dividir em páginas/chunks e exportar CSVs
  └─ rag_faq_demo.ipynb          # Demo RAG completo: construir FAQs, incorporar e consultar interativamente
```

## Configuração (`config.yaml`)
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

  evaluator:
    provider: openai
    model: gpt-5
    temperature: 0.0
    api_key: ***API_KEY***

embedding:
  model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

retrieval:
  top_k: 5

indexing:
  questions_per_text: 10

evaluation:
  num_questions: 10
  num_variations: 5
  top_k_values: [1, 3, 5]

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
- `llm.faq_generator`, `llm.rag_answer` e `llm.evaluator`: provider/model/temperature/api_key (use variáveis de ambiente; não commite segredos).
- `embedding.model`: ex., `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- `retrieval.top_k`: número de FAQs para passar como contexto.
- `indexing.questions_per_text`: quantos pares Q/A por texto de entrada.
- `evaluation.num_questions`: número de perguntas simuladas usadas para avaliar o sistema (por persona/curso).
- `evaluation.num_variations`: quantas variações por pergunta o avaliador LLM deve gerar para testes.
- `evaluation.top_k_values`: valores de top-k para avaliar a busca (exemplo: avalie recall@1, recall@3, recall@5).
- `pipeline.personas`: personas disponíveis para geração de FAQs.
- `pipeline.data_sources`: fontes de dados configuradas com nomes, arquivos CSV e nomes dos cursos.
- `paths.projects_dir`: onde os artefatos são criados por projeto.
- `paths.prompts_dir`: diretório com templates de prompt usados por `utils.py`.

## Detalhes dos Módulos

### `rag_faq/indexer.py`
- **Função:** `generate_faqs(config, project_dir, texts)`
- **LLM:** `langchain_openai.ChatOpenAI` (OpenRouter) com um prompt system+user construído via `utils.load_prompt_template/format_prompt`.
- **Saída:** `project_dir / 'faq.csv'` com colunas: `source_text`, `question`, `answer`.

### `rag_faq/embedder.py`
- **Função:** `embed_faqs(config, project_dir)`
- **Backend:** `SentenceTransformer(config['embedding']['model'])`.
- **Entradas:** `faq.csv`.
- **Saídas:** `embeddings.npy` (array NumPy alinhado com `faq.csv`) e `faq_with_embeddings.csv`.

### `rag_faq/retriever.py`
- **Função:** `retrieve_similar_faqs(config, project_dir, user_question)`
- **Passos:** codificar pergunta → similaridade de cosseno (`sklearn`) vs. `embeddings.npy` → ranquear → selecionar `top_k` de `faq.csv`.
- **Retorna:** lista com chaves `source_text`, `question`, `answer`, `score`.

### `rag_faq/generator.py`
- **Função:** `generate_rag_answer(config, project_dir, user_question, debug=False)`
- **Pipeline:** chama `retrieve_similar_faqs` → constrói contexto → solicita LLM para produzir resposta final → retorna chaves `answer`, `context`, `raw_response`.

### `rag_faq/server.py`
- **Flask** aplicação de arquivo único. Entrada CLI `start_server()` lê `--project` e `--config`, resolve `project_dir`, e serve:
  - `GET /` formulário HTML.
  - `POST /api/ask` → JSON `{ question: str }` → `{ answer, context }`.

### `rag_faq/main.py`
- **CLI:** `--mode index|query`, `--project`, `--config`.
  - `index`: `generate_faqs` depois `embed_faqs`.
  - `query`: console interativo via `run_rag()`.

### `run_index.py`
- Módulo principal de indexação que implementa a lógica de geração de FAQs e embeddings. Contém as funções `run_index()` e `run_batch_indexing()` que são chamadas quando o modo `--mode index` é executado. Suporta processamento individual (persona única) e unificado (multi-persona), além de processamento em lote para múltiplas fontes de dados.

### `run_evaluation.py`
- **Função:** Script principal para avaliação automática do sistema RAG FAQ, medindo tanto a qualidade da recuperação (retrieval) quanto da geração (generation) de respostas.
- **Pipeline de Avaliação:**
  1. **Criação do Dataset de Teste:** Utiliza um LLM (configurado em `llm.evaluator`) para gerar perguntas de teste baseadas nas FAQs existentes, simulando diferentes personas e contextos por curso.
  2. **Geração de Variações:** Para cada pergunta original, cria múltiplas variações semânticas (paráfrases) para testar a robustez do sistema.
  3. **Avaliação de Recuperação:** Testa se as FAQs corretas são recuperadas para cada pergunta/variação, calculando métricas de retrieval (Hit@k, MRR, Precision@k, Recall@k, NDCG@k) para diferentes valores de top-k.
  4. **Avaliação de Geração:** Utiliza um LLM Judge para avaliar a qualidade das respostas geradas, medindo correção, completude e relevância.
- **Entradas:** Lê os arquivos `faq.csv` e `embeddings.npy` do projeto especificado.
- **Saídas:** Gera relatórios JSON detalhados com métricas agregadas por projeto, curso e persona, salvos em `evaluation_results/`.
- **Parâmetros de Configuração:** Definidos no `config.yaml` via a seção `evaluation`:
  - `num_questions`: Número de perguntas de teste geradas por persona/curso (padrão: 10)
  - `num_variations`: Quantas variações semânticas são criadas para cada pergunta (padrão: 5)
  - `top_k_values`: Lista de valores de top-k para avaliar (ex: [1, 3, 5])
- **Métricas Geradas:**
  - **Retrieval:** Hit@k, MRR (Mean Reciprocal Rank), Precision@k, Recall@k, NDCG@k
  - **Generation:** Score LLM, Correctness, Completeness, Relevance
- **Como Usar:**
  ```bash
  # Avaliar um projeto específico (individual ou unificado)
  python run_evaluation.py --config config.yaml --projects projects/batch_proj/ppp_bcc
  
  # Avaliar múltiplos projetos de uma vez
  python run_evaluation.py --config config.yaml --projects projects/batch_proj/ppp_bcc projects/batch_proj/ppp_bcd
  
  # Especificar diretório de saída customizado
  python run_evaluation.py --config config.yaml --projects projects/batch_proj/ppp_bcc --output-dir my_results
  
  # Modo dry-run (apenas mostra o que seria avaliado, sem executar)
  python run_evaluation.py --config config.yaml --projects projects/batch_proj/ppp_bcc --dry-run
  ```
- **Estrutura dos Resultados:** Os resultados são salvos em `evaluation_results/` com:
  - Relatórios individuais por projeto (JSON com métricas detalhadas)
  - Resumos agregados por projeto pai (agrupa `individual/` e `unificado/`)
  - Relatório geral consolidando todos os projetos avaliados


## Artefatos por Projeto
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

## Como Instalar
```bash
pip install -e .
# ou
pip install .
```

> **Para instruções detalhadas de uso, consulte o [USAGE_GUIDE.md](USAGE_GUIDE.md)**