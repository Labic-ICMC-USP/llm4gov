# Guia de Uso do Sistema RAG FAQ

## Visão Geral

O sistema RAG FAQ oferece três modos principais de operação:

### **Modo Index** - Geração de FAQs e Embeddings
- **Persona única** (modo individual) ou **Multi-persona** (modo unificado)
- **Múltiplas fontes de dados** (todos os cursos configurados no config.yaml)
- **Processamento em lote** para todas as fontes de dados
- **Parâmetros CLI flexíveis** para configuração fácil

### **Modo Query** - Consulta via CLI
- Interface de linha de comando
- Busca semântica nas FAQs geradas
- Respostas contextuais baseadas nos embeddings

### **Modo Server** - Servidor Web Flask
- Interface web para consultas
- API REST para integração
- Interface HTML amigável

## Exemplos de Uso - Modo Index

### 1. Curso Único, Persona Única (Modo Individual)
```bash
# Gerar FAQs para curso BCC (file: ppp_bcc) com persona aluno
python -m rag_faq.main --mode index --project test_proj --data-source ppp_bcc --persona aluno --index-mode individual

# Gerar FAQs para curso BEST (ppp_best) com persona professor
python -m rag_faq.main --mode index --project test_proj --data-source ppp_best --persona professor --index-mode individual
```

### 2. Curso Único, Multi-Persona (Modo Unificado)
```bash
# Gerar FAQs para curso BCC com todas as personas (aluno, professor, pesquisador)
python -m rag_faq.main --mode index --project test_proj --data-source ppp_bcc --index-mode unificado
```

### 3. Processamento em Lote - Todos os Cursos
```bash
# Processar TODOS os cursos com modo individual (persona única por curso)
python -m rag_faq.main --mode index --project batch_proj --batch --index-mode individual

# Processar TODOS os cursos com modo unificado (multi-persona por curso)
python -m rag_faq.main --mode index --project batch_proj --batch --index-mode unificado
```

### 4. Fonte de Dados Customizada
```bash
# Usar um arquivo CSV customizado
python -m rag_faq.main --mode index --project test_proj --data-source "data/custom_data.csv" --persona aluno
```

### Estrutura de Saída - Modo Index

#### Saída do Modo Individual
```
projects/
└── test_proj/
    └── ppp_<curso>/                        # Pasta específica do curso
        └── individual/
            ├── faq.csv                     # FAQs de persona única
            ├── faq_with_embeddings.csv     # FAQs com embeddings
            └── embeddings.npy              # Vetores de embedding
```

#### Saída do Modo Unificado
```
projects/
└── test_proj/
    └── ppp_<curso>/                        # Pasta específica do curso
        └── unificado/
            ├── faq.csv                     # FAQs mescladas de todas as personas
            ├── faq_aluno.csv               # FAQs específicas do aluno
            ├── faq_professor.csv           # FAQs específicas do professor
            ├── faq_pesquisador.csv         # FAQs específicas do pesquisador
            ├── faq_with_embeddings.csv     # FAQs com embeddings
            └── embeddings.npy              # Vetores de embedding
```

#### Saída do Processamento em Lote
```
projects/
└── batch_proj/
    ├── ppp_bcc/
    │   └── individual/ (ou unificado/)
    ├── ppp_bcd/
    │   └── individual/ (ou unificado/)
    ├── ppp_best/
    │   └── individual/ (ou unificado/)
    └── ... (todos os outros cursos)
```

## Exemplos de Uso - Modo Query

### Consulta Básica
```bash
# Consulta com projeto individual (persona única)
python -m rag_faq.main --mode query --project test_proj/ppp_bcc/individual

# Consulta com projeto unificado (múltiplas personas)
python -m rag_faq.main --mode query --project test_proj/ppp_bcc/unificado
```

## Exemplo de Uso - Modo Server

```bash
# Para usar FAQs de persona única (modo individual)
python -m rag_faq.server --project test_proj/ppp_bcc/individual --port 8000

# Para usar FAQs de múltiplas personas (modo unificado)
python -m rag_faq.server --project test_proj/ppp_bcc/unificado --port 8000

- **Interface Web**: Abra `http://localhost:8000` no navegador
- **Importante**: O `--project` deve incluir o caminho completo para o diretório do projeto (individual ou unificado)

# Estrutura de diretórios esperada:
# projects/
# └── test_proj/
#     └── ppp_bcc/
#         ├── individual/     ← Use este para persona única
#         │   ├── faq.csv
#         │   └── embeddings.npy
#         └── unificado/      ← Use este para múltiplas personas
#             ├── faq.csv
#             └── embeddings.npy
```

## Interface de Consulta (Modo Query)

### Como Funciona
1. **Inicialização**: O sistema carrega os embeddings e FAQs do projeto especificado
2. **Consulta**: Você digita uma pergunta em linguagem natural
3. **Busca Semântica**: O sistema encontra as FAQs mais relevantes usando similaridade de cosseno
4. **Geração de Resposta**: Um LLM gera uma resposta contextual baseada nas FAQs encontradas
5. **Exibição**: Mostra a resposta e as fontes utilizadas

## Interface do Servidor Web (Modo Server)

### Como Funciona
1. **Inicialização**: O servidor Flask carrega os embeddings e FAQs do projeto especificado
2. **Interface Web**: Usuários acessam via navegador em `http://localhost:porta`
3. **Formulário HTML**: Interface amigável para inserir perguntas
4. **Processamento**: Mesmo sistema de busca semântica do modo query
5. **Resposta Visual**: Exibe resposta e contexto de forma organizada

## Opções de Configuração

### Fontes de Dados Disponíveis (no config.yaml)
- `ppp_bcc` - Bacharelado em Ciência da Computação
- `ppp_bcd` - Bacharelado em Ciência de Dados
- `ppp_best` - Bacharelado em Estatística
- `ppp_bmacc` - Bacharelado em Matemática Aplicada e Computação Científica
- `ppp_bmat` - Bacharelado em Matemática
- `ppp_bsi` - Bacharelado em Sistemas de Informação
- `ppp_engcomp` - Engenharia da Computação
- `ppp_lce` - Licenciatura em Ciências Exatas
- `ppp_lmat` - Licenciatura em Matemática
- `dataset_sample` - Dataset de Exemplo

### Personas Disponíveis
- `aluno` - Perspectiva do estudante
- `professor` - Perspectiva do professor
- `pesquisador` - Perspectiva do pesquisador

> **⚠️ Aviso**: Para adicionar novas personas, será necessário criar o arquivo de prompt correspondente na pasta `/prompts` com o nome `persona_[nome_da_persona].txt`. O sistema automaticamente carregará o prompt baseado no nome da persona especificada.

## Uso Avançado

### Configuração Personalizada
Você pode modificar o `config.yaml` para:
- Adicionar novas fontes de dados
- Alterar tipos de persona
- Modificar configurações do LLM
- Ajustar modelos de embedding
- Ajustar número de FAQs recuperadas (`top_k` para retrieval)
- Configurar número de perguntas por texto (`questions_per_text`)

## Comandos de Início Rápido (Curso BCC com persona única e multi-persona)

### Modo Index
```bash
# Gerar curso específico (BCC) com multi-persona
python -m rag_faq.main --mode index --project test_proj --data-source ppp_bcc --index-mode unificado

# Gerar curso específico (BCC) com persona específica (aluno)
python -m rag_faq.main --mode index --project test_proj --data-source ppp_bcc --persona aluno
```

### Modo Query
```bash
# Consulta com projeto individual
python -m rag_faq.main --mode query --project test_proj/ppp_bcc/individual

# Consulta com projeto unificado
python -m rag_faq.main --mode query --project test_proj/ppp_bcc/unificado
```

### Modo Server
```bash
# Servidor com projeto individual
python -m rag_faq.server --project test_proj/ppp_bcc/individual --port 8080

# Servidor com projeto unificado
python -m rag_faq.server --project test_proj/ppp_bcc/unificado --port 8080
```

## Resumo dos Parâmetros

### Parâmetros Gerais
- `--mode`: `index` (geração) ou `query` (consulta)
- `--project`: Nome do projeto (para index/query) ou caminho completo (para server)
- `--config`: Arquivo de configuração (padrão: config.yaml)

### Parâmetros do Modo Index
- `--data-source`: Fonte de dados (nome do curso ou caminho do arquivo)
- `--persona`: Tipo de persona (aluno, professor, pesquisador)
- `--index-mode`: `individual` (persona única) ou `unificado` (multi-persona)
- `--batch`: Processar todas as fontes de dados configuradas

### Parâmetros do Modo Query
- Apenas `--project` e `--config` são necessários
- Interface interativa para consultas

### Parâmetros do Modo Server (rag_faq.server)
- `--project`: Caminho completo para o diretório do projeto (ex: myproj/ppp_bcc/individual)
- `--port`: Porta do servidor Flask
- `--config`: Arquivo de configuração
- Interface web e API REST disponíveis
