# LLM4Gov: Tutorial de Configuração e Execução

## Introdução

Grandes modelos de linguagem (LLMs) revolucionaram o processamento de linguagem natural, mas o seu uso no setor público pode ser limitado por fatores como custo, necessidade de hardware e restrições legais relacionadas à privacidade. No _paper_ do [**LLM4Gov**](https://sol.sbc.org.br/index.php/wcge/article/view/36338), destacamos que tais modelos costumam ser proprietários, caros e inacessíveis para organizações governamentais, especialmente quando os dados contêm informações sensíveis.

Para contornar esses problemas, foi investigada uma arquitetura de **aprendizado professor–aluno** (_teacher‑student_) com foco em privacidade e baixo custo.  O pipeline é composto por três etapas principais: 

1. **Anonimização** – remoção de informações pessoais identificáveis (PII) antes de qualquer interação com um LLM externo.

2. **Geração de instruções pelo professor** – um modelo de grande porte gera exemplos rotulados a partir do conjunto (amostra) anonimizado.

3. **Ajusto fino do estudante** – um LLM destilado é ajustado usando técnicas eficientes (LoRA e quantização) para aprender com as instruções do professor.

A seguir, é detalhado como executar o framework do LLM4Gov, incluindo como instalar as dependências, preparar os dados, executar o modelo **teacher**, realizar o ajuste fino do modelo **student** e realizar inferência com o modelo treinado. Ao final, também é fornecida uma versão em notebook Jupyter com os mesmos passos.

## Visão geral do Framework LLM4Gov

O **LLM4Gov** foi concebido para preservar a privacidade de documentos sensíveis ao longo de todo o fluxo de treinamento. A sequência geral é:

1. **Anonimização dos dados** – Um módulo de anonimização  identifica entidades como nomes, endereços, emails e telefones e as substitui por _placeholders_ ([NAME], [ADDRESS], etc.), mantendo a estrutura e o significado das sentenças. A princípio, você pode usar qualquer módulo de anonimização de dados. Também pode escolher pular esta etapa, caso não esteja trabalhando com dados sensíveis. Se desejar utilizar o conhecer o projeto de anonimização do LLM4Gov, consulte o módulo `anonymizer` do repositório. Se você apenas deseja aprender a usar o LLM4Gov, sugerimos usar o dataset `training_unlabeled.json`, que foi gerado de forma sintética para fins didáticos.


2. **Geração de instruções (Teacher)** – O dataset é passado a um LLM de grande porte (professor). A escolha do modelo professor é uma decisão importante, pois o modelo estudante irá aproximar o conhecimento gerado por este professor (incluindo acertos e erros). Esse modelo gera pares ⟨instrução, entrada, saída⟩ que instruem como resolver a tarefa, explicando a decisão em formato JSON.  A estrutura de cada exemplo segue o formato **Instruction → Input → Response**; a saída contém predições e explicações da tarefa.

3. **Ajusto Fino do estudante (Student)** – Um LLM destilado é ajustado a partir dos dados do modelo professor. Você pode escolher diferentes modelos estudantes pré-treinados nesta versão do **LLM4Gov** para realizar o ajuste fino (veja o arquivo `student_fine_tuning_config.yaml`).  O **LLM4Gov** usa LoRA (Low‑Rank Adaptation) para congelar os pesos originais do modelo e adicionar matrizes de baixa dimensão para treinamento.  Também é aplicado 4‑bit quantization para reduzir o consumo de memória.

4. **Inferência** – Após o treinamento, o modelo estudante pode gerar saídas para novos documentos, imitando o comportanto e desempenho do modelo professor. Vale destacar que o modelo estudante tem comportante específico para a tarefa. Embora ele possa ter bom desempenho para a tarefa do ajusto fino, o modelo estudante terá dificuldades em resolver outras tarefas não ensinadas pelo modelo professor. Por ser um LLM mais compacto, o modelo estudante pode ser executado em hardware mais modesto e, preferencialmente, numa estrutura local e preservando a privacidade.


## Pré-requisitos e instalação

Siga os passos abaixo em uma máquina com Python ≥ 3.10 e com GPU.

1. **Clone o repositório do LLM4Gov** para um diretório local.

   ```bash
   git clone https://github.com/Labic-ICMC-USP/llm4gov.git
   ```

2. **Crie um ambiente virtual** (opcional, mas recomendado):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .\.venv\Scripts\activate  # Windows
   ```
   OBS: o ambiente virtual não precisa ser criado se você estiver utilizando Google Colab.

3. **Instale as dependências** listadas em `requirements.txt`:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   O arquivo inclui bibliotecas como `unsloth`, `transformers`, `trl`, `langchain` e `openai` necessárias para o **LLM4Gov**.

4. **Dados para treinamento e inferência**: você precisará de um conjunto de documentos para treinamento (`training_unlabeled.json`) que o modelo professor irá receber e outro para teste (`test_unlabeled.json`) que o modelo estudante irá processar depois de ser treinado.

   Os exemplos devem ter os campos `system_prompt` (string com instruções gerais) e `user_prompt` (texto ou JSON com o documento). 

   - **`system_prompt`** → sempre uma **string**, define o papel ou o contexto da tarefa.  
   - **`user_prompt`** → pode ser **uma string simples** ou **um dicionário estruturado** com os dados de entrada.

   #### Exemplo 1 — `user_prompt` como string

    ```json
    [
        {
            "system_prompt": "Você é um especialista em análise de contratos.",
            "user_prompt": "O fornecedor não cumpriu o prazo de entrega estabelecido no contrato."
        }
    ]
    ````

    #### Exemplo 2 — `user_prompt` como dicionário

    ```json
    [
        {
            "system_prompt": "Você é um avaliador de relatórios técnicos.",
            "user_prompt": {
                "id": "DOC-001",
                "text": "A entrega dos equipamentos foi adiada em 10 dias devido a falhas logísticas.",
                "metadata": {
                    "categoria": "infraestrutura",
                    "prioridade": "alta"
                }
            }
        }
    ]
    ```

   Prepare também um arquivo `structure_example.json`, definido no `teacher_config.yaml`, que descreve o formato de saída desejado do professor. Esse arquivo serve como **modelo de estrutura de saída** que o *Teacher* deve gerar. Na prátic, ele define **como deve ser o formato do JSON** que o LLM precisa retornar, ou seja, as chaves e tipos esperados. Este formato é livro, definido pela necessidade do proejto, e é seguido depois também pelo modelo estudante. A seguir, um exemplo de estrutura:

    ```json
    {
        "document_id": "IDENTIFICADOR_DO_DOCUMENTO",
        "label": "RÓTULO_GERADO_PELO_MODELO",
        "explanation": "Breve justificativa da decisão ou classificação feita pelo modelo."
    }
    ```

## Geração de instruções com o Teacher

Com os dados de trienamento, utilizamos o script `teacher.py` para solicitar a um LLM de grande porte que gere instruções e rótulos.

 O **Teacher** implementa um executor com validação de esquema usando `pydantic`, garantindo que as saídas sigam a estrutura definida em `structure_example.json`. Algumas observações:

* O script se baseia na classe `ChatOpenAI` da biblioteca LangChain para conectar‑se em qualquer serviço de LLM via API que suporte este protocolo.

* O arquivo de configuração `teacher_config.yaml` permite definir o modelo (por exemplo, `deepseek/deepseek-chat-v3-0324`), a chave da API (`api_key`) e o arquivo de exemplo de estrutura (`schema_example_path`).  Certifique‑se de fornecer uma chave válida de API do serviço (por exemplo, OpenRouter).  Se preferir, defina a variável de ambiente `OPENAI_API_KEY` em vez de colocar a chave no YAML.

* Os campos `input_json` e `output_json` especificam, respectivamente, o arquivo de dados de entrada e o arquivo de saída com rótulos.

Para executar o professor:

1. Edite o arquivo `teacher_config.yaml` e ajuste os seguintes itens:
   * `model`: nome do modelo LLM que servirá como professor (por exemplo, `deepseek/deepseek-chat-v3-0324` ou outro compatível com a API escolhida).
   * `api_key` e `base_url`: credenciais e URL do serviço de LLM.  Se `api_key` ficar vazia, a chave será lida de `OPENAI_API_KEY`.
   * `input_json`: defina o arquivo anonimizado (por exemplo, `training_unlabeled_anon.json`).
   * `output_json`: nome do arquivo de saída, onde as instruções do professor serão salvas (por padrão `training_labeled.json`).
   * `schema_example_path`: caminho para o JSON que descreve o esquema de saída esperado.

2. Execute o script:

   ```bash
   python teacher.py --config teacher_config.yaml
   ```

   O script carrega cada item, envia ao modelo, valida a saída JSON e salva no arquivo indicado.  O campo `teacher_output` conterá o JSON gerado pelo professor.  O log de progresso é exibido no terminal e gravado no arquivo especificado em `log_file`.

## Ajuste fino do Student (fine‑tuning)

Após gerar o conjunto rotulado, treinamos um modelo estudante mais leve.  O script `student_fine_tuning.py` utiliza a biblioteca **Unsloth** para realizar _fine‑tuning_ via LoRA e quantização.  A abordagem LoRA congela os pesos originais e atualiza apenas matrizes de baixa dimensão, reduzindo drasticamente o número de parâmetros a treinar, enquanto a quantização em 4 bits diminui o consumo de memória.  

### Configuração do fine‑tuning

O arquivo `student_fine_tuning_config.yaml` possui diversas seções:

* `model`: define o nome do modelo base (e.g. `unsloth/Phi-3.5-mini-instruct`), o tamanho máximo de sequência (`max_seq_length`), o tipo de _dtype_ e se o modelo deve ser carregado em 4 bits. Atualmente, o **LLM4Gov** é funcionado para os **modelos estudantes** abaixo, mas em princípio qualquer modelo disponível no **Unsloth** pode ser utilizado, desde que sejam feitos os devidos ajustes no script de ajuste fino (`student_fine_tuning.py`):

    ```python
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 405B também disponível em 4bit!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # Novo Mistral 12B, 2x mais rápido!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3, 2x mais rápido!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5, 2x mais rápido!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit"             # Gemma, 2x mais rápido!



* `lora`: parâmetros do LoRA, como a dimensão interna `r`, os módulos alvo (`target_modules`), `lora_alpha` e `lora_dropout`.
* `train_args`: hiperparâmetros de treinamento, incluindo tamanho de lote (`per_device_train_batch_size`), número de passos ou épocas, taxa de aprendizado e diretório de saída.
* `save`: especifica onde salvar os adaptadores LoRA (`save_lora_dir`) e se devem ser gerados modelos combinados (16 ou 4 bits) ou arquivos GGUF.
* `data`: indica o arquivo rotulado produzido pelo professor (`labeled_file`) e permite limitar a quantidade de exemplos.

Ajuste esses parâmetros conforme seus recursos de hardware e tamanho do dataset.  Para um treinamento rápido de teste, limite `max_steps` ou `max_examples`.

### Execução do treinamento

1. Verifique se o arquivo rotulado `training_labeled.json` (gerado pelo professor) está acessível no caminho indicado em `student_fine_tuning_config.yaml`.
2. Inicie o treinamento com:

   ```bash
   python student_fine_tuning.py --config student_fine_tuning_config.yaml
   ```

Durante o processo, o script:

* Carrega o modelo base com quantização de 4 bits.
* Anexa as camadas LoRA com as configurações especificadas.
* Treina o modelo usando o `SFTTrainer` da biblioteca TRL.
* Salva os adaptadores LoRA e, se configurado, versões fundidas ou quantizadas.

O tempo de treinamento depende do tamanho do conjunto e da GPU. As mensagens de log indicarão a utilização de GPU e o tempo aproximado.

## Inferência com o Student

Com o modelo estudante treinado, podemos gerar respostas para novos documentos.  O script `student_inference.py` carrega o modelo LoRA e processa exemplos em lote.  As configurações estão no arquivo `student_inference_config.yaml`:

* `model_path`: diretório onde os adaptadores LoRA ou modelo fundido foram salvos (por padrão, `lora_model`). 
* `max_seq_length`: tamanho máximo de contexto na entrada.
* `max_new_tokens`: limite de tokens que o modelo pode gerar (ajuste conforme a tarefa e a GPU disponível).
* `batch_size`: número de exemplos processados por vez na inferência.
* `test_file`: arquivo JSON/JSONL com exemplos a serem processados (deve conter `system_prompt` e `user_prompt`).
* `output_file`: caminho para salvar as respostas geradas pelo estudante.

Para rodar a inferência:

```bash
python student_inference.py --config student_inference_config.yaml
```

O script realiza os seguintes passos:

* Carrega o modelo LoRA em 4 bits.
* Gera as respostas usando decodificação determinística (sem amostragem).  Após a geração, tenta converter a resposta em JSON; se não for possível, mantém como texto bruto.
* Escreve as saídas em `output_file` com os campos `system_prompt`, `user_prompt` e `student_output`.

