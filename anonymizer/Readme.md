# TextAnonymizer

Classe Python para **anonimização de texto em batches**, com **saída estruturada (Pydantic v2)**, **cliente OpenAI-compatible (OpenRouter/vLLM/Ollama local)**, **regras personalizadas (custom_rules)** injetadas no prompt, **dry-run**, **logs verbosos** e **barra de progresso** via `tqdm`.

Utilizado como módulo de anonimização do LLM4Gov

> Arquivo do módulo: `anonymizer.py`

---

## ✨ Recursos

* ✅ **Processamento em batches**: define `sentences_per_batch`.
* ✅ **Saída 100% JSON** validada com **Pydantic v2**.
* ✅ **Cliente OpenAI-compatible**: funciona com **OpenRouter**, **vLLM** (OpenAI server) e **Ollama** (router OpenAI).
* ✅ **Preserva idioma original** (ou força um idioma com `explicit_language`).
* ✅ **`custom_rules`** entram **naturalmente** na seção *Rules* do prompt (e também podem ser aplicadas localmente).
* ✅ **Dry-run** de prompt para debug.
* ✅ **Logs** (structlog JSON se disponível, senão `logging`) + **tqdm** para progresso.

---

## Instalação

```bash
pip install pydantic tqdm structlog requests
# opcional: structlog (para logs JSON); requests é usado no cliente OpenAI-compatible
```

> Requer **Pydantic v2** (usa `@field_validator`).

---

## Estrutura principal

* `TextAnonymizer`: orquestra pipeline (split → batch → LLM → parse → recomposição).
* `OpenAICompatibleClient`: cliente `/chat/completions` estilo OpenAI (funciona com OpenRouter).
* `LLMConfig`, `AnonymizationConfig`: configurações.
* `AnonymizedBatchOutput`: esquema Pydantic da resposta JSON.
* `dry_run_prompt(text)`: mostra o prompt final (sem chamar a LLM).
* `get_last_batches_info()`: métricas por batch.

---

## Como importar

```python
from anonymizer import (
    TextAnonymizer,
    OpenAICompatibleClient,
    LLMConfig,
    AnonymizationConfig
)
```

---

## Exemplo 1 — Uso básico

Sem `custom_rules` (apenas anonimização padrão).

> *Observação*: a saída exata depende do modelo/provedor; abaixo está um **exemplo** de como pode vir.

```python
from anonymizer import TextAnonymizer, OpenAICompatibleClient, LLMConfig

llm_cfg = LLMConfig(
    base_url="xxx",
    api_key="yyyy",
    model="modelname",
    temperature=0.0,
    provider="localprovider",
    extra_params={"top_p": 1.0},
)

client = OpenAICompatibleClient(
    base_url=llm_cfg.base_url,
    api_key=llm_cfg.api_key,
    timeout=60,
)

anonymizer = TextAnonymizer(
    llm_client=client,
    llm_config=llm_cfg,
    sentences_per_batch=8,
)

text = '''
Em 18 de fevereiro de 2024, o cientista Tony Solaris enviou um e-mail para a Dra. Luna Vega, diretora da Corporação NeoQuantum, informando sobre um vazamento crítico no laboratório subterrâneo de Órion. O assunto da mensagem era “Falha no Reator Estelar – Prioridade Máxima”. O conteúdo incluía registros de segurança com o identificador 777-999-314 e credenciais temporárias de acesso: login “t.solaris@neoquantum.io” e senha “Nova@2024”.

O relatório anexo, intitulado “Análise de Energia Estelar – Fase Beta”, descrevia como o núcleo do reator, baseado no elemento fictício Helion-9, havia atingido níveis de instabilidade acima do permitido. O documento mencionava que o experimento fora financiado pela Fundação Chronos, em parceria com o Ministério da Ciência Galáctica. A pesquisa fazia parte do projeto MAI-DAI Nebula, coordenado pelo Prof. Bruce Orion, da Universidade de Nova Arcádia.

Durante os testes de contenção, o engenheiro Peter Blaze detectou uma anomalia nos sensores de temperatura. Ele relatou que o sistema de monitoramento Quasar-X, desenvolvido pela StarkWave Industries, não estava respondendo aos comandos do módulo central. “Se o núcleo atingir 1200 graus, a fusão será irreversível”, escreveu Peter em um relatório enviado às 03:47 da manhã de 19 de fevereiro. O relatório incluía o endereço IP 203.88.45.172 e o código interno de operação QX-A17-94.

Enquanto isso, na superfície, a agente especial Diana Sky coordenava a evacuação da equipe. Usando seu comunicador da série Pegasus-V, ela manteve contato direto com a base lunar Helix-01, onde o comandante Clark Volt aguardava instruções. Diana reportou que 42 pesquisadores já haviam sido realocados para a colônia Alfa-Centauri II. O restante deveria embarcar no transporte quântico “Aurora Prime” até o fim do dia.

Em uma transmissão de emergência, Bruce Orion declarou: “A situação está sob controle parcial. Precisamos isolar o Helion-9 e recalibrar o sistema Quasar-X antes do colapso do campo gravitacional local.” A mensagem foi gravada e arquivada sob o protocolo de segurança GA-2024-12. O backup automático foi salvo no servidor interno “nebula-core.local” com hash de verificação 4fa7-x91-12b-9e3.

No dia seguinte, a imprensa interplanetária publicou manchetes sensacionalistas: “Falha no Núcleo Helion-9 ameaça colônia orbital!”. O portal **Cosmic Times** afirmou que a NeoQuantum já havia enfrentado incidentes similares em 2022, quando o projeto StellarBridge foi encerrado após um vazamento de energia. Fontes anônimas alegaram que o atual acidente poderia ter sido causado por sabotagem.

Enquanto as autoridades investigavam, Luna Vega solicitou apoio da empresa de IA SentinelAI, sediada em Neo Tokyo. O modelo linguístico utilizado, chamado Nova-LLM, foi treinado para gerar relatórios automáticos de risco. O algoritmo aplicava uma matriz de severidade-probabilidade semelhante à usada pela agência espacial da Federação de Sistemas Unidos. O primeiro relatório gerado indicou 87% de chance de falha catastrófica se o reator não fosse estabilizado em 12 horas.

No mesmo dia, Peter Blaze reconfigurou o sistema de controle utilizando um patch experimental assinado por “B.Wayne@starkwave.io”. Após a atualização, o sistema respondeu normalmente, e as leituras do Helion-9 retornaram aos níveis seguros. Diana Sky relatou o sucesso à base Helix-01 e registrou o evento sob o código de missão SKY-2045-BR.

Três meses depois, em 27 de maio de 2024, Tony Solaris apresentou os resultados na Conferência Galáctica de Engenharia Avançada (CGEA), realizada em Nova Arcádia. O artigo “Containment Strategies for Helion-9 Cores” foi coassinado por Luna Vega, Bruce Orion e Peter Blaze. O DOI fictício do trabalho é 10.9999/cgea.2024.0420. O evento contou com mais de 3.000 participantes e patrocínio da organização Stellar Council.

Em agosto de 2025, a equipe iniciou o projeto **QuasarNet**, voltado para aprendizado federado entre colônias. O novo modelo, chamado HyperNova-3, foi implementado usando a API pública do OpenRouter Galactic Hub. O relatório final foi compartilhado via link seguro: https://galactic-hub.net/reports/quasarnet_final_v3.json. Espera-se que os resultados sejam revisados pelo Conselho Científico Central antes do fim do ano estelar.
'''

example1 = anonymizer.anonymize_text(text)
print(example1)
```

**Exemplo de saída (pode variar por modelo):**

> Em 18 de fevereiro de 2024, o cientista [NOME] enviou um e-mail para a Dra. [NOME], diretora da [ORGANIZAÇÃO], informando sobre um vazamento crítico no laboratório subterrâneo de [LOCAL]. O assunto da mensagem era “Falha no Reator Estelar – Prioridade Máxima”. O conteúdo incluía registros de segurança com o identificador [ID] e credenciais temporárias de acesso: login “[EMAIL]” e senha “[SENHA]”. … *(continua no mesmo padrão de anonimização preservando a ordem das sentenças).*

---

## Exemplo 2 — Com `custom_rules` (define regras de anonimização específicas)

Aqui definimos duas regras que **entram direto** na seção *Rules* (linhas `- Replace …`) **apenas se existirem**, e **também** são aplicadas localmente no pós-processamento.

```python
from anonymizer import TextAnonymizer, OpenAICompatibleClient, LLMConfig, AnonymizationConfig

llm_cfg = LLMConfig(
    base_url="xxx",
    api_key="yyyy",
    model="modelname",
    temperature=0.0,
    provider="localprovider",
    extra_params={"top_p": 1.0},
)

client = OpenAICompatibleClient(
    base_url=llm_cfg.base_url,
    api_key=llm_cfg.api_key,
    timeout=60,
)

anon = TextAnonymizer(
    llm_client=client,
    llm_config=llm_cfg,
    sentences_per_batch=8,
    anonymization_config=AnonymizationConfig(
        custom_rules={
            "Remover Assunto de Mensagens": "<ASSUNTO_REMOVIDO>",
            "Remover Datas": "<DATA_REMOVIDA>",
        }
    ),
)

example2 = anon.anonymize_text(text)
print(example2)
```

**Exemplo de saída (pode variar por modelo):**

> Em <DATA_REMOVIDA>, o cientista <NOME_REMOVIDO> enviou um e-mail para a Dra. <NOME_REMOVIDO>, diretora da <ORGANIZAÇÃO_REMOVIDA>, informando sobre um vazamento crítico no laboratório subterrâneo de <LOCAL_REMOVIDO>. O assunto da mensagem era “<ASSUNTO_REMOVIDO>”. O conteúdo incluía registros… *(e assim por diante, com substituições de acordo com as regras).*

---

## Dry-run do prompt

Veja como fica o prompt **sem chamar** a LLM:

```python
preview = anonymizer.dry_run_prompt(text, limit_sentences=5)
print(preview)
```

Útil para verificar se `custom_rules` foram **de fato injetadas** na seção *Rules*.

---

## Dicas de depuração

* **Logs**: se `structlog` estiver instalado, os logs saem em JSON (nível DEBUG).
* **Validação Pydantic**: se a LLM não retornar JSON válido, você verá `LLMResponseParseError` ou `ValidationError`.
* **Chaves no prompt**: quando usar `{}` literais no template, **escape** com `{{` e `}}`.

---

## Trocar de provedor de LLM

O modelo (LLM) de anonimização deve ser preferencialmente local!

Se o seu endpoint é OpenAI-compatible (mesmo caminho `/chat/completions`), basta trocar:

* `base_url` para o do seu servidor (ex.: `https://openrouter.ai/api/v1`, `http://localhost:8000` para vLLM).
* `api_key` e `model` conforme o provedor.

> Para provedores não-compatíveis, implemente outro cliente que siga o **protocolo `LLMClient`**.

---

## Notas

* A saída final também aplica `custom_rules` **localmente** no `_postprocess_sentences` (case-insensitive), além das instruções no prompt (abordagem híbrida).
* O split de sentenças usa regex pragmática para PT/EN; ajuste `sentence_split_pattern` se necessário.
* Use `set_prompt_template`, `set_sentences_per_batch`, `set_anonymization_config` para ajustar o comportamento em runtime.

