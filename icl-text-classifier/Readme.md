# In-Context Learning Text Classification

`icl-text-classifier` is a small Python package that performs **zero-shot / few-shot text classification**
using a Large Language Model (LLM) in an **in-context learning (ICL)** setup.

The package is designed for experiments where you:
- Have a list of **high-level classes** (e.g., *health*, *education*, *public_safety*).
- Have a CSV file with texts to classify (e.g., short news, policy descriptions, user feedback).
- Want the LLM to decide which classes are **highly relevant** for each text and **justify** its choice.

All configuration is done via a simple **YAML file**, so you can change:
- The underlying LLM model (e.g., via OpenRouter).
- The base URL and API key.
- The list of classes and their descriptions.
- The CSV input path and number of threads.
- The output JSONL path.

---

## Why use this model?

Instead of training a traditional supervised classifier (which requires labeled data), this package:
- Uses an LLM to **infer labels on demand**, directly from the text.
- Allows **multiple labels per text** (multi-label classification).
- Returns **human-readable justifications** for each assigned class.
- Is easy to plug into existing pipelines because input/output are just **CSV + JSONL**.

Typical use cases:
- Rapid prototyping of taxonomies for public policies (health, education, security, governance, etc.).
- Weak-label generation for later training of classical models.
- Qualitative exploration of how an LLM separates themes in a corpus.

---

## Installation

From a local clone of this project:

```bash
pip install .
```

Or, in editable (development) mode:

```bash
pip install -e .
```

The main dependencies are:
- `langchain-openai` for the LLM client;
- `pydantic` for validating the JSON response schema;
- `PyYAML` for configuration;
- `tqdm` for progress bars.

---

## Configuration

All settings are defined in a YAML file, for example:

```yaml
model:
  name: "mistralai/mistral-nemo"
  base_url: "https://openrouter.ai/api/v1"
  api_key: "YOUR_OPENROUTER_API_KEY"
  temperature: 0.0

classification:
  system_prompt: |
    You are an expert classifier that assigns highly relevant classes
    to each input text, based on public policy themes.

    The predefined classes are:
    {CLASSES_DESCRIPTION}

    Rules:
    - Only assign a class if it is clearly supported by the text.
    - You may assign MORE THAN ONE class if they are all highly relevant.
    - If no class is clearly relevant, return an empty list.
    - Use only the class_id values provided.

  classes:
    - id: "health"
      description: >
        Texts about hospitals, clinics, primary care, emergency rooms,
        doctors, nurses, vaccination campaigns, mental health support,
        telemedicine, chronic disease monitoring, mobile clinics, and
        other healthcare services or policies.

    - id: "education"
      description: >
        Texts about schools, universities, teachers, curricula, exams,
        remote learning platforms, tutoring, literacy programs, scholarships,
        educational infrastructure, and teacher training.

    - id: "public_safety"
      description: >
        Texts about crime, robberies, assaults, domestic violence, drug
        trafficking, policing strategies, community policing, surveillance
        cameras, patrols, checkpoints, and measures to improve safety in
        public spaces or transport.

    - id: "governance"
      description: >
        Texts about justice and legal systems, anti-corruption campaigns,
        transparency, reporting misuse of public funds, and citizen oversight
        of government actions.

  csv_input_path: "examples/input.csv"
  id_column: "ID"
  text_column: "TEXT"

  num_threads: 4
  max_tries: 3

  output_path: "examples/output_classification.jsonl"
```

---

## How it works

For each row in the CSV file:

1. The text is placed inside a **user prompt** that lists all predefined classes and instructions.
2. The LLM is called with a **system prompt** (from YAML) and the constructed user prompt.
3. The LLM must return a JSON object that matches the schema:

   ```json
   {
     "relevant_classes": [
       {
         "class_id": "health",
         "justification": "The text mentions hospital, nurses and vaccination."
       }
     ]
   }
   ```

4. The JSON is validated with **Pydantic**. If parsing or validation fails, the request is retried
   up to `max_tries`. If all attempts fail, an empty list of `relevant_classes` is returned for that document.
5. All results are saved to a **JSONL file** (one JSON per line).

---

## Usage

### 1. CLI

After installing the package (`pip install .`), you can run:

```bash
icl-classify --config path/to/config.yaml
```

If `--config` is omitted, it defaults to `config.yaml` in the current directory.

The output JSONL file path is defined in the YAML, e.g. `classification.output_path`.

### 2. Python API

You can also use the classifier directly from Python:

```python
from icl_classifier import ICLClassifier
import json

classifier = ICLClassifier(config_path="config.yaml")
results = classifier.run()
classifier.save_results(results)  # uses output_path from YAML

print(json.dumps(results[:3], ensure_ascii=False, indent=2))
```

---

## Example

This repository ships with:

- `examples/input.csv`: small CSV with synthetic short texts.
- `examples/config.yaml`: configuration file ready to run (you only need to fill your API key).

From the project root, after setting your API key in `examples/config.yaml`, run:

```bash
pip install .
icl-classify --config examples/config.yaml
```

This will create `examples/output_classification.jsonl` with one JSON object per document.

---

## Short note in Portuguese

Este pacote implementa um classificador de textos baseado em *in-context learning* usando LLMs.
A ideia é evitar a necessidade de um conjunto de treino rotulado, permitindo que um modelo de linguagem
grande atribua rótulos temáticos diretamente aos textos, com justificativas. A configuração é feita via YAML
e os textos são lidos de um CSV, o que facilita a integração com pipelines existentes de ciência de dados.
