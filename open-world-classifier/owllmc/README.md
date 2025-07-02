# OWLLMc: Open-World LLM-based Classifier

`OWLLMc` is a two-stage, open-world text classification framework that uses large language models (LLMs) to first filter relevant candidate classes and then refine the classification using a specialist model. This architecture is especially useful when working with thousands of potential categories or evolving class taxonomies.

## Key Features

* **Two-stage inference**: Generalist model filters top-k likely classes; specialist model provides a final decision.
* **Zero-shot/Few-shot capability**: No need for fine-tuning. Just provide your class definitions and prompt templates.
* **LLM agnostic**: Works with any LangChain-compatible LLMs (e.g., OpenAI, Azure, local models).
* **Transparent decision history**: The full prediction trace (inputs/outputs of both stages) is available for inspection.

---

## Installation

```bash
git clone https://github.com/Labic-ICMC-USP/llm4gov/
cd llm4gov/open-world-classifier/owllmc
pip install .
```

---

## Configuration

Edit the `config.yaml` file to define the following:

```yaml
llm:
  generalist:
    model: "gpt-4"
    api_key: "your-openai-key"
    api_base: "https://api.openai.com/v1"
  specialist:
    model: "gpt-4"
    api_key: "your-openai-key"
    api_base: "https://api.openai.com/v1"

paths:
  prompt_generalist: "data/prompt_generalist.txt"
  prompt_specialist: "data/prompt_specialist.txt"
  classes_csv: "data/classes.csv"
```

---

## Input Files

* `classes.csv`: A list of classes and optional descriptions.
* `input_texts_examples.csv`: Texts to be classified.
* Prompt files (`prompt_generalist.txt`, `prompt_specialist.txt`) define the system prompt used by each LLM stage.

---

## How to Use

```python
import yaml
import pandas as pd
from langchain.chat_models import ChatOpenAI
from owllmc.core import OWLLMc
from owllmc.utils import load_prompt, load_classes_from_csv

# Load configuration
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Initialize LLMs
llm_generalist = ChatOpenAI(**config["llm"]["generalist"], temperature=0)
llm_specialist = ChatOpenAI(**config["llm"]["specialist"], temperature=0)

# Load prompts and classes
system_prompt_generalist = load_prompt(config["paths"]["prompt_generalist"])
system_prompt_specialist = load_prompt(config["paths"]["prompt_specialist"])
classes = load_classes_from_csv(config["paths"]["classes_csv"])

# Initialize the classifier
owllmc = OWLLMc(
    llm_generalist=llm_generalist,
    llm_specialist=llm_specialist,
    system_prompt_generalist=system_prompt_generalist,
    system_prompt_specialist=system_prompt_specialist,
)
owllmc.fit(classes)

# Run prediction
df_inputs = pd.read_csv("data/input_texts_examples.csv")
text_inputs = df_inputs["text"].tolist()
results = owllmc.predict(text_inputs)

# Output results
print("Final Prediction (Specialist):", results)
print("\nPrediction History:", owllmc.history())
```

---

## Output Format

Each prediction contains the final output from the specialist model, and the `history()` method provides full trace logs including:

* Input text
* Candidate classes from generalist
* Final prediction from specialist
* Status (success, failure in generalist/specialist)


Aqui está a seção de agradecimentos em inglês com a referência em LaTeX/BibTeX formatada corretamente para ser usada dentro do `README.md`:

---

## Acknowledgments

This work was inspired by the research presented in:

ZITEI, Daniel Pereira; SAKIYAMA, Kenzo; MARCACINI, Ricardo Marcondes. *Open-world text classification by combining weak models and large language models*. Proceedings of the **ENIAC 2024 – National Meeting on Artificial Intelligence**, Belém-PA, Brazil. SBC – Brazilian Computer Society, 2024.


```bibtex
@inproceedings{zitei2024openworld,
  author    = {Daniel Pereira Zitei and Kenzo Sakiyama and Ricardo Marcondes Marcacini},
  title     = {Open-world text classification by combining weak models and large language models},
  booktitle = {Proceedings of the ENIAC 2024 – National Meeting on Artificial Intelligence},
  year      = {2024},
  address   = {Belém-PA, Brazil},
  publisher = {Brazilian Computer Society (SBC)}
}
```


