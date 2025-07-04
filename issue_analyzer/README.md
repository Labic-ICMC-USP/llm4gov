# Issue Analyzer

**Issue Analyzer** is a modular Python package for detecting and classifying issues from short text inputs using a Large Language Model (LLM) via LangChain. It returns enriched JSON outputs with structured information, including severity, issue category, impacted entities, named entities, and reasoning for classification.

This tool is part of the larger **LLM4Gov** and **Websensors** initiative, focused on using language models for public sector applications, such as real-time issue detection, public monitoring, and data-driven policy design.

---

## Project Structure

```
issue_analyzer/
├── data/
│   ├── raw/                        # Raw input text files (e.g., CSV)
│   ├── processed/                  # LLM-processed output (JSON)
│
├── prompts/
│   └── issue_classifier_prompt.txt  # System prompt for the LLM
│
├── config/
│   └── llm_config.yaml              # Configuration for the LLM (model, keys, etc.)
│
├── notebooks/
│   └── example.ipynb                # Installation and usage walkthrough
│
├── src/
│   └── issue_analyzer/
│       ├── __init__.py
│       ├── analyzer.py              # Main class to analyze input text
│       ├── schema.py                # JSON output schema using Pydantic
│       ├── connector.py             # LangChain integration with LLM
│       ├── logger.py                # Structlog configuration
│       └── config_loader.py         # YAML config file loader
│
├── tests/
│   └── test_analyzer.py             # Unit tests for validation
│
├── cli.py                           # CLI tool for batch processing
├── setup.py                         # Installation script for the package
├── requirements.txt                 # Dependencies
└── README.md                        # Project documentation
```

---

## Installation

Clone this repository and install it in development mode:

```bash
git clone https://your-repo-url/issue_analyzer.git
cd issue_analyzer
pip install -e .
```

Make sure to configure your API access in `config/llm_config.yaml`.

---

## Usage (Python API)

```python
from issue_analyzer.analyzer import IssueAnalyzer

analyzer = IssueAnalyzer()

text = "Power outage in downtown São Paulo disrupted hospitals and transit systems."
result = analyzer.analyze(text)

print(result.json(indent=2, ensure_ascii=False))
```

This will return a structured JSON object containing classification results, including:

* `is_issue`: whether the input describes a negative issue
* `severity`: severity level and justification
* `category`: concise label and confidence
* `type`: domain and subcategory
* `named_entities`: extracted persons, locations, organizations, etc.
* `explanation`: reasoning and step-by-step rationale
* `meta`: processing metadata

---

## Usage (Command-Line Interface)

To analyze a batch of inputs from a CSV file:

```bash
python cli.py data/raw/issues_examples_rich.csv data/processed/results.json
```

The CSV file should have the following columns:

```
id,text
ex01,Power outage in São Paulo affected several districts.
...
```

The output will be a JSON file with one structured object per row.

---

## Usage (Jupyter Notebook)

See `notebooks/example.ipynb` for a complete walkthrough including:

* Installation
* API usage
* Batch CLI usage
* Output interpretation

---

## Dependencies

* langchain
* openai (or other LLM backends)
* structlog
* pydantic
* pyyaml
* pandas

Install with:

```bash
pip install -r requirements.txt
```

---

## License and Credits

This project is part of the LLM4Gov and [Websensors](https://websensors.icmc.usp.br) ecosystem for building language-based governance solutions. Contributions and integrations are welcome.

