# LLM4Gov: Exploring Large Language Models for Public Sector Applications

**LLM4Gov** is an open research and development framework that enables the safe, efficient, and privacy-preserving use of **Large Language Models (LLMs)** in the public sector.  
Its goal is to bring the capabilities of LLMs — reasoning, classification, summarization, and retrieval — to sensitive government and institutional contexts, where data confidentiality, explainability, and local execution are essential.

The project promotes the use of **smaller, fine-tuned, and locally deployable models** to ensure:
- **Data sovereignty** — sensitive information never leaves the organization’s infrastructure.  
- **Reproducibility** — all steps (from data preparation to inference) can be executed with open tools.  
- **Cost efficiency** — reduced hardware requirements via LoRA and 4-bit quantization.  
- **Transparency** — auditable and explainable model outputs for policy-making and research.

---

## Repository structure

The repository is organized into modular subprojects, each addressing a key aspect of how LLMs can assist in public-sector data analysis and decision support.

### [`anonymizer/`](./anonymizer)
This module contains an LLM-based anonymization system that produces a privacy-preserving version of an input text.  
It replaces personal or sensitive information according to configurable rules defined by the organization, ensuring that downstream models process only safe, compliant data.

---

### [`issue_analyzer/`](./issue_analyzer)
Focuses on extracting and mapping **issues** (problems, risks, alerts) from reports and news articles.  
The system identifies possible root causes, constructs a **risk matrix**, and generates structured diagnostics — supporting early detection of operational or policy challenges.

---

### [`open-world-classifier/`](./open-world-classifier)
A general-purpose **open-world classification** framework.  
It uses a two-stage pipeline:
1. A **generalist classifier** ranks the most likely classes from a large initial set.  
2. A **specialist classifier** (another LLM) refines the final decision.  
This approach allows organizations to define broad, extensible class taxonomies while maintaining accuracy and scalability.

---

### [`rag_faq/`](./rag_faq)
Implements a **Retrieval-Augmented Generation (RAG)** pipeline for FAQ creation and intelligent question answering.  
First, a set of Q&A pairs is generated from an organization’s textual knowledge base within a defined **persona** context.  
Then, this Q&A corpus is indexed so that new user questions retrieve the most relevant entries and receive augmented, contextual responses from an LLM.

---

### [`teacher_student/`](./teacher_student)
Implements the **Teacher–Student fine-tuning framework** central to LLM4Gov.  
A large *Teacher* LLM generates labeled training data, while a compact *Student* model is fine-tuned locally with LoRA and 4-bit quantization.  
The result is a smaller, private model capable of performing the same task securely on-premise — enabling safe AI deployment inside government or institutional infrastructure.

## Learn more
For detailed usage instructions, configuration examples, and step-by-step tutorials, see the documentation inside each subdirectory.
