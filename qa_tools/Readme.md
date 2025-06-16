# qa_tools/sequential_qa.py

This module is part of the `qa_tools` package within the broader `LLM4Gov` project. It provides reusable components for building **sequential question-answering pipelines** using LangChain-compatible Large Language Models (LLMs). The goal is to enable structured decision-making where each step involves asking an LLM a specific question based on structured data and receiving a JSON-compliant answer.

---

## 🎯 Purpose

Use `sequential_qa.py` when you need to:

- Ask a **sequence of structured questions** to an LLM
- Enforce **JSON schema compliance** in each response
- Aggregate all results in a form that is easily integrable with external systems (APIs, dashboards, databases, etc.)

This is especially useful in public administration contexts, where decisions must be reproducible, auditable, and automatable — as in the case of `LLM4Gov`.

---

## 📁 File Location

To import from another module in the `LLM4Gov` project:

```python
from qa_tools.sequential_qa import ToolInput, ToolOutput, LLMGenericTool, LLMToolChain
````

---

## ✅ Features

* Structured input (`ToolInput`) and output (`ToolOutput`)
* Generic tool execution using any LangChain-compatible LLM (`LLMGenericTool`)
* Simple integration with `SequentialChain` using `LLMToolChain`
* Supports OpenAI, Ollama, and other LLMs that implement `.invoke()` with message inputs

---

## 📥 Input Example

```python
ToolInput(
    system_prompt="You are an assistant checking pension eligibility.",
    user_prompt="Based on the data below, determine if the applicant qualifies:",
    input_dict={
        "Name": "Maria",
        "Age": "63",
        "Status": "Retired civil servant"
    },
    output_schema=\"\"\"
    {
        "eligible": true,
        "reason": "Meets retirement and employment criteria."
    }
    \"\"\"
)
```

---

## 📤 Output Example

```python
ToolOutput(
    raw_output="{\"eligible\": true, \"reason\": \"Meets criteria.\"}",
    json_output={
        "eligible": True,
        "reason": "Meets criteria."
    }
)
```

If the output cannot be parsed as JSON, `json_output` will be `None`.

---

## 🔁 Sequential Usage

You can create multiple tools and run them in sequence using `SequentialChain`:

```python
from langchain.chains import SequentialChain
from qa_tools.sequential_qa import LLMGenericTool, LLMToolChain, ToolInput

# One tool step
tool = LLMGenericTool(name="step1", description="Check step 1", llm=llm)
input = ToolInput(...)

chain = LLMToolChain(tool=tool, tool_input=input)

# Combine multiple chains if needed
seq = SequentialChain(
    chains=[chain],
    input_variables=[],
    output_variables=["step1"],
    verbose=True
)
result = seq.invoke({})
```

---

## 🧪 Dependencies

See `requirements.txt`. Main dependencies:

* `langchain>=0.1.7`
* `langchain-openai>=0.1.6`
* `langchain-community`
* `openai`
* `pydantic>=2.0`

---

