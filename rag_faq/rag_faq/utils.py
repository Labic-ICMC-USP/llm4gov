import json

def load_prompt_template(path):
    """Load prompt template from a text file"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def format_prompt(persona, rules, text, k):
    """
    Construct the system and user prompts for LangChain,
    enforcing JSON output format with a specific schema.
    """
    system_prompt = f"""{persona}

{rules}

Retorne exatamente k = {k} itens.
"""
    user_prompt = f"""Base text:
\"\"\"{text}\"\"\""""

    return {"system": system_prompt, "user": user_prompt}

def parse_faq_response(response_text, k):
    """
    Parse LLM response that should be a JSON list of {question, answer} pairs.
    """
    try:
        data = json.loads(response_text.replace("```json","").replace("```",""))
        if isinstance(data, list):
            return data[:k]
        else:
            print(response_text)
            raise ValueError("Parsed object is not a list")
    except json.JSONDecodeError as e:
        print(response_text)
        raise ValueError(f"Invalid JSON returned by LLM: {e}")
