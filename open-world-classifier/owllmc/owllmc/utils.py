import json
import re
from tqdm import tqdm as tqdm_terminal
from tqdm.notebook import tqdm as tqdm_notebook
import sys
import pandas as pd

def parse_llm_json(text):
    # Remove blocos ```json e tenta carregar JSON
    cleaned = re.sub(r"```json|```", "", text).strip()
    print(cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError("Resposta da LLM não pôde ser interpretada como JSON.")


def get_tqdm():
    """
    Retorna a função tqdm apropriada dependendo do ambiente de execução.
    - tqdm.notebook.tqdm se estiver em Jupyter
    - tqdm.tqdm se estiver em terminal
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter Notebook ou JupyterLab
            return tqdm_notebook
        else:  # Terminal IPython ou outro
            return tqdm_terminal
    except NameError:
        return tqdm_terminal  # Terminal puro (sem IPython)


def load_prompt(path: str) -> str:
    with open(path, encoding='utf-8') as f:
        return f.read()

def load_classes_from_csv(path: str):
    df = pd.read_csv(path)
    return df.to_dict(orient="records")