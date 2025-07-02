
import json
import time
from typing import List, Dict, Union
from langchain.schema import SystemMessage, HumanMessage
from .utils import parse_llm_json
from .utils import get_tqdm

class OWLLMc:
    def __init__(self, llm_generalist, llm_specialist, system_prompt_generalist, system_prompt_specialist, max_retries=3):
        self.llm_generalist = llm_generalist
        self.llm_specialist = llm_specialist
        self.system_prompt_generalist = system_prompt_generalist,
        self.system_prompt_specialist = system_prompt_specialist,
        self.max_retries = max_retries
        self.classes = []
        self._history = []

    def fit(self, class_data: List[Dict]):
        self.classes = class_data

    def _run_llm(self, llm, system_message: str, user_message: dict) -> Union[List[Dict], None]:
        for attempt in range(self.max_retries):
            try:
                print(system_message)
                print(user_message)
                messages = [
                    SystemMessage(content=json.dumps(system_message)),
                    HumanMessage(content=json.dumps(user_message))
                ]
                response = llm.invoke(messages)
                parsed = parse_llm_json(response.content)
                if isinstance(parsed, list):
                    return parsed
            except Exception as e:
                print(f"[Tentativa {attempt+1}/{self.max_retries}] Erro na chamada LLM: {e}")
                time.sleep(1)
        return None  # falha após retries

    def predict(self, texts: Union[str, List[str]]) -> List[Dict]:
        if isinstance(texts, str):
            texts = [texts]

        results = []

        tqdm = get_tqdm()

        for text in tqdm(texts):
            history_entry = {
                "input": text,
                "generalist_output": None,
                "specialist_output": None,
                "status": "success"
            }

            # Estágio 1 - Generalista
            generalist_input = {
                "document": text,
                "labels": self.classes
            }

            generalist_output = self._run_llm(self.llm_generalist, self.system_prompt_generalist, generalist_input)
            history_entry["generalist_output"] = generalist_output

            if not generalist_output:
                history_entry["status"] = "generalist_failed"
                self._history.append(history_entry)
                results.append({"error": "Falha no estágio generalista"})
                continue

            # Estágio 2 - Especialista
            specialist_input = {
                "document": text,
                "labels": generalist_output
            }

            specialist_output = self._run_llm(self.llm_specialist, self.system_prompt_specialist, specialist_input)
            history_entry["specialist_output"] = specialist_output

            if not specialist_output:
                history_entry["status"] = "specialist_failed"
                results.append({"error": "Falha no estágio especialista", "top_k": top_k})
            else:
                results.append(specialist_output)

            self._history.append(history_entry)

        return results

    def history(self) -> List[Dict]:
        return self._history
