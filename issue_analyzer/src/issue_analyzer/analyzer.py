import json
from .connector import LLMConnector
from .schema import IssueOutput
from .logger import setup_logger

logger = setup_logger()

def clean_json_response(response: str) -> dict:
    """
    Try parsing the response directly. If it fails, attempt to extract JSON from markdown block or raw braces.
    """
    # 1. Primeira tentativa: tentar carregar diretamente
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 2. Tentar extrair entre blocos markdown ```json
    if "```json" in response:
        try:
            start = response.index("```json") + len("```json")
            end = response.index("```", start)
            cleaned = response[start:end].strip()
            return json.loads(cleaned)
        except (ValueError, json.JSONDecodeError):
            pass

    # 3. Procurar da primeira { até a última }
    try:
        start = response.index("{")
        end = response.rindex("}") + 1
        cleaned = response[start:end].strip()
        return json.loads(cleaned)
    except (ValueError, json.JSONDecodeError) as e:
        logger.error("Failed to clean and parse LLM response", error=str(e), raw_response=response)
        raise ValueError("Unable to parse JSON from response.") from e


class IssueAnalyzer:
    def __init__(self):
        self.connector = LLMConnector()



    def analyze(self, text: str) -> IssueOutput:
        logger.info("Starting analysis", input_text=text)
        try:
            response = self.connector.send(text)
            logger.info("Raw response received from LLM")

            parsed = clean_json_response(response)
            validated = IssueOutput(**parsed)
            logger.info("Output successfully validated")

            return validated
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON", error=str(e), raw_response=response)
            raise
        except Exception as e:
            logger.error("Unexpected error during issue analysis", error=str(e))
            raise
