from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from .config_loader import load_config
from .logger import setup_logger

logger = setup_logger()

class LLMConnector:
    def __init__(self):
        config = load_config()
        self.llm = ChatOpenAI(
            model=config["model"],
            temperature=config.get("temperature", 0.2),
            openai_api_key=config["api_key"],
            openai_api_base=config.get("api_base")
        )
        with open(config["system_prompt_path"], "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def send(self, input_text: str) -> str:
        logger.info("Sending input to LLM")
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=input_text)
        ]
        response = self.llm.invoke(messages)
        logger.info("Response received from LLM")
        return response.content
