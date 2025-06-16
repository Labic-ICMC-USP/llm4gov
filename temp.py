import json
from typing import Optional, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains.base import Chain

class ToolInput(BaseModel):
    """
    Input model for each tool step.
    - system_prompt: instructions to guide the LLM's behavior
    - user_prompt: main task question to be asked
    - input_dict: dictionary of structured context to be analyzed
    - output_schema: expected JSON schema (as string) for LLM output
    """
    system_prompt: str
    user_prompt: str
    input_dict: dict
    output_schema: str

class ToolOutput(BaseModel):
    """
    Output model for each tool step.
    - raw_output: raw string response from the LLM
    - json_output: parsed JSON dictionary (if successful)
    """
    raw_output: str
    json_output: Optional[dict]

class LLMGenericTool(BaseTool):
    """
    Generic LangChain-compatible Tool for LLM queries with structured prompts and output.
    """
    name: str
    description: str
    llm: Any  # Any langchain-compatible LLM with invoke() method

    def _run(self, system_prompt, user_prompt, input_dict, output_schema) -> ToolOutput:
        formatted_input = "\n".join([f"## {k}\n\n{v}" for k, v in input_dict.items()])
        full_user_prompt = f"{user_prompt}\n\n{formatted_input}"
        full_system_prompt = f"{system_prompt}\n\nThe output MUST be in this JSON format:\n{output_schema}"

        messages = [
            SystemMessage(content=full_system_prompt),
            HumanMessage(content=full_user_prompt)
        ]

        try:
            response = self.llm.invoke(messages).content
            parsed = json.loads(response)
            return ToolOutput(raw_output=response, json_output=parsed)
        except Exception:
            return ToolOutput(raw_output=response if 'response' in locals() else '', json_output=None)

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported.")

    def __call__(self, tool_input: ToolInput) -> ToolOutput:
        return self._run(**tool_input.model_dump())

class LLMToolChain(Chain):
    """
    Chain wrapper for sequential use of LLMGenericTool in LangChain pipelines.
    """
    tool: LLMGenericTool = Field()
    tool_input: ToolInput = Field()

    @property
    def input_keys(self):
        return []

    @property
    def output_keys(self):
        return [self.tool.name]

    def _call(self, inputs, run_manager=None):
        result = self.tool(self.tool_input)
        return {self.tool.name: result.json_output or {"raw_output": result.raw_output}}

    def _chain_type(self):
        return "llm_tool_chain"
