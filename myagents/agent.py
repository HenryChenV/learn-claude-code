"""
Agent class for the myagents package.
"""


from typing import Iterable, Union

from anthropic import Anthropic, Omit, omit
from anthropic.types import Message, TextBlockParam

from .tools.core.provider import ToolProvider

from .tools.core import Tool


class Agent:

    _name: str
    _client: Anthropic
    _model_id: str | None
    _system_prompt: Union[str, Iterable[TextBlockParam]] | Omit = omit
    _allowed_tools: list[str]
    _max_tokens: int

    def __init__(
            self, 
            name: str, 
            base_url: str | None, 
            model_id: str | None, 
            allowed_tools: list[str] = [],
            system_prompt: Union[str, Iterable[TextBlockParam]] | Omit = omit,
            max_tokens: int = 8000) -> None:
        self._name = name
        self._client = Anthropic(base_url=base_url)
        self._model_id = model_id
        self._system_prompt = system_prompt
        self._allowed_tools = allowed_tools
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return self._name

    @property
    def allowed_tools(self) -> list[str]:
        return self._allowed_tools

    def step(self, inputs: list[dict], tools: list[Tool]) -> Message:
        return self._client.messages.create(
            model=self._model_id, # type: ignore
            system=self._system_prompt,
            messages=inputs, # type: ignore
            tools=[self._resolve_tool_desc(t) for t in tools], # type: ignore
            max_tokens=self._max_tokens,
        )

    def _resolve_tool_desc(self, tool: Tool):
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }


    def close(self):
        self._client.close()
