"""
Agent class for the myagents package.
"""


from typing import Iterable, Union

from anthropic import Anthropic, Omit, omit
from anthropic.types import Message, TextBlockParam

from .tools.core import Tool


class AnthropicAgent:

    _name: str
    _client: Anthropic
    _model_id: str | None
    _system_prompt: Union[str, Iterable[TextBlockParam]] | Omit = omit
    _tools: list[Tool]
    _tool_descs: list[dict]
    _allowed_tools: list[str]

    def __init__(
            self, 
            name: str, 
            base_url: str | None, 
            model_id: str | None, 
            tools: list[Tool] = [],
            system_prompt: Union[str, Iterable[TextBlockParam]] | Omit = omit) -> None:
        self._name = name
        self._client = Anthropic(base_url=base_url)
        self._model_id = model_id
        self._system_prompt = system_prompt
        self._tools = tools
        self._tool_descs = [tool.to_anthropic_tool() for tool in tools]
        self._allowed_tools = [tool.name for tool in tools]

    @property
    def name(self) -> str:
        return self._name

    @property
    def tools(self) -> list[Tool]:
        return self._tools

    @property
    def allowed_tools(self) -> list[str]:
        return self._allowed_tools

    def step(self, inputs: list[dict]) -> Message:
        return self._client.messages.create(
            model=self._model_id, # type: ignore
            system=self._system_prompt,
            messages=inputs, # type: ignore
            tools=self._tool_descs, # type: ignore
            max_tokens=8000,
        )

    def close(self):
        self._client.close()
