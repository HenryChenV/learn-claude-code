"""Conversation 
"""


from typing import Iterable, Literal, Optional

from anthropic.types import ContentBlock, MessageParam, ToolResultBlockParam, ToolUseBlockParam
from rich.prompt import PromptError


class Conversation:

    _messages: list[MessageParam]

    def __init__(self):
        self._messages = []

    def append_user_prompt(self, prompt: str) -> None:
        self._messages.append({"role": "user", "content": prompt})

    def append_assistant_content(self, 
                                 content: Iterable[ContentBlock]) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def append_tool_use_result(self, 
                               tool_use_id: str, 
                               tool_output: str) -> None:
        block: ToolResultBlockParam = {
            "type": "tool_result", 
            "tool_use_id": tool_use_id, 
            "content": tool_output
        }
        if self._messages \
                and self._messages[-1]["role"] == "user" \
                and isinstance(self._messages[-1]["content"], list):
            self._messages[-1]["content"].append(block)

        else:
            self._messages.append({"role": "user", "content": [block]})

    @property
    def messages(self) -> Iterable[MessageParam]:
        return tuple(self._messages)

    @property
    def latest(self) -> Optional[MessageParam]:
        if self._messages:
            return self._messages[-1]
        return None