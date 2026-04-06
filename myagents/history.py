"""Conversation History
"""


from typing import Iterable, Literal, Optional

from anthropic.types import MessageParam


class History:

    _messages: list[MessageParam]

    def __init__(self):
        self._messages = []

    def append(self, role: Literal["user", "assistant"], content):
        self._messages.append({"role": role, "content": content})

    @property
    def messages(self) -> Iterable[MessageParam]:
        return tuple(self._messages)

    @property
    def latest(self) -> Optional[MessageParam]:
        if self._messages:
            return self._messages[-1]
        return None