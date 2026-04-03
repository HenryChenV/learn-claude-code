"""Conversation History
"""


from typing import Literal, Sequence


class History:

    _messages: list[dict]

    def __init__(self):
        self._messages = []

    def append(self, role: Literal["user", "assistant"], content):
        self._messages.append({"role": role, "content": content})

    @property
    def messages(self) -> Sequence[dict]:
        return tuple(self._messages)