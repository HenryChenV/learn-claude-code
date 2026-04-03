""" ToolsProvider"
"""

from abc import abstractmethod
from typing import Protocol

from .tool import Tool


class ToolProvider(Protocol):

    def get_tools(self) -> list[Tool]: ...