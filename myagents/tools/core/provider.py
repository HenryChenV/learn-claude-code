""" ToolsProvider"
"""

from abc import abstractmethod
from typing import Iterable, Protocol

from .tool import Tool


class ToolProvider(Protocol):

    def get_tools(self) -> Iterable[Tool]: 
        """get all tools

        Returns:
            all tools
        """
        ...
