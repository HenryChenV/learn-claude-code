"""
Tools for the myagents package.
"""

from typing import Iterable, Optional, Self

from .core import Tool
from .impls import *


BUILDIN_TOOLS: list[Tool] = [
    run_bash, 
    edit_file, 
    write_file, 
    read_file
]


class BuildinToolProvider:

    _singleton: Optional['BuildinToolProvider'] = None

    def __init__(self, tools: Iterable[Tool]):
        self._tools: Iterable[Tool] = tools

    @classmethod
    def get_instance(cls) -> 'BuildinToolProvider':
        if cls._singleton is None:
            cls._singleton = cls(BUILDIN_TOOLS)
        return cls._singleton

    def get_tools(self) -> Iterable[Tool]:
        return self._tools
