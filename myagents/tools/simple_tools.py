"""
Simple tools for the myagents package.
"""


from typing import Callable

from .base import Tool


class SimpleTool(Tool):

    def __init__(self, name: str, description: str, input_schema: dict, func: Callable) -> None:
        super().__init__(name=name, description=description, input_schema=input_schema)
        self._func = func

    def _run(self, **kwargs) -> str:
        return self._func(**kwargs)