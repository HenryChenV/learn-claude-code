"""Common
"""


from pathlib import Path
from typing import Protocol


WORKDIR = Path.cwd()


class HumanInput(Protocol):

    def input(self, prompt: str) -> str: ...