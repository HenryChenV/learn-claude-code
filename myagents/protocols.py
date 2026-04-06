"""Common Protocols
"""

from typing import Optional, Protocol, Self


class Spawnable(Protocol):

    def spawn(self, allow_sub_spawn: bool = False) -> Optional[Self]: 
        """spawn children

        Args:
            allow_sub_spawn (bool, optional): if allow child to spawn. Defaults to False.

        Returns:
            Optional[Self]: non-None means successful, None means spawn is not allowed
        """
        ...