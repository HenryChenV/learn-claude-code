"""ID
"""


from datetime import datetime
from typing import Optional, Self, Type, Union


class HierarchicalID:

    _id: str
    _name: str
    _parent: Optional[Self]
    _created_at: datetime
    _sub_counter: int

    @classmethod
    def wrap(cls: Type[Self], sid: Union[str, Self]) -> Self:
        if isinstance(sid, str):
            return cls(sid)
        if isinstance(sid, cls):
            return sid
        raise ValueError(
            f"Unsupported type {type(sid)} from {sid} for SessionID")

    def __init__(self, name, parent: Optional[Self] = None):
        self._name = name
        self._parent = parent
        self._created_at = datetime.now()
        self._sub_counter = 0
        self._id = self._gen_id()

    def _gen_id(self) -> str:
        paths = [self._name]
        parent = self._parent
        while parent:
            paths.append(parent.name)
            parent = parent._parent
        return ":".join(reversed(paths))

    def spawn(self, allow_sub_spawn: bool = False) -> Self:
        self._sub_counter += 1
        return self.__class__(f"sub{self._sub_counter}", self)

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return self._id

    @property
    def parent(self):
        return self._parent

    @property
    def created_at(self):
        return self._created_at

    @property
    def is_root(self) -> bool:
        return self._parent is None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.id}/{self.created_at})"

    __repr__ = __str__