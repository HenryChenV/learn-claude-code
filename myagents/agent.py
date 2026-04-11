"""
Agent class for the myagents package.
"""


from typing import Generator, Iterable, Literal, Optional, Self, Union
from anthropic.types import Message, MessageParam, TextBlockParam

from .tools.core.provider import ToolProvider
from .skill import SkillMeta, SkillProvider
from .ids import HierarchicalID
from .capability import CapabilityRule
from .tools.core import ToolMeta
from .events import *
from .chat_model import ModelSpec


class AgentRunHooks:

    def post_step(self, resp: Message) -> Generator[Event, None, Optional[bool]]:
        """hook before loop end

        If some messages are appended, you may want the loop continues.

        Returns:
            bool: 
                True: completed
                False: continue
                None: determined by caller
        """
        yield from []
        return None

class AgentRunContext(AgentRunHooks, ABC):

    @abstractmethod
    def get_inputs(self) -> Iterable[MessageParam]:
        """get inputs
        """
        pass

    @abstractmethod
    def append_message(self, role: Literal["user", "assistant"], content):
        """append message to conversation
        """
        pass

    @abstractmethod
    def resolve_skills(self, 
                       allowed_capabilities: list[CapabilityRule],
                       extra_providers: Iterable[SkillProvider] = []) -> Iterable[SkillMeta]:
        """resolve skills by capabilities

        Args:
            allowed_capabilities (list[CapabilityRule]): capabilities

        Returns:
            tuple[str, ...]: (name, description) of skills
        """
        pass

    @abstractmethod
    def is_skill_use(self, tool_name: str) -> bool:
        """if the given tool is to use skill
        """
        pass

    @abstractmethod
    def use_skill(self, 
                  allowed_capabilities: list[CapabilityRule], 
                  skill_kwargs: dict, 
                  extra_providers: Iterable[SkillProvider] = []) -> str:
        """use skill

        Args:
            allowed_capabilities (list[CapabilityRule]): allowed capabilities
            skill_name (str): name of the skill
            extra_providers (Iterable[ToolProvider], optional): extra providers

        Returns:
            _type_: content of the skill
        """
        pass

    @abstractmethod
    def resolve_tools(self, 
                      allowed_capabilities: list[CapabilityRule], 
                      extra_providers: list[ToolProvider] = []) -> Iterable[ToolMeta]:
        """resolve tools by allowed capabilities
        """
        pass

    @abstractmethod
    def use_tool(self, 
                 allowed_capabilities: list[CapabilityRule], 
                 tool_name: str, 
                 tool_kwargs: dict, 
                 extra_providers: Iterable[ToolProvider] = []) -> str:
        """use tools subjected to allowed capabilities
        """
        pass

    @abstractmethod
    def extend_paths(self, *parts: str) -> list[str]:
        """extend paths to the paths in context and return full path
        """
        pass

    @abstractmethod
    def get_final_message(self) -> str:
        """get final message
        """
        pass

    def get_evnet_extra(self) -> dict[str, Any]:
        """get extra info as map for event
        """
        return {}


class AgentID(HierarchicalID):
    pass


class Agent:
    """
    Agent

    Agents are like persons with a specific role, 
    having their own preference, capabilities, etc.

    Attributes:
        _name (str): name
        _system_prompt (str): system prompt, define its preference in nature languange.
        _allowed_capabilities (list[str]): capabilities
    """

    def __init__(
            self, 
            aid: Union[str, AgentID], 
            models: Iterable[ModelSpec],
            allowed_capabilities: list[str] = [],
            sys_prompt: Optional[str] = None,
            max_tokens: int = 8000) -> None:
        self._aid: AgentID = AgentID.wrap(aid)
        self._models: Iterable[ModelSpec] = models
        self._sys_prompt: Iterable[TextBlockParam] = \
            [{"type": "text", "text": sys_prompt}] if sys_prompt else []
        self._allowed_capabilities: Iterable[CapabilityRule] = [CapabilityRule.wrap(c) for c in allowed_capabilities]
        self._max_tokens: int = max_tokens

    def spawn(self, allow_sub_spawn: bool = False) -> Self:
        if allow_sub_spawn:
            capabilities = [c.raw for c in self._allowed_capabilities]
        else:
            capabilities = [c.raw for c in self._allowed_capabilities] + ["!subagent.spawn"]
        return self.__class__(
            self._aid.spawn(),
            models=self._models,
            allowed_capabilities=capabilities,
            max_tokens=self._max_tokens
        )

    @property
    def name(self):
        return self._aid.name

    @property
    def allowed_capabilities(self):
        return self._allowed_capabilities

    @property
    def sys_prompt(self):
        return self._sys_prompt

    @property
    def models(self):
        return self._models

    @property
    def max_tokens(self):
        return self._max_tokens

    @property
    def event_extra(self) -> dict[str, Any]:
        return {}
