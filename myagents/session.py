"""
Session management for myagents.
"""


from functools import cached_property
from mailbox import Message
from typing import Iterable, Literal, Optional, Protocol, Self, Any

from anthropic.types import Message, MessageParam

from .chat_model import ChatModel, ChatModelManager, ModelSpec
from .skill import SkillManager, SkillMeta, SkillProvider

from .ids import HierarchicalID
from .tools.core.tool import FunctionTool, ToolMeta

from .capability import Capability, CapabilityRule


from .agent import Agent
from .tools.core import Tool, ToolManager
from .tools.core.provider import ToolProvider
from .events import *
from .conversation import Conversation


class SessionID(HierarchicalID):

    pass


class Session:

    def __init__(self, 
                 sid: str | SessionID,
                 agent: Agent, 
                 model_manager: Optional[ChatModelManager] = None,
                 tool_providers: list[ToolProvider] = [],
                 skill_providers: list[SkillProvider] = [],
                 middleware_factories: list['SessionMiddlewareFactory'] = [],
                 event_bus: Optional[EventBus] = None,
                 user="You",
                 theme_color="") -> None:

        self._sid: SessionID = SessionID.wrap(sid)
        self._agent: Agent = agent
        self._model_manager = model_manager or ChatModelManager.get_default()
        self._tool_providers: list[ToolProvider] = tool_providers
        self._skill_providers: list[SkillProvider] = skill_providers
        self._middleware_factories: list[SessionMiddlewareFactory] = middleware_factories
        self._user: str = user
        self._theme_color: str = theme_color

        self._round_counter: int = 0
        self._conversation: Conversation = Conversation()

        self._tool_manager: ToolManager = ToolManager(initial_providers=tool_providers + [self])
        self._skill_manager: SkillManager = SkillManager(initial_providers=skill_providers)

        self._event_bus: EventBus = event_bus or EventBus()

        # create middlewares
        self._middlewares: list[SessionMiddleware] = []
        for factory in middleware_factories:
            obj = factory.create(self)
            if obj:
                self._middlewares.append(obj)

        self._post_init()

    def _post_init(self):
        for middleware in self._middlewares:
            middleware.post_session_init(self)

    def spawn(self, allow_sub_spawn: bool = False) -> Self:
        subsession = self.__class__(
            sid=self._sid.spawn(),
            agent=self._agent.spawn(),
            model_manager=self._model_manager,
            tool_providers=self._tool_providers,
            skill_providers=self._skill_providers,
            middleware_factories=self._middleware_factories,
            user=self._sid.name,
        )
        return subsession

    def subscribe(self, subscriber: EventSubscriber) -> None:
        """subscribe events
        """
        self._event_bus.add_subscriber(subscriber)

    def publish(self, *events: Event) -> None:
        """publish event
        """
        for event in events:
            self._event_bus.publish(event)

    def append_user_prompt(self, prompt: str) -> None:
        self._round_counter += 1
        self._conversation.append_user_prompt(prompt=prompt)
        self.publish(UserPromptEvent(
            source_name=self._user,
            paths=self.extend_paths(), 
            prompt=prompt, 
            extra=self.event_extra
        ))

    def append_assistant_content(self, 
                                 content: Iterable[ContentBlock], 
                                 *,
                                 paths:list[str], 
                                 agent_name: str, 
                                 extra={},
                                 exlcluded_blocks: set[str] = set()) -> None:
        self._conversation.append_assistant_content(content=content)
        for event in EventFactory.generate(content=content, 
                                           paths=paths, 
                                           agent_name=agent_name, 
                                           extra=extra,
                                           exlcluded_blocks=exlcluded_blocks):
            self.publish(event)

    def append_tool_use_result(self, 
                               tool_use_id: str, 
                               tool_name: str,
                               tool_output: str, 
                               *,
                               paths:list[str], 
                               extra={}) -> None:
        self._conversation.append_tool_use_result(
            tool_use_id=tool_use_id,
            tool_output=tool_output
        )
        if self.is_skill_use(tool_name):
            self.publish(SkillResultEvent(
                paths=paths,
                tool_name=tool_name, 
                tool_use_id=tool_use_id, 
                output=tool_output,
                extra=extra,
            ))
        else:
            self.publish(ToolResultEvent(
                paths=paths,
                tool_name=tool_name, 
                tool_use_id=tool_use_id, 
                output=tool_output,
                extra=extra,
            ))

    def resolve_tools(self, 
                      allowed_capabilities: Iterable[CapabilityRule],
                      extra_providers: Iterable[ToolProvider] = []) -> Iterable[ToolMeta]:
        return self._tool_manager.resolve_tools(allowed_capabilities, extra_providers)

    def resolve_skills(self, 
                       allowed_capabilities: Iterable[CapabilityRule], 
                       extra_providers: Iterable[SkillProvider] = []) -> Iterable[SkillMeta]:
        return self._skill_manager.resolve_skills(allowed_capabilities, extra_providers)

    def resolve_model(self, 
                      model_full_ids: Iterable[str], 
                      excluded: set[ModelSpec]=set()) -> Optional[ChatModel]:
        for full_id in model_full_ids:
            if full_id in excluded:
                continue
            model = self._model_manager.get_model(full_id)
            if not model:
                continue
            return model
        return None

    def use_skill(self, 
                  allowed_capabilities: Iterable[CapabilityRule], 
                  skill_kwargs: dict, 
                  extra_providers: Iterable[SkillProvider] = []) -> str:
        try:
            self.skill_use.validate(skill_kwargs)
            return self._skill_manager.get_content(
                allowed_capabilities=allowed_capabilities,
                skill_name=str(skill_kwargs.get("skill_name")),
                extra_providers=extra_providers
            )
        except Exception as e:
            return f"Error: Failed to use skill with '{skill_kwargs}': {e}"

    def is_skill_use(self, tool_name: str) -> bool:
        return self.skill_use.meta.name == tool_name
        
    def use_tool(self, 
                allowed_capabilities: Iterable[CapabilityRule], 
                tool_name: str, 
                tool_kwargs: dict,
                extra_providers: Iterable[ToolProvider] = []) -> str:
        try:
            return self._tool_manager.execute(
                allowed_capabilities, 
                tool_name, 
                tool_kwargs, 
                extra_providers=extra_providers
            )
        except Exception as e:
            return f"Error: Failed to use tool '{tool_name}': {e}"

    def get_tools(self) -> Iterable[Tool]:
        return [
            self.skill_use,
        ]

    @cached_property
    def skill_use(self) -> Tool:
        @FunctionTool.wrapper(required_capabilities=Capability.SKILL_USE.value)
        def use_skill(skill_name: str) -> str:
            """load content of the skill by name 
            if the description matches your needs.
            """
            raise RuntimeError(
                "The tool should not be used in this way. "
                "It MUST be a bug. Ask user to fix it"
            )
        return use_skill

    def post_step(self, resp: Message) -> bool:
        """invoked after step
        Return: if the loop shoue continue
        """
        continue_loop = False
        for middleware in self._middlewares:
            if middleware.post_agent_step(self, resp):
                continue_loop = True
        return continue_loop

    def add_tool_provider(self, provider: ToolProvider):
        self._tool_manager.add_provider(provider)

    def extend_paths(self, *parts: str) -> list[str]:
        return [f"session:{self.sid.id}", f"round:{self.rounds}"] + list(parts)

    @property
    def rounds(self) -> int:
        return self._round_counter

    @property
    def messages(self) -> Iterable[MessageParam]:
        return self._conversation.messages

    @property
    def latest_message(self) -> Optional[MessageParam]:
        """
        get latest message

        Returns:
            Optional[dict]: 
                None if no message or the latest one
        """
        return self._conversation.latest

    @property
    def sid(self) -> SessionID:
        return self._sid;

    @property
    def theme_color(self) -> str:
        return self._theme_color

    @property
    def event_extra(self) -> dict[str, Any]:
        return {
            "theme_color": self._theme_color,
            "session_id": self.sid
        }


class SessionMiddleware(Protocol):

    def post_session_init(self, session: 'Session') -> None: ...

    def post_agent_step(self, session: 'Session', resp: Message) -> bool:
        return False


class SessionMiddlewareFactory(Protocol):

    def create(self, session: Session) -> SessionMiddleware:...


class ContexManager:

    pass
