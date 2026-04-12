"""LLM Models
"""


from abc import ABC, abstractmethod
import copy
import os
from anthropic import Anthropic, Omit, omit
from anthropic.types import Message, MessageParam, TextBlockParam, ToolUnionParam
from typing import Any, Iterable, Union

from dataclasses import dataclass

from myagents.tools.core.tool import ToolMeta


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    context_window: int

    @property
    def full_id(self):
        return f"{self.provider}/{self.model}"


class ChatModel:

    def __init__(self, 
                 spec: ModelSpec,
                 client: Anthropic):
        self._spec: ModelSpec = spec
        self._client = client

    @property
    def provider(self):
        return self._spec.provider

    @property
    def model(self):
        return self._spec.model

    @property
    def context_window(self) -> int:
        return self._spec.context_window

    def chat(self, 
             max_tokens: int,
             messages: Iterable[MessageParam], 
             system_prompt:Union[str, Iterable[TextBlockParam]] | Omit = omit,
             tools: Iterable[ToolMeta] = []) -> Message:
        return self._client.messages.create(
            max_tokens=max_tokens,
            model=self.model, 
            system=system_prompt,
            messages=messages, 
            tools=[self._build_tool_desc(t) for t in tools], 
        )

    def _build_tool_desc(self, meta: ToolMeta) -> ToolUnionParam:
        return {
            "name": meta.name,
            "description": meta.description,
            "input_schema": meta.input_schema,
        }


class ChatModelProvider(ABC):

    def __init__(self, 
                 name, 
                 model_specs: list[ModelSpec], 
                 api_key, 
                 base_url=None):
        self._name: str = name
        self._client: Anthropic = Anthropic(base_url=base_url, api_key=api_key)
        self._models: dict[str, ChatModel] = {ms.model: ChatModel(ms, self._client) for ms in model_specs}

    def get_model(self, model_id: str) -> ChatModel: 
        """get mdoel by model_id
        """
        if model_id not in self._models:
            raise ValueError(f"Provider {self._name} doesn't have the model {model_id}")
        return self._models[model_id]


MODEL_LIST = {
    "MiniMax": {
        "base_url": "https://api.minimaxi.com/anthropic",
        "api_key_env_var": "MINIMAX_API_KEY",
        "models": {
            "MiniMax-M2.7": {"context_window": 204800},
            "MiniMax-M2.5": {"context_window": 204800},
        }
    }
}


class ChatModelManager:

    _model_list: dict[str, dict[str, Any]]
    _provider_cache: dict[str, ChatModelProvider]

    def __init__(self, 
                 model_list: dict[str, dict[str, Any]]): 
        self._model_list = copy.deepcopy(model_list)
        self._provider_cache = {}

    def get_model(self, model_full_id: str) -> ChatModel:
        """get model by full id
        Args:
            model_full_id (str): in format "{provider}/{model_id}" 
                                    e.g. MiniMax/MiniMax-M2.7
        """
        if not model_full_id:
            raise ValueError("model_full_id is requried.")

        provider, model_id = model_full_id.split("/", 2)

        if provider not in self._model_list:
            raise ValueError(
                f"Provider {provider} is not supported." 
                f" Available providers are {self._model_list.keys}"
            )

        provider_conf = self._model_list[provider]
        if model_id not in provider_conf["models"]:
            raise ValueError(
                f"Model {model_id} is not supported by Provider {provider}"
                f"Available models are {provider_conf['models']}"
            )

        if provider not in self._provider_cache:
            env_var = provider_conf.get("api_key_env_var")
            if not env_var or not isinstance(env_var, str):
                raise ValueError(
                    f"The envrionment variable name of API Key "
                    f"for Provider {provider} is not configured."
                )
            api_key = os.environ.get(env_var)
            if not api_key:
                raise ValueError(
                    f"The envrionment variable of API Key '{env_var}' "
                    f"for Provider {provider} is not configured."
                )
            self._provider_cache[provider] = ChatModelProvider(
                name=provider,
                model_specs=[ModelSpec(provider, m, s["context_window"]) 
                             for m, s in provider_conf["models"].items()],
                api_key=api_key,
                base_url=provider_conf.get("base_url"),
            )
        
        return self._provider_cache[provider].get_model(model_id=model_id) 

    @classmethod
    def get_default(cls) -> 'ChatModelManager':
        return DEFAULT_MODEL_MANAGER


DEFAULT_MODEL_MANAGER = ChatModelManager(MODEL_LIST)