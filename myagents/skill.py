"""Skills
"""


from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Protocol

import yaml

from .capability import Capability, CapabilityEvaluator, CapabilityRule

from .log import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class SkillMeta:
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.name:
            raise ValueError("Skill name cannot be empty.")
        if not self.description:
            raise ValueError("Skill description cannot be empty.")

        # make tags immuatable
        object.__setattr__(self, 'tags', tuple(self.tags or []))

        # make sure required_capabilities is not empty
        required_capabilities = self.required_capabilities or []
        # SKILL_USE is required
        if Capability.SKILL_USE.value not in required_capabilities:
            required_capabilities.append(Capability.SKILL_USE.value)
        # make required_capablities immuatable
        object.__setattr__(
            self, 
            'required_capabilities', 
            tuple(required_capabilities)
        )

    def __str__(self) -> str:
        return f"{self.name}: {self.description} [{self.tags}]"


@dataclass(frozen=True)
class Skill:
    meta: SkillMeta
    path: Path

    @property
    def content(self) -> str:
        try:
            text = self.path.read_text(encoding="utf-8")

            # strip front-matter（如果存在）
            if text.startswith("---"):
                parts = text.split("---", 2)
                if len(parts) >= 3:
                    return parts[2].strip()

            return text
        except Exception as e:
            raise RuntimeError(
                f"Failed to read skill file: {self.path}") from e


class SkillProvider(Protocol):
    """Skill Provider
    """

    def get_skills(self) -> Iterable[Skill]:
        """get all skills
        """
        ...


class StaticSkillsLoader:

    def __init__(self, skills_dir: Path):
        self._skills_dir: Path = skills_dir 
        self._skills: Optional[dict[str, Skill]] = None

    def get_skills(self) -> Iterable[Skill]:
        if self._skills is None:
            self._skills = self._load_skills()
        return self._skills.values()

    def _load_skills(self) -> dict[str, Skill]:
        if not self._skills_dir.exists():
            raise RuntimeError(
                f"Skill Directory {self._skills_dir} doesn't exists.")

        skills = {}
        for skill_dir in self._skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                # skip illegal dir
                continue

            meta = self._parse_meta(skill_file, skill_dir.name)
            if not meta:
                continue

            if meta.name in skills:
                raise RuntimeError(f"Duplicate skill name: {meta.name}")

            skills[meta.name] = Skill(
                meta=meta,
                path=skill_file,
            )

        return skills

    def _parse_meta(self, path: Path, default_name: str) -> Optional[SkillMeta]:
        try:
            # only read front-matter
            with path.open("r", encoding="utf-8") as f:
                first_line = f.readline()

                if not first_line.startswith("---"):
                    logger.warning(f"No front-matter of skill {default_name} is missing")
                    return None

                yaml_lines = []
                closed = False
                for line in f:
                    if line.startswith("---"):
                        closed = True
                        break
                    yaml_lines.append(line)

                if not closed:
                    raise RuntimeError(
                        f"Invalid front-matter in {path}: missing closing '---'")

            data = yaml.safe_load("".join(yaml_lines)) or {}

        except Exception as e:
            raise RuntimeError(f"Failed to parse skill meta: {path}") from e

        return SkillMeta(
            name=data.get("name") or default_name,
            description=data.get("description") or f"Skill from {default_name}",
            tags=data.get("tags") or [],
            required_capabilities=data.get("required_capabilities") or [],
        )


class ResolvedSkillsCacheValue:

    def __init__(self, skills: Iterable[Skill]):
        self._skills: dict[str, Skill] = {s.meta.name: s for s in skills}
        self._meta_by_name: dict[str, SkillMeta] = {s.meta.name: s.meta for s in skills}

    def get_metas(self) -> dict[str, SkillMeta]:
        return self._meta_by_name

    def get(self, skill_name) -> Optional[Skill]:
        return self._skills.get(skill_name)


class SkillManager:

    _providers: list[SkillProvider]
    _resolved_skills_cache: dict[frozenset[CapabilityRule], ResolvedSkillsCacheValue]

    def __init__(self, initial_providers: list[SkillProvider] = []):
        self._providers = initial_providers if initial_providers else []
        self._resolved_skills_cache = {}

    def add_providers(self, *providers: SkillProvider):
        """add providers in order

        if the provider is already in list, 
        it will be removed and append to the end
        """
        for p in providers:
            self.add_provider(p)

    def add_provider(self, provider: SkillProvider):
        """add provider

        if the provider is already in list, 
        it will be removed and append to the end
        """
        if provider in self._providers:
            self._providers.remove(provider)
        self._providers.append(provider)
        self._invalid_tools_cache()

    def remove_provider(self, provider: SkillProvider):
        """remove provider
        """
        if provider not in self._providers:
            return
        self._providers.remove(provider)
        self._invalid_tools_cache()

    def resolve_skills(self, 
                       allowed_capabilities: Iterable[CapabilityRule], 
                       extra_providers: Iterable[SkillProvider] = []) -> Iterable[SkillMeta]:
        """resolve skills by capabilities

        Args:
            capabilities (_type_): allowed capabilities

        Returns:
            Iterable[SkillMeta]: metas of skills
        """
        skills = self._get_or_resolve_skills(allowed_capabilities).get_metas()

        if extra_providers:
            extra_skills = self._find_skills(
                allowed_capabilities=allowed_capabilities,
                providers=extra_providers
            )
            if extra_skills:
                # overwrite tools by name
                skills.update({name: tool.meta for name, tool in extra_skills.items()})

        return skills.values()

    def _get_or_resolve_skills(self, allowed_capabilities: Iterable[CapabilityRule]) -> ResolvedSkillsCacheValue:
        """get or resolve skills by capabilities

        If the capabilities hint cache, return it from cache.
        Or resolve skills, update cache and return

        Args:
            capabilities (list[CapabilityRule]): allowed capabilities

        Returns:
            ResolvedToolsCacheValue: cached value
        """
        if self._resolved_skills_cache is None:
            self._resolved_skills_cache = {}

        cache_key = frozenset(allowed_capabilities)

        if cache_key not in self._resolved_skills_cache:
            tools = self._find_skills(
                allowed_capabilities=allowed_capabilities,
                providers=self._providers
            )
            cache_value = ResolvedSkillsCacheValue(tuple(tools.values()))
            self._resolved_skills_cache[cache_key] = cache_value
            logger.debug(f"resolve skills by {allowed_capabilities} -> {tools}")

        return self._resolved_skills_cache[cache_key]

    def _find_skills(self, 
                    allowed_capabilities: Iterable[CapabilityRule], 
                    providers: Iterable[SkillProvider]) -> dict[str, Skill]:
        evaluator = CapabilityEvaluator(allowed_capabilities)
        allowed_skills: dict[str, Skill] = {}

        # the older one will be skipped
        for provider in reversed(list(providers)):
            provided_skills = provider.get_skills()
            for skill in provided_skills:
                if not evaluator.is_allowed(skill.meta.required_capabilities):
                    # capabilities not matched
                    continue
                if skill.meta.name in allowed_skills:
                    # skip if name is duplicated
                    continue
                # select the first allowed one
                allowed_skills[skill.meta.name] = skill
        return allowed_skills

    def _find_first_skill(self, 
                         tool_name: str, 
                         allowed_capabilities: Iterable[CapabilityRule], 
                         providers: Iterable[SkillProvider]) -> Optional[Skill]:
        evaluator = CapabilityEvaluator(allowed_capabilities)

        # the older one will be skipped
        for provider in reversed(list(providers)):
            provided_skills = provider.get_skills()
            for skill in provided_skills:
                if tool_name != skill.meta.name:
                    continue
                if not evaluator.is_allowed(skill.meta.required_capabilities):
                    # capabilities not matched
                    continue
                # return the first caplibities matched skill
                return skill

        return None

    def _invalid_tools_cache(self):
        self._resolved_tools_cache = None

    def get_content(self, 
                    allowed_capabilities: list[CapabilityRule], 
                    skill_name: str, 
                    extra_providers: Iterable[SkillProvider] = []) -> str:
        """get content of the skill

        Args:
            name (str): name of skill

        Returns:
            str: content of skill
        """

        if not skill_name:
            raise ValueError("skill_name is requried.")

        skill = None
        # try to find the tool in extra_providers first
        if extra_providers:
            skill = self._find_first_skill(
                tool_name=skill_name,
                allowed_capabilities=allowed_capabilities,
                providers=extra_providers,
            )

        # If tool doesn't exist in extra_providers,
        # try to find the tool in holded providers
        if not skill:
            # this method is the same one used in resolve_tools,
            # which provides the same view as the resolution phase.
            cached = self._get_or_resolve_skills(allowed_capabilities)
            skill = cached.get(skill_name)

        if not skill:
            raise RuntimeError(
                f"Tool {skill_name} is not supported yet. Please check if the tool exists or capabilities are allowed")

        return skill.content