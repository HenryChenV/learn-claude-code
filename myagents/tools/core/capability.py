"""Capablity
"""


from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class CapabilityRule:
    """ Capabilty Rule

    supports:
        1. full match: "file.read"
        2. prefix match: "file.*"
        3. not match: "!file.delete", "!file.*"

    Attributes:
        raw(str): raw input, 
                    e.g. "file.read", "file.*"", "!file.delete"

        pattern(str): pattern to match, 
                    e.g. "file.read", "file.", "file.delete"
        is_prefix(bool): if the pattern is a prefix or not
        allowed(bool): if allowed if pattern is matched
    """

    raw: str
    pattern: str
    is_prefix: bool
    allowed: bool

    @classmethod
    def wrap(cls, rule: Union[str, 'CapabilityRule']):
        if isinstance(rule, CapabilityRule):
            return rule
        return cls.parse(rule=rule)

    @classmethod
    def parse(cls, rule: str) -> 'CapabilityRule':
        if not rule:
            raise ValueError("rule is required.")

        if rule.startswith("!"):
            allowed = False
            pattern = rule[1:]
        else:
            allowed = True
            pattern = rule

        if pattern.endswith(".*"):
            # remove "*" from the ends
            pattern = pattern[:-1]
            is_prefix = True
        else:
            is_prefix = False

        # wildcards are only allowd at the end
        if "*" in pattern:
            raise ValueError("'*' is only allowed at the end.")

        return CapabilityRule(
            raw=rule, 
            pattern=pattern, 
            is_prefix=is_prefix, 
            allowed=allowed
        )

    def evaluate(self, capability: str) -> bool | None:
        """evaluate if the given capability is allowed

        Args:
            capability (str): capability

        Returns:
            bool: 
                True represents allowed, 
                False represents denied
                None represents not matched
        """
        matched = self._match(capability)
        if matched:
            return self.allowed
        return None

    def _match(self, capabiliy: str) -> Optional[bool]:
        """ check if the given capability is match the pattern
        """
        if self.is_prefix:
            return capabiliy.startswith(self.pattern)
        return capabiliy == self.pattern

    def __str__(self):
        return self.raw


class CapabilityEvaluator:
    """Capablity Evaluator to evaluate if the 

    Returns:
        _type_: _description_
    """

    _rules: frozenset[CapabilityRule]

    def __init__(self, rules: list[CapabilityRule]):
        self._rules = frozenset(rules)

    def is_allowed(self, required_capabilities: list[str]) -> bool:
        """if all required capabilities are allowed

        Args:
            required_capabilities (list[str]): requried capabilites

        Returns:
            bool: allowed or not
        """
        return all(self._is_capability_allowed(cap) for cap in required_capabilities)

    def _is_capability_allowed(self, cap) -> bool:
        allowed = None
        for rule in self._rules:
            result = rule.evaluate(cap)
            if result is None:
                # check the next if not matched
                continue
            if result is False:
                # the final result is False if any rule said no
                allowed = False
                break
            # although the rule said yes, 
            # remained rules still need to evaluate to avoid any dismatched
            allowed = True

        return allowed is True
