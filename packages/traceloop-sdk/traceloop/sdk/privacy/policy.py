"""Validation and normalization of redaction policies."""

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

_MAX_RULES = 64
_MIN_DEPTH = 1
_MAX_DEPTH = 16
_MIN_VALUE_BYTES = 1024
_MAX_VALUE_BYTES = 262144
_MAX_KEEP_LAST = 8
_DEFAULT_MASK_TOKEN = "[REDACTED:{kind}]"
_MIN_SALT_LENGTH = 8

DETECTORS = frozenset(
    {
        "email",
        "phone_e164",
        "credit_card",
        "ipv4",
        "bearer",
        "api_key",
        "secret_assignment",
    }
)

_KEEP_LAST_DETECTORS = frozenset({"credit_card", "phone_e164"})


class RedactionAction(str, Enum):
    """What to do with a detector hit."""

    MASK = "mask"
    HASH = "hash"
    DROP = "drop"
    KEEP = "keep"


def _action_from_value(value: object, *, code_on_error: str = "unknown_action") -> RedactionAction:
    if isinstance(value, RedactionAction):
        return value
    if isinstance(value, str):
        try:
            return RedactionAction(value)
        except ValueError:
            raise RedactionPolicyError(
                f"unknown redaction action: {value!r}", code=code_on_error
            ) from None
    raise RedactionPolicyError(
        f"unknown redaction action: {value!r}", code=code_on_error
    )


class RedactionPolicyError(ValueError):
    """Raised when a redaction policy payload is invalid."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RedactionRule:
    """A single attribute rule."""

    name: str
    attribute: str
    detector: str
    action: RedactionAction
    keep_last: int = 0
    salt: str = ""

    def matches(self, attribute: str) -> bool:
        """Return whether ``attribute`` matches this rule's pattern."""
        if self.attribute.endswith("*"):
            return attribute.startswith(self.attribute[:-1])
        return attribute == self.attribute


def _validate_glob(attribute: object) -> str:
    if not isinstance(attribute, str) or not attribute:
        raise RedactionPolicyError(
            f"invalid attribute glob: {attribute!r}", code="bad_glob"
        )
    if attribute.count("*") > 1:
        raise RedactionPolicyError(
            f"attribute glob may contain a single trailing '*': {attribute!r}",
            code="bad_glob",
        )
    if "*" in attribute:
        if not attribute.endswith("*"):
            raise RedactionPolicyError(
                f"only a single trailing '*' is allowed: {attribute!r}",
                code="bad_glob",
            )
        prefix = attribute[:-1]
        if not prefix or "*" in prefix:
            raise RedactionPolicyError(
                f"attribute glob must end with '*' after a literal prefix: {attribute!r}",
                code="bad_glob",
            )
    return attribute


def _parse_rule(payload: Mapping[str, object], index: int) -> RedactionRule:
    raw_attribute = payload.get("attribute")
    attribute = _validate_glob(raw_attribute)

    detector = payload.get("detector")
    if not isinstance(detector, str) or detector not in DETECTORS:
        raise RedactionPolicyError(
            f"unknown detector: {detector!r}", code="unknown_detector"
        )

    action = _action_from_value(payload.get("action"))

    raw_keep_last = payload.get("keep_last", 0)
    if isinstance(raw_keep_last, bool) or not isinstance(raw_keep_last, int):
        raise RedactionPolicyError(
            f"keep_last must be an integer: {raw_keep_last!r}",
            code="bad_keep_last",
        )
    keep_last = raw_keep_last
    if keep_last < 0 or keep_last > _MAX_KEEP_LAST:
        raise RedactionPolicyError(
            f"keep_last must be between 0 and {_MAX_KEEP_LAST}: {keep_last!r}",
            code="bad_keep_last",
        )
    if keep_last and detector not in _KEEP_LAST_DETECTORS:
        raise RedactionPolicyError(
            f"keep_last is only supported for credit_card and phone_e164, not {detector!r}",
            code="bad_keep_last",
        )

    raw_salt = payload.get("salt", "")
    if raw_salt is None:
        raw_salt = ""
    if not isinstance(raw_salt, str):
        raise RedactionPolicyError(
            f"salt must be a string: {raw_salt!r}", code="weak_salt"
        )
    salt = raw_salt
    if action is RedactionAction.HASH and len(salt) < _MIN_SALT_LENGTH:
        raise RedactionPolicyError(
            f"hash rules require a salt of at least {_MIN_SALT_LENGTH} characters",
            code="weak_salt",
        )

    raw_name = payload.get("name")
    name = raw_name if isinstance(raw_name, str) and raw_name else f"rule_{index}"

    return RedactionRule(
        name=name,
        attribute=attribute,
        detector=detector,
        action=action,
        keep_last=keep_last,
        salt=salt,
    )


@dataclass(frozen=True)
class RedactionPolicy:
    """A validated, immutable redaction policy."""

    version: int = 1
    rules: tuple[RedactionRule, ...] = ()
    default_action: RedactionAction = RedactionAction.KEEP
    mask_token: str = _DEFAULT_MASK_TOKEN
    max_depth: int = 8
    max_value_bytes: int = 65536
    fail_closed: bool = True
    fingerprint: str = field(default="")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RedactionPolicy":
        """Validate ``payload`` and return a normalized policy."""
        if not isinstance(payload, Mapping):
            raise RedactionPolicyError(
                "redaction policy must be a mapping", code="bad_version"
            )

        version = payload.get("version", 1)
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise RedactionPolicyError(
                f"version must be a positive integer: {version!r}",
                code="bad_version",
            )

        raw_rules = payload.get("rules", ())
        if raw_rules is None:
            raw_rules = ()
        if not isinstance(raw_rules, Sequence) or isinstance(
            raw_rules, (str, bytes, bytearray)
        ):
            raise RedactionPolicyError(
                "rules must be a list of rule mappings", code="bad_version"
            )
        if len(raw_rules) > _MAX_RULES:
            raise RedactionPolicyError(
                f"at most {_MAX_RULES} rules are allowed", code="too_many_rules"
            )

        rules = []
        for index, raw_rule in enumerate(raw_rules):
            if not isinstance(raw_rule, Mapping):
                raise RedactionPolicyError(
                    f"rule at index {index} must be a mapping", code="bad_glob"
                )
            rules.append(_parse_rule(raw_rule, index))

        default_action = _action_from_value(
            payload.get("default_action", RedactionAction.KEEP.value)
        )

        raw_mask_token = payload.get("mask_token", _DEFAULT_MASK_TOKEN)
        if not isinstance(raw_mask_token, str) or "{kind}" not in raw_mask_token:
            raise RedactionPolicyError(
                "mask_token must be a string containing the '{kind}' placeholder",
                code="bad_mask_token",
            )
        mask_token = raw_mask_token

        raw_depth = payload.get("max_depth", 8)
        if (
            isinstance(raw_depth, bool)
            or not isinstance(raw_depth, int)
            or raw_depth < _MIN_DEPTH
            or raw_depth > _MAX_DEPTH
        ):
            raise RedactionPolicyError(
                f"max_depth must be between {_MIN_DEPTH} and {_MAX_DEPTH}",
                code="bad_depth",
            )
        max_depth = raw_depth

        raw_value_bytes = payload.get("max_value_bytes", 65536)
        if (
            isinstance(raw_value_bytes, bool)
            or not isinstance(raw_value_bytes, int)
            or raw_value_bytes < _MIN_VALUE_BYTES
            or raw_value_bytes > _MAX_VALUE_BYTES
        ):
            raise RedactionPolicyError(
                f"max_value_bytes must be between {_MIN_VALUE_BYTES} and {_MAX_VALUE_BYTES}",
                code="bad_value_bytes",
            )
        max_value_bytes = raw_value_bytes

        raw_fail_closed = payload.get("fail_closed", True)
        if not isinstance(raw_fail_closed, bool):
            raise RedactionPolicyError(
                "fail_closed must be a boolean", code="bad_version"
            )
        fail_closed = raw_fail_closed

        normalized: dict[str, Any] = {
            "version": version,
            "rules": tuple(rules),
            "default_action": default_action,
            "mask_token": mask_token,
            "max_depth": max_depth,
            "max_value_bytes": max_value_bytes,
            "fail_closed": fail_closed,
        }
        fingerprint = _fingerprint(normalized)

        return cls(
            version=version,
            rules=tuple(rules),
            default_action=default_action,
            mask_token=mask_token,
            max_depth=max_depth,
            max_value_bytes=max_value_bytes,
            fail_closed=fail_closed,
            fingerprint=fingerprint,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the policy."""
        return {
            "version": self.version,
            "rules": [
                {
                    "name": rule.name,
                    "attribute": rule.attribute,
                    "detector": rule.detector,
                    "action": rule.action.value,
                    "keep_last": rule.keep_last,
                    "salt": rule.salt,
                }
                for rule in self.rules
            ],
            "default_action": self.default_action.value,
            "mask_token": self.mask_token,
            "max_depth": self.max_depth,
            "max_value_bytes": self.max_value_bytes,
            "fail_closed": self.fail_closed,
            "fingerprint": self.fingerprint,
        }

    def rule_for(self, attribute: str) -> RedactionRule | None:
        """Return the first rule matching ``attribute``, if any."""
        for rule in self.rules:
            if rule.matches(attribute):
                return rule
        return None


def _fingerprint(normalized: Mapping[str, Any]) -> str:
    payload = {
        "version": normalized["version"],
        "rules": [
            {
                "name": rule.name,
                "attribute": rule.attribute,
                "detector": rule.detector,
                "action": rule.action.value,
                "keep_last": rule.keep_last,
                "salt": rule.salt,
            }
            for rule in normalized["rules"]
        ],
        "default_action": normalized["default_action"].value,
        "mask_token": normalized["mask_token"],
        "max_depth": normalized["max_depth"],
        "max_value_bytes": normalized["max_value_bytes"],
        "fail_closed": normalized["fail_closed"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
