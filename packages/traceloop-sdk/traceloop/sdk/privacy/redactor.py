"""Detectors, redaction and the runtime that performs it.

The redaction layer is deliberately boring: regular expressions are compiled
once, every attribute value is scanned at most once, and a failure while
handling one attribute never escapes to the application.
"""

import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from opentelemetry import context as context_api

from traceloop.sdk.privacy import attributes as privacy_attributes
from traceloop.sdk.privacy.policy import (
    RedactionAction,
    RedactionPolicy,
    RedactionPolicyError,
    RedactionRule,
)
from traceloop.sdk.privacy.scope import (
    AssociationScope,
    AssociationSnapshot,
    get_association_registry,
)

_REDACTION_POLICY_ENV_VAR = "TRACELOOP_REDACTION_POLICY"
_SALT_SEPARATOR = ":"
_HASH_DIGEST_LENGTH = 12

# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


@dataclass
class _Candidate:
    """A detector hit inside a single string."""

    start: int
    end: int
    kind: str
    canonical: str
    digits: str | None = None


_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+\-])"
    r"[A-Za-z0-9._%+\-]{1,64}"
    r"@[A-Za-z0-9\-]{1,63}"
    r"(?:\.[A-Za-z0-9\-]{1,63}){0,8}"
    r"\.[A-Za-z]{2,24}"
    r"(?![A-Za-z0-9\-])"
)
_PHONE_RE = re.compile(r"(?<![\w+])\+\d[\d \-]{6,22}\d(?![\w])")
_CREDIT_CARD_RE = re.compile(r"(?<!\d)\d[\d \-]{11,23}\d(?!\d)")
_IPV4_RE = re.compile(r"(?<![\d.])\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?![\d.])")
_BEARER_RE = re.compile(r"(?i)\bBearer[ \t]+([A-Za-z0-9._~+/=\-]{1,}=*)")
_API_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:sk-[A-Za-z0-9_\-]{8,}|AKIA[0-9A-Z]{8,}|ghp_[A-Za-z0-9]{8,})(?![A-Za-z0-9])"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])"
    r"(password|passwd|secret|token|api_key|authorization)"
    r"(?![A-Za-z0-9_])"
    r"(\s*[:=]\s*)"
    r"(\"[^\"\n]*\"|'[^'\n]*'|(?:Bearer|Basic|Token|Digest)\s+[^\s,;&\"']+|[^\s,;&\"']+)"
)


def _digit_count(value: str) -> int:
    return sum(1 for char in value if char.isdigit())


def _luhn_valid(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        number = ord(char) - ord("0")
        if index % 2 == parity:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return total % 10 == 0


def _detect_email(text: str) -> list[_Candidate]:
    return [
        _Candidate(match.start(), match.end(), "email", match.group(0))
        for match in _EMAIL_RE.finditer(text)
    ]


def _detect_phone(text: str) -> list[_Candidate]:
    candidates = []
    for match in _PHONE_RE.finditer(text):
        digits = "".join(char for char in match.group(0) if char.isdigit())
        if 8 <= len(digits) <= 15:
            candidates.append(
                _Candidate(
                    match.start(),
                    match.end(),
                    "phone_e164",
                    match.group(0),
                    digits,
                )
            )
    return candidates


def _detect_credit_card(text: str) -> list[_Candidate]:
    candidates = []
    for match in _CREDIT_CARD_RE.finditer(text):
        digits = "".join(char for char in match.group(0) if char.isdigit())
        if not 13 <= len(digits) <= 19:
            continue
        if not _luhn_valid(digits):
            continue
        candidates.append(
            _Candidate(
                match.start(),
                match.end(),
                "credit_card",
                match.group(0),
                digits,
            )
        )
    return candidates


def _detect_ipv4(text: str) -> list[_Candidate]:
    candidates = []
    for match in _IPV4_RE.finditer(text):
        octets = match.group(0).split(".")
        if all(0 <= int(octet) <= 255 for octet in octets):
            candidates.append(
                _Candidate(match.start(), match.end(), "ipv4", match.group(0))
            )
    return candidates


def _detect_bearer(text: str) -> list[_Candidate]:
    candidates = []
    for match in _BEARER_RE.finditer(text):
        start, end = match.span(1)
        candidates.append(_Candidate(start, end, "bearer", match.group(1)))
    return candidates


def _detect_api_key(text: str) -> list[_Candidate]:
    return [
        _Candidate(match.start(), match.end(), "api_key", match.group(0))
        for match in _API_KEY_RE.finditer(text)
    ]


def _detect_secret_assignment(text: str) -> list[_Candidate]:
    candidates = []
    for match in _SECRET_ASSIGNMENT_RE.finditer(text):
        start, end = match.span(3)
        value = match.group(3)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            start += 1
            end -= 1
            value = value[1:-1]
        candidates.append(
            _Candidate(start, end, "secret_assignment", value)
        )
    return candidates


Detector = Callable[[str], list[_Candidate]]

DETECTORS: dict[str, Detector] = {
    "email": _detect_email,
    "phone_e164": _detect_phone,
    "credit_card": _detect_credit_card,
    "ipv4": _detect_ipv4,
    "bearer": _detect_bearer,
    "api_key": _detect_api_key,
    "secret_assignment": _detect_secret_assignment,
}

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class RedactionStats:
    """Counters describing what a redaction pass did."""

    scans: int = 0
    values_masked: int = 0
    values_hashed: int = 0
    attributes_dropped: int = 0
    payloads_parsed: int = 0
    payloads_unparsed: int = 0
    truncated_values: int = 0
    redaction_errors: int = 0
    content_dropped: int = 0
    unsupported_values: int = 0

    def copy(self) -> "RedactionStats":
        return RedactionStats(
            scans=self.scans,
            values_masked=self.values_masked,
            values_hashed=self.values_hashed,
            attributes_dropped=self.attributes_dropped,
            payloads_parsed=self.payloads_parsed,
            payloads_unparsed=self.payloads_unparsed,
            truncated_values=self.truncated_values,
            redaction_errors=self.redaction_errors,
            content_dropped=self.content_dropped,
            unsupported_values=self.unsupported_values,
        )

    def add(self, other: "RedactionStats") -> None:
        self.scans += other.scans
        self.values_masked += other.values_masked
        self.values_hashed += other.values_hashed
        self.attributes_dropped += other.attributes_dropped
        self.payloads_parsed += other.payloads_parsed
        self.payloads_unparsed += other.payloads_unparsed
        self.truncated_values += other.truncated_values
        self.redaction_errors += other.redaction_errors
        self.content_dropped += other.content_dropped
        self.unsupported_values += other.unsupported_values


@dataclass(frozen=True)
class PrivacySnapshot:
    """A point-in-time view of the privacy layer."""

    policy_fingerprint: str | None
    redaction: RedactionStats
    association: AssociationSnapshot
    policy_reload_failures: int
    unreadable_spans: int


_SUPPORTED_SCALARS = (int, float, bool, bytes)


# ---------------------------------------------------------------------------
# Redactor
# ---------------------------------------------------------------------------


class Redactor:
    """Applies a :class:`RedactionPolicy` to span attributes."""

    def __init__(self, policy: RedactionPolicy | None = None) -> None:
        self._policy = policy

    def redact_attributes(
        self,
        attributes: Mapping[str, object],
        scope: AssociationScope,
        *,
        content_allowed: bool,
    ) -> tuple[dict[str, object], RedactionStats]:
        """Return a redacted copy of ``attributes`` and its stats.

        The input mapping is never mutated.
        """
        policy = self._policy
        if policy is None:
            return dict(attributes), RedactionStats()

        stats = RedactionStats()
        result: dict[str, object] = {}

        for key, value in attributes.items():
            if key == privacy_attributes.TRACELOOP_PRIVACY_SCAN_STATE:
                result[key] = value
                continue

            if not content_allowed and privacy_attributes.is_content_attribute(key):
                stats.content_dropped += 1
                continue

            try:
                dropped, new_value = self._process_attribute(
                    key, value, policy, stats
                )
            except Exception:
                # A single bad attribute must never fail the whole span.
                stats.redaction_errors += 1
                if policy.fail_closed:
                    continue
                result[key] = value
                continue

            if dropped:
                stats.attributes_dropped += 1
                continue
            result[key] = new_value

        return result, stats

    # ------------------------------------------------------------------
    # Attribute processing
    # ------------------------------------------------------------------
    def _process_attribute(
        self,
        key: str,
        value: object,
        policy: RedactionPolicy,
        stats: RedactionStats,
    ) -> tuple[bool, object]:
        rule = policy.rule_for(key)
        action = rule.action if rule is not None else policy.default_action

        if action is RedactionAction.KEEP:
            return False, value

        if action is RedactionAction.DROP:
            # Dropping is all-or-nothing: the whole attribute goes away.
            return True, None

        if isinstance(value, str):
            return self._process_string(key, value, rule, action, policy, stats)

        if isinstance(value, (Mapping, list, tuple)):
            if privacy_attributes.is_json_content_attribute(key):
                return self._walk(value, rule, action, policy, 0, set(), stats)
            # A non-string container attribute cannot be scanned safely; leave
            # it untouched and record that we could not handle it.
            stats.unsupported_values += 1
            return False, value

        if value is None or not isinstance(value, _SUPPORTED_SCALARS):
            stats.unsupported_values += 1
        return False, value

    def _process_string(
        self,
        key: str,
        value: str,
        rule: RedactionRule | None,
        action: RedactionAction,
        policy: RedactionPolicy,
        stats: RedactionStats,
    ) -> tuple[bool, object]:
        if privacy_attributes.has_redaction_marker(value):
            # Already redacted values are never scanned again.
            return False, value

        if privacy_attributes.is_json_content_attribute(key):
            return self._process_json(value, rule, action, policy, stats)

        return self._scan_string(value, rule, action, policy, stats)

    def _process_json(
        self,
        value: str,
        rule: RedactionRule | None,
        action: RedactionAction,
        policy: RedactionPolicy,
        stats: RedactionStats,
    ) -> tuple[bool, object]:
        if len(value.encode("utf-8")) > policy.max_value_bytes:
            # Only the bounded prefix is eligible for scanning.
            stats.payloads_unparsed += 1
            return self._scan_string(value, rule, action, policy, stats)

        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            stats.payloads_unparsed += 1
            return self._scan_string(value, rule, action, policy, stats)

        stats.payloads_parsed += 1
        dropped, redacted = self._walk(
            parsed, rule, action, policy, 0, set(), stats
        )
        if dropped:
            return True, None
        try:
            return False, json.dumps(redacted, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            raise

    def _walk(
        self,
        node: object,
        rule: RedactionRule | None,
        action: RedactionAction,
        policy: RedactionPolicy,
        depth: int,
        seen: set[int],
        stats: RedactionStats,
    ) -> tuple[bool, object]:
        if depth > policy.max_depth:
            stats.truncated_values += 1
            return False, privacy_attributes.TRUNCATED_MARKER

        if isinstance(node, str):
            return self._scan_string(node, rule, action, policy, stats)

        if isinstance(node, (Mapping, list, tuple)):
            node_id = id(node)
            if node_id in seen:
                stats.truncated_values += 1
                return False, privacy_attributes.TRUNCATED_MARKER
            seen = seen | {node_id}

            if isinstance(node, Mapping):
                output: dict[object, object] = {}
                for child_key, child in node.items():
                    dropped, new_child = self._walk(
                        child, rule, action, policy, depth + 1, seen, stats
                    )
                    if dropped:
                        return True, None
                    output[child_key] = new_child
                return False, output

            children = []
            for child in node:
                dropped, new_child = self._walk(
                    child, rule, action, policy, depth + 1, seen, stats
                )
                if dropped:
                    return True, None
                children.append(new_child)
            if isinstance(node, tuple):
                return False, tuple(children)
            return False, children
        return False, node

    def _scan_string(
        self,
        text: str,
        rule: RedactionRule | None,
        action: RedactionAction,
        policy: RedactionPolicy,
        stats: RedactionStats,
    ) -> tuple[bool, object]:
        if privacy_attributes.has_redaction_marker(text):
            return False, text

        suffix = ""
        encoded = text.encode("utf-8")
        if len(encoded) > policy.max_value_bytes:
            head = encoded[: policy.max_value_bytes].decode("utf-8", errors="ignore")
            suffix = text[len(head):]
            text = head
            stats.truncated_values += 1

        # Each value is scanned at most once.
        stats.scans += 1
        candidates = self._collect(text, rule)
        if not candidates:
            return False, text + suffix

        selected = _select_candidates(candidates, len(text))
        if not selected:
            return False, text + suffix

        pieces = []
        cursor = 0
        for candidate in selected:
            pieces.append(text[cursor:candidate.start])
            if action is RedactionAction.HASH:
                pieces.append(self._render_hash(candidate, rule))
                stats.values_hashed += 1
            else:
                pieces.append(self._render_mask(candidate, rule, policy))
                stats.values_masked += 1
            cursor = candidate.end
        pieces.append(text[cursor:])
        pieces.append(suffix)
        return False, "".join(pieces)

    @staticmethod
    def _collect(
        text: str, rule: RedactionRule | None
    ) -> list[_Candidate]:
        if rule is not None:
            detector = DETECTORS.get(rule.detector)
            if detector is None:
                raise RedactionPolicyError(
                    f"unknown detector: {rule.detector!r}", code="unknown_detector"
                )
            return detector(text)

        candidates: list[_Candidate] = []
        for detector in DETECTORS.values():
            candidates.extend(detector(text))
        return candidates

    @staticmethod
    def _render_mask(
        candidate: _Candidate, rule: RedactionRule | None, policy: RedactionPolicy
    ) -> str:
        kind = candidate.kind
        if rule is not None and rule.keep_last > 0 and candidate.digits:
            last = candidate.digits[-rule.keep_last:]
            kind = f"{kind}:{last}"
        return policy.mask_token.format(kind=kind)

    @staticmethod
    def _render_hash(candidate: _Candidate, rule: RedactionRule | None) -> str:
        salt = rule.salt if rule is not None else ""
        digest = hashlib.sha256(
            f"{salt}{_SALT_SEPARATOR}{candidate.canonical}".encode("utf-8")
        ).hexdigest()[:_HASH_DIGEST_LENGTH]
        return f"{privacy_attributes.HASHED_PREFIX}{candidate.kind}:{digest}]"


def _select_candidates(
    candidates: Sequence[_Candidate], text_length: int
) -> list[_Candidate]:
    """Pick a left-to-right, longest-match-first, non-overlapping set."""
    ordered = sorted(candidates, key=lambda c: (c.start, -(c.end - c.start)))
    selected: list[_Candidate] = []
    cursor = 0
    for candidate in ordered:
        if candidate.start < cursor or candidate.end > text_length:
            continue
        selected.append(candidate)
        cursor = candidate.end
    return selected


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


def _content_allowed(scope: AssociationScope) -> bool:
    # Preserve the existing content-tracing semantics first: when the env var
    # (or the per-association override) allows content, nothing is dropped.
    env_allowed = (
        os.getenv(privacy_attributes.TRACE_CONTENT_ENV_VAR, "true") or "true"
    ).lower() == "true"
    if env_allowed:
        return True
    override = context_api.get_value(privacy_attributes.CONTENT_OVERRIDE_CONTEXT_KEY)
    if override:
        return True
    # Content is globally disabled; the allow list can still re-enable it for
    # a matching association scope.
    return scope.content_decision == privacy_attributes.CONTENT_DECISION_ALLOW


class RedactionRuntime:
    """Holds the active policy, cumulative stats and the scan state machine."""

    def __init__(
        self,
        policy: RedactionPolicy | Mapping[str, object] | None = None,
    ) -> None:
        if isinstance(policy, Mapping):
            policy = RedactionPolicy.from_dict(policy)
        self._lock = threading.Lock()
        self._policy = policy
        self._redactor = Redactor(policy)
        self._stats = RedactionStats()
        self._reload_failures = 0
        self._unreadable_spans = 0

    @property
    def policy(self) -> RedactionPolicy | None:
        return self._policy

    def set_policy(
        self, policy: RedactionPolicy | Mapping[str, object] | None
    ) -> None:
        if isinstance(policy, Mapping):
            policy = RedactionPolicy.from_dict(policy)
        with self._lock:
            self._policy = policy
            self._redactor = Redactor(policy)

    def reload_policy(self, payload: Mapping[str, object]) -> bool:
        """Atomically replace the policy; keep the old one on failure."""
        try:
            new_policy = RedactionPolicy.from_dict(payload)
        except RedactionPolicyError:
            with self._lock:
                self._reload_failures += 1
            return False
        except Exception:
            with self._lock:
                self._reload_failures += 1
            return False

        with self._lock:
            self._policy = new_policy
            self._redactor = Redactor(new_policy)
        return True

    def record_reload_failure(self) -> None:
        with self._lock:
            self._reload_failures += 1

    def merge_reload_failures(self, count: int) -> None:
        if count <= 0:
            return
        with self._lock:
            self._reload_failures += count

    def record_unreadable_span(self) -> None:
        with self._lock:
            self._unreadable_spans += 1

    def on_start(self, span: object) -> None:
        """Mark a freshly started span as armed for scanning."""
        if self._policy is None:
            return
        try:
            span.set_attribute(  # type: ignore[attr-defined]
                privacy_attributes.TRACELOOP_PRIVACY_SCAN_STATE,
                privacy_attributes.SCAN_STATE_ARMED,
            )
        except Exception:
            pass

    def is_active(self) -> bool:
        """Return whether a policy is currently installed."""
        return self._policy is not None

    def redact_span(self, span: object) -> RedactionStats:
        """Scan and rewrite ``span``'s attributes in place."""
        # Capture the policy/redactor pair once: a reload mid-scan must not
        # affect the span currently being exported.
        policy = self._policy
        redactor = self._redactor
        if policy is None:
            return RedactionStats()

        try:
            attributes = getattr(span, "_attributes", None)
        except Exception:
            attributes = None
        if attributes is None or not hasattr(attributes, "get"):
            self.record_unreadable_span()
            return RedactionStats()

        try:
            state = attributes.get(
                privacy_attributes.TRACELOOP_PRIVACY_SCAN_STATE
            )
        except Exception:
            self.record_unreadable_span()
            return RedactionStats()

        was_immutable = getattr(attributes, "_immutable", False)
        mutable = self._make_mutable(attributes, was_immutable)

        stats = RedactionStats()
        try:
            if not mutable:
                self.record_unreadable_span()
                return stats

            if state in (
                privacy_attributes.SCAN_STATE_SCANNED,
                privacy_attributes.SCAN_STATE_SKIPPED,
                privacy_attributes.SCAN_STATE_FAILED,
            ):
                self._write_state(attributes, privacy_attributes.SCAN_STATE_SKIPPED)
                return stats

            scope = get_association_registry().current()
            content_allowed = _content_allowed(scope)

            try:
                snapshot = dict(attributes.items())
            except Exception:
                self.record_unreadable_span()
                return stats

            new_attributes, stats = redactor.redact_attributes(
                snapshot, scope, content_allowed=content_allowed
            )
            self._write_attributes(attributes, new_attributes)
            self._write_state(attributes, privacy_attributes.SCAN_STATE_SCANNED)
        except Exception:
            self._write_state(attributes, privacy_attributes.SCAN_STATE_FAILED)
        finally:
            if was_immutable:
                try:
                    attributes._immutable = True
                except Exception:
                    try:
                        attributes._immutable = False
                        self._write_state(
                            attributes, privacy_attributes.SCAN_STATE_FAILED
                        )
                    except Exception:
                        pass

        with self._lock:
            self._stats.add(stats)
        return stats

    def snapshot(self) -> PrivacySnapshot:
        with self._lock:
            stats = self._stats.copy()
            failures = self._reload_failures
            unreadable = self._unreadable_spans
        return PrivacySnapshot(
            policy_fingerprint=(
                self._policy.fingerprint if self._policy is not None else None
            ),
            redaction=stats,
            association=get_association_registry().snapshot(),
            policy_reload_failures=failures,
            unreadable_spans=unreadable,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _make_mutable(attributes: object, was_immutable: bool) -> bool:
        if not was_immutable:
            return True
        try:
            setattr(attributes, "_immutable", False)
        except Exception:
            return False
        return True

    @staticmethod
    def _write_state(attributes: object, state: str) -> None:
        try:
            mapping: Any = attributes
            mapping[privacy_attributes.TRACELOOP_PRIVACY_SCAN_STATE] = state
        except Exception:
            pass

    @staticmethod
    def _write_attributes(
        attributes: object, new_attributes: Mapping[str, object]
    ) -> None:
        if not isinstance(attributes, MutableMapping):
            raise TypeError("span attributes are not mutable")
        mutable: MutableMapping[str, object] = attributes
        try:
            keys = list(mutable.keys())
        except Exception as exc:
            raise TypeError("span attributes are not readable") from exc
        for key in keys:
            if key not in new_attributes:
                del mutable[key]
        for key, value in new_attributes.items():
            mutable[key] = value


# ---------------------------------------------------------------------------
# Module-level runtime singleton
# ---------------------------------------------------------------------------

_runtime: RedactionRuntime | None = None
_runtime_lock = threading.Lock()
_fallback_failures = 0


def get_privacy_runtime() -> RedactionRuntime | None:
    """Return the installed runtime, if any."""
    return _runtime


def configure_privacy_runtime(
    policy: RedactionPolicy | Mapping[str, object] | None,
) -> RedactionRuntime:
    """Create or update the process-wide runtime and return it."""
    global _runtime, _fallback_failures
    # The registry is a process-wide singleton as well; make sure it exists
    # by the time a runtime is installed.
    get_association_registry()
    with _runtime_lock:
        runtime = _runtime
        if runtime is None:
            runtime = RedactionRuntime(policy)
            runtime.merge_reload_failures(_fallback_failures)
            _fallback_failures = 0
            _runtime = runtime
        else:
            runtime.set_policy(policy)
        return runtime


def record_policy_failure() -> None:
    """Record a policy load failure even when no runtime is installed yet."""
    global _fallback_failures
    runtime = _runtime
    if runtime is not None:
        runtime.record_reload_failure()
        return
    with _runtime_lock:
        _fallback_failures += 1


def load_policy_from_env() -> RedactionPolicy | None:
    """Resolve ``TRACELOOP_REDACTION_POLICY``, failing closed to ``None``."""
    raw = os.getenv(_REDACTION_POLICY_ENV_VAR)
    if not raw:
        return None

    try:
        if raw.lstrip().startswith("{"):
            payload = json.loads(raw)
        else:
            payload = json.loads(Path(raw).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logging.warning(
            "Could not read TRACELOOP_REDACTION_POLICY; redaction stays disabled: %s",
            exc,
        )
        record_policy_failure()
        return None

    if not isinstance(payload, Mapping):
        logging.warning(
            "TRACELOOP_REDACTION_POLICY must contain a JSON object; redaction stays disabled"
        )
        record_policy_failure()
        return None

    try:
        return RedactionPolicy.from_dict(payload)
    except RedactionPolicyError as exc:
        logging.warning(
            "Invalid TRACELOOP_REDACTION_POLICY (%s); redaction stays disabled: %s",
            exc.code,
            exc,
        )
        record_policy_failure()
        return None


def get_privacy_snapshot() -> PrivacySnapshot:
    """Return the current privacy snapshot."""
    runtime = _runtime
    if runtime is not None:
        return runtime.snapshot()
    with _runtime_lock:
        failures = _fallback_failures
    return PrivacySnapshot(
        policy_fingerprint=None,
        redaction=RedactionStats(),
        association=get_association_registry().snapshot(),
        policy_reload_failures=failures,
        unreadable_spans=0,
    )


def reset_privacy_state_for_tests() -> None:
    """Drop the runtime singleton (test helper)."""
    global _runtime, _fallback_failures
    with _runtime_lock:
        _runtime = None
        _fallback_failures = 0
