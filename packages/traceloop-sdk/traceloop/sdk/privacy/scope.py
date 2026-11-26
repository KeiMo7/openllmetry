"""Association scopes propagated through the OpenTelemetry context.

An :class:`AssociationRegistry` binds a validated, immutable snapshot of
association properties into the current OTel context. Because the OTel context
is backed by ``contextvars``, a scope is visible only to the thread/task that
bound it (and to work spawned from it), never to sibling threads or tasks.
"""

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

from opentelemetry import context as context_api
from opentelemetry import trace

from traceloop.sdk.privacy import attributes as privacy_attributes

if TYPE_CHECKING:
    from traceloop.sdk.tracing.content_allow_list import ContentAllowList

_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_STRING_LENGTH = 1024
_ROOT_GENERATION = 0


@dataclass(frozen=True)
class AssociationScope:
    """Immutable snapshot of the association properties in effect."""

    properties: tuple[tuple[str, object], ...]
    generation: int
    allow_list_generation: int
    content_decision: str
    source: str
    snapshot_id: str

    def as_dict(self) -> dict[str, object]:
        """Return the scope properties as a plain dictionary."""
        return dict(self.properties)


@dataclass(frozen=True)
class AssociationToken:
    """Token returned by :meth:`AssociationRegistry.bind`."""

    token: object
    scope: AssociationScope
    previous_generation: int


@dataclass(frozen=True)
class AssociationSnapshot:
    """Cumulative registry counters plus the current generation."""

    generation: int
    allow_list_generation: int
    bind_count: int
    out_of_order_reset: int
    invalid_keys: int
    unsupported_association_values: int
    truncated_strings: int


def _canonical_snapshot_id(
    properties: Sequence[tuple[str, object]], generation: int
) -> str:
    payload = {
        "generation": generation,
        "properties": [[key, value] for key, value in properties],
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


_ROOT_SNAPSHOT_ID = _canonical_snapshot_id((), _ROOT_GENERATION)


def _normalize_properties(
    properties: Mapping[str, object],
) -> tuple[
    tuple[tuple[str, object], ...], int, int, int
]:
    """Return ``(normalized, invalid_keys, unsupported, truncated)``."""
    normalized: dict[str, object] = {}
    invalid_keys = 0
    unsupported = 0
    truncated = 0

    for key, value in properties.items():
        if not isinstance(key, str) or _KEY_PATTERN.match(key) is None:
            invalid_keys += 1
            continue

        if isinstance(value, str):
            if len(value) > _MAX_STRING_LENGTH:
                normalized[key] = value[:_MAX_STRING_LENGTH]
                truncated += 1
            else:
                normalized[key] = value
        elif isinstance(value, bool) or isinstance(value, (int, float)):
            normalized[key] = value
        else:
            unsupported += 1

    ordered = tuple(sorted(normalized.items(), key=lambda item: item[0]))
    return ordered, invalid_keys, unsupported, truncated


class AssociationRegistry:
    """Owns association scopes, the allow list and the cumulative counters."""

    def __init__(self, allow_list: "ContentAllowList | None" = None) -> None:
        if allow_list is None:
            from traceloop.sdk.tracing.content_allow_list import ContentAllowList

            allow_list = ContentAllowList()
        self._allow_list = allow_list
        self._entries: tuple[Mapping[str, object], ...] = ()
        self._allow_list_generation = 0
        self._allow_list_lock = threading.Lock()

        self._bind_count = 0
        self._out_of_order_reset = 0
        self._invalid_keys = 0
        self._unsupported_association_values = 0
        self._truncated_strings = 0
        self._counter_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Binding
    # ------------------------------------------------------------------
    def bind(
        self, properties: Mapping[str, object], *, source: str = "explicit"
    ) -> AssociationToken:
        """Bind ``properties`` for the current context and return a token."""
        ordered, invalid_keys, unsupported, truncated = _normalize_properties(
            properties
        )

        if invalid_keys or unsupported or truncated:
            with self._counter_lock:
                self._invalid_keys += invalid_keys
                self._unsupported_association_values += unsupported
                self._truncated_strings += truncated

        parent = self._scope_from_context()
        previous_generation = parent.generation if parent is not None else 0
        generation = previous_generation + 1

        allow_list_generation = self._allow_list_generation
        content_decision = self._decide_content(ordered, source)
        resolved_source = (
            source
            if source in privacy_attributes.ASSOCIATION_SOURCES
            else privacy_attributes.ASSOCIATION_SOURCE_EXPLICIT
        )
        scope = AssociationScope(
            properties=ordered,
            generation=generation,
            allow_list_generation=allow_list_generation,
            content_decision=content_decision,
            source=resolved_source,
            snapshot_id=_canonical_snapshot_id(ordered, generation),
        )

        scope_context = context_api.set_value(
            privacy_attributes.ASSOCIATION_SCOPE_CONTEXT_KEY, scope
        )
        # Keep the legacy context key populated for code that reads it
        # directly (the default span processor, the LangChain instrumentation
        # and metrics enrichment all expect a plain dictionary).
        scope_context = context_api.set_value(
            privacy_attributes.ASSOCIATION_PROPERTIES_CONTEXT_KEY,
            dict(ordered),
            context=scope_context,
        )
        token = context_api.attach(scope_context)

        with self._counter_lock:
            self._bind_count += 1

        self._write_current_span(ordered)

        return AssociationToken(
            token=token,
            scope=scope,
            previous_generation=previous_generation,
        )

    def current(self) -> AssociationScope:
        """Return the scope in effect for the current context."""
        scope = self._scope_from_context()
        if scope is not None:
            return scope
        return AssociationScope(
            properties=(),
            generation=_ROOT_GENERATION,
            allow_list_generation=self._allow_list_generation,
            content_decision=privacy_attributes.CONTENT_DECISION_INHERIT,
            source=privacy_attributes.ASSOCIATION_SOURCE_PROPAGATED,
            snapshot_id=_ROOT_SNAPSHOT_ID,
        )

    def reset(self, token: AssociationToken) -> bool:
        """Release a binding, last-in-first-out.

        Returns ``False`` (and leaves the current scope untouched) when the
        token is not the most recent binding in this context.
        """
        if not isinstance(token, AssociationToken):
            with self._counter_lock:
                self._out_of_order_reset += 1
            return False

        scope = self._scope_from_context()
        if scope is not token.scope:
            with self._counter_lock:
                self._out_of_order_reset += 1
            return False

        try:
            context_api.detach(cast(Any, token.token))
        except Exception:
            with self._counter_lock:
                self._out_of_order_reset += 1
            return False

        with self._counter_lock:
            self._bind_count += 1
        return True

    # ------------------------------------------------------------------
    # Allow list
    # ------------------------------------------------------------------
    def reload_allow_list(self, payload: dict) -> int:
        """Replace the allow list and return the new allow-list generation."""
        entries: tuple[Mapping[str, object], ...]
        if isinstance(payload, Mapping):
            raw_entries = payload.get("associationPropertyAllowList")
        elif isinstance(payload, (list, tuple)):
            raw_entries = payload
        else:
            raw_entries = None

        if isinstance(raw_entries, (list, tuple)):
            entries = tuple(
                entry for entry in raw_entries if isinstance(entry, Mapping)
            )
        else:
            entries = ()

        with self._allow_list_lock:
            self._entries = entries
            self._allow_list_generation += 1
            generation = self._allow_list_generation
            if isinstance(payload, Mapping):
                try:
                    self._allow_list.load(payload)
                except Exception:
                    # The payload had an unexpected shape; our own snapshot is
                    # authoritative for content decisions.
                    pass

        return generation

    def _decide_content(
        self, properties: Sequence[tuple[str, object]], source: str
    ) -> str:
        if source == privacy_attributes.ASSOCIATION_SOURCE_ALLOW_LIST:
            return privacy_attributes.CONTENT_DECISION_ALLOW
        entries = self._entries
        if not entries:
            return privacy_attributes.CONTENT_DECISION_INHERIT
        values = dict(properties)
        for entry in entries:
            if all(values.get(key) == value for key, value in entry.items()):
                return privacy_attributes.CONTENT_DECISION_ALLOW
        return privacy_attributes.CONTENT_DECISION_DENY

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    def snapshot(self) -> AssociationSnapshot:
        scope = self._scope_from_context()
        generation = scope.generation if scope is not None else _ROOT_GENERATION
        with self._counter_lock:
            return AssociationSnapshot(
                generation=generation,
                allow_list_generation=self._allow_list_generation,
                bind_count=self._bind_count,
                out_of_order_reset=self._out_of_order_reset,
                invalid_keys=self._invalid_keys,
                unsupported_association_values=self._unsupported_association_values,
                truncated_strings=self._truncated_strings,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _scope_from_context() -> AssociationScope | None:
        value = context_api.get_value(
            privacy_attributes.ASSOCIATION_SCOPE_CONTEXT_KEY
        )
        if isinstance(value, AssociationScope):
            return value
        return None

    @staticmethod
    def _write_current_span(
        properties: Sequence[tuple[str, object]],
    ) -> None:
        has_scope = (
            context_api.get_value(privacy_attributes.WORKFLOW_NAME_CONTEXT_KEY)
            is not None
            or context_api.get_value(privacy_attributes.ENTITY_NAME_CONTEXT_KEY)
            is not None
        )
        if not has_scope:
            return

        span = trace.get_current_span()
        if not span.is_recording():
            return
        for key, value in properties:
            span.set_attribute(
                privacy_attributes.association_property_key(key),
                cast(Any, value),
            )


_registry: AssociationRegistry | None = None
_registry_lock = threading.Lock()


def get_association_registry() -> AssociationRegistry:
    """Return the process-wide :class:`AssociationRegistry` singleton."""
    global _registry
    registry = _registry
    if registry is not None:
        return registry
    with _registry_lock:
        if _registry is None:
            _registry = AssociationRegistry()
        return _registry
