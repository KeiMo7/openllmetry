"""Constants owned by the privacy layer.

Every span attribute name introduced by the privacy layer lives here so that
business code never has to spell a literal attribute name.
"""

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

# Attribute written on every span handled by the redaction runtime. It records
# the lifecycle of the scan and is reserved: no drop rule may remove it.
TRACELOOP_PRIVACY_SCAN_STATE = "traceloop.privacy.scan.state"

SCAN_STATE_ARMED = "armed"
SCAN_STATE_SCANNED = "scanned"
SCAN_STATE_FAILED = "failed"
SCAN_STATE_SKIPPED = "skipped"

SCAN_STATES = frozenset(
    {
        SCAN_STATE_ARMED,
        SCAN_STATE_SCANNED,
        SCAN_STATE_FAILED,
        SCAN_STATE_SKIPPED,
    }
)

# Context keys shared with the existing association-property plumbing.
ASSOCIATION_PROPERTIES_CONTEXT_KEY = "association_properties"
ASSOCIATION_SCOPE_CONTEXT_KEY = "traceloop.privacy.association_scope"
WORKFLOW_NAME_CONTEXT_KEY = "workflow_name"
ENTITY_NAME_CONTEXT_KEY = "entity_name"
CONTENT_OVERRIDE_CONTEXT_KEY = "override_enable_content_tracing"
TRACE_CONTENT_ENV_VAR = "TRACELOOP_TRACE_CONTENT"

# Existing semantic-convention attributes handled by the redaction layer.
TRACELOOP_ENTITY_INPUT = SpanAttributes.TRACELOOP_ENTITY_INPUT
TRACELOOP_ENTITY_OUTPUT = SpanAttributes.TRACELOOP_ENTITY_OUTPUT
TRACELOOP_PROMPT_TEMPLATE = SpanAttributes.TRACELOOP_PROMPT_TEMPLATE
TRACELOOP_ASSOCIATION_PROPERTIES = SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES

GEN_AI_INPUT_MESSAGES = GenAIAttributes.GEN_AI_INPUT_MESSAGES
GEN_AI_OUTPUT_MESSAGES = GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
GEN_AI_PROMPT_PREFIX = f"{GenAIAttributes.GEN_AI_PROMPT}."
GEN_AI_COMPLETION_PREFIX = f"{GenAIAttributes.GEN_AI_COMPLETION}."

TRACELOOP_ASSOCIATION_PROPERTIES_PREFIX = (
    f"{TRACELOOP_ASSOCIATION_PROPERTIES}."
)

# Attributes carrying a JSON payload that is traversed recursively.
JSON_CONTENT_ATTRIBUTES = frozenset(
    {
        TRACELOOP_ENTITY_INPUT,
        TRACELOOP_ENTITY_OUTPUT,
        GEN_AI_INPUT_MESSAGES,
        GEN_AI_OUTPUT_MESSAGES,
    }
)

# Attributes carrying plain text content.
TEXT_CONTENT_ATTRIBUTE_PREFIXES = (
    GEN_AI_PROMPT_PREFIX,
    GEN_AI_COMPLETION_PREFIX,
)
TEXT_CONTENT_ATTRIBUTES = frozenset({TRACELOOP_PROMPT_TEMPLATE})

# Attributes removed when content capture is not allowed. ``prompt.template``
# is intentionally not part of this set (its existing semantics are unchanged).
CONTENT_DROP_ATTRIBUTES = JSON_CONTENT_ATTRIBUTES
CONTENT_DROP_PREFIXES = TEXT_CONTENT_ATTRIBUTE_PREFIXES

# Value markers produced by the redaction layer.
REDACTED_PREFIX = "[REDACTED:"
HASHED_PREFIX = "[HASHED:"
TRUNCATED_MARKER = "[TRUNCATED]"

# Association scope vocabulary.
CONTENT_DECISION_INHERIT = "inherit"
CONTENT_DECISION_ALLOW = "allow"
CONTENT_DECISION_DENY = "deny"
CONTENT_DECISIONS = frozenset(
    {
        CONTENT_DECISION_INHERIT,
        CONTENT_DECISION_ALLOW,
        CONTENT_DECISION_DENY,
    }
)

ASSOCIATION_SOURCE_EXPLICIT = "explicit"
ASSOCIATION_SOURCE_ALLOW_LIST = "allow_list"
ASSOCIATION_SOURCE_PROPAGATED = "propagated"
ASSOCIATION_SOURCES = frozenset(
    {
        ASSOCIATION_SOURCE_EXPLICIT,
        ASSOCIATION_SOURCE_ALLOW_LIST,
        ASSOCIATION_SOURCE_PROPAGATED,
    }
)


def association_property_key(key: str) -> str:
    """Return the span attribute name for an association property."""
    return f"{TRACELOOP_ASSOCIATION_PROPERTIES_PREFIX}{key}"


def is_content_attribute(key: str) -> bool:
    """Return whether ``key`` carries captured prompt/completion content."""
    if key in CONTENT_DROP_ATTRIBUTES:
        return True
    return key.startswith(CONTENT_DROP_PREFIXES)


def is_json_content_attribute(key: str) -> bool:
    """Return whether ``key`` carries a JSON prompt/completion payload."""
    return key in JSON_CONTENT_ATTRIBUTES


def has_redaction_marker(value: str) -> bool:
    """Return whether ``value`` was already redacted by this layer."""
    return REDACTED_PREFIX in value or HASHED_PREFIX in value
