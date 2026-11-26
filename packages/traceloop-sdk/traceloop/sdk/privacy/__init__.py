"""Association scoping and span redaction for the Traceloop SDK.

The layer is opt-in: it is only installed when ``Traceloop.init`` receives a
redaction policy (or ``TRACELOOP_REDACTION_POLICY`` is set). With no policy it
is entirely inert.
"""

from traceloop.sdk.privacy.policy import (
    RedactionAction as RedactionAction,
    RedactionPolicy as RedactionPolicy,
    RedactionPolicyError as RedactionPolicyError,
    RedactionRule as RedactionRule,
)
from traceloop.sdk.privacy.redactor import (
    PrivacySnapshot as PrivacySnapshot,
    RedactionRuntime as RedactionRuntime,
    RedactionStats as RedactionStats,
    Redactor as Redactor,
    get_privacy_snapshot as get_privacy_snapshot,
)
from traceloop.sdk.privacy.scope import (
    AssociationRegistry as AssociationRegistry,
    AssociationScope as AssociationScope,
    AssociationSnapshot as AssociationSnapshot,
    AssociationToken as AssociationToken,
    get_association_registry as get_association_registry,
)

__all__ = [
    "RedactionAction",
    "RedactionRule",
    "RedactionPolicy",
    "RedactionPolicyError",
    "RedactionStats",
    "Redactor",
    "AssociationScope",
    "AssociationToken",
    "AssociationSnapshot",
    "AssociationRegistry",
    "RedactionRuntime",
    "PrivacySnapshot",
    "get_association_registry",
    "get_privacy_snapshot",
]
