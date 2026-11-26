"""Tests for the redaction layer."""

import json
import os
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from traceloop.sdk import Traceloop
from traceloop.sdk.privacy import (
    RedactionAction,
    RedactionPolicy,
    RedactionPolicyError,
    RedactionRuntime,
    RedactionStats,
    Redactor,
    get_association_registry,
    get_privacy_snapshot,
)
from traceloop.sdk.privacy import attributes as privacy_attributes
from traceloop.sdk.privacy import redactor as redactor_module
from traceloop.sdk.privacy.redactor import reset_privacy_state_for_tests
from traceloop.sdk.tracing.tracing import TracerWrapper


def _policy(rules: list[dict[str, Any]] | None = None, **overrides: Any) -> RedactionPolicy:
    payload: dict[str, Any] = {
        "version": 1,
        "rules": rules or [],
        "default_action": "keep",
    }
    payload.update(overrides)
    return RedactionPolicy.from_dict(payload)


def _messages_policy() -> RedactionPolicy:
    return _policy(
        [
            {
                "name": "email-in-messages",
                "attribute": "gen_ai.input.messages",
                "detector": "email",
                "action": "mask",
            }
        ]
    )


def _card_policy() -> RedactionPolicy:
    return _policy(
        [
            {
                "name": "cards",
                "attribute": "traceloop.entity.output",
                "detector": "credit_card",
                "action": "mask",
                "keep_last": 4,
            }
        ]
    )


def test_email_in_json_messages_keeps_role_and_type() -> None:
    payload = json.dumps(
        [
            {
                "role": "user",
                "type": "text",
                "parts": [{"type": "text", "content": "mail alice@example.com now"}],
            }
        ]
    )
    redactor = Redactor(_messages_policy())
    attributes = {"gen_ai.input.messages": payload}

    result, stats = redactor.redact_attributes(
        attributes,
        get_association_registry().current(),
        content_allowed=True,
    )

    assert "alice@example.com" not in result["gen_ai.input.messages"]
    assert "[REDACTED:email]" in result["gen_ai.input.messages"]
    decoded = json.loads(result["gen_ai.input.messages"])
    assert decoded[0]["role"] == "user"
    assert decoded[0]["type"] == "text"
    assert stats.values_masked == 1
    # The input mapping is untouched.
    assert attributes["gen_ai.input.messages"] == payload


def test_credit_card_keep_last() -> None:
    redactor = Redactor(_card_policy())
    scope = get_association_registry().current()

    valid, _ = redactor.redact_attributes(
        {"traceloop.entity.output": "4111 1111 1111 1111"},
        scope,
        content_allowed=True,
    )
    assert valid["traceloop.entity.output"] == "[REDACTED:credit_card:1111]"

    invalid, stats = redactor.redact_attributes(
        {"traceloop.entity.output": "4111 1111 1111 1112"},
        scope,
        content_allowed=True,
    )
    assert invalid["traceloop.entity.output"] == "4111 1111 1111 1112"
    assert stats.values_masked == 0


def test_hash_is_deterministic_and_salt_sensitive() -> None:
    first = _policy(
        [
            {
                "name": "hash-email",
                "attribute": "traceloop.entity.input",
                "detector": "email",
                "action": "hash",
                "salt": "0123456789abcdef",
            }
        ]
    )
    second = _policy(
        [
            {
                "name": "hash-email",
                "attribute": "traceloop.entity.input",
                "detector": "email",
                "action": "hash",
                "salt": "fedcba9876543210",
            }
        ]
    )
    scope = get_association_registry().current()
    payload = json.dumps({"email": "alice@example.com"})

    first_a, _ = Redactor(first).redact_attributes(
        {"traceloop.entity.input": payload}, scope, content_allowed=True
    )
    first_b, _ = Redactor(first).redact_attributes(
        {"traceloop.entity.input": payload}, scope, content_allowed=True
    )
    second_result, _ = Redactor(second).redact_attributes(
        {"traceloop.entity.input": payload}, scope, content_allowed=True
    )

    assert first_a == first_b
    assert first_a != second_result
    assert "[HASHED:email:" in first_a["traceloop.entity.input"]


def test_second_scan_of_redacted_attributes_is_all_zero() -> None:
    redactor = Redactor(_messages_policy())
    scope = get_association_registry().current()
    payload = json.dumps({"role": "user", "content": "alice@example.com"})

    first, first_stats = redactor.redact_attributes(
        {"gen_ai.input.messages": payload}, scope, content_allowed=True
    )
    assert first_stats.values_masked == 1

    second, second_stats = redactor.redact_attributes(
        first, scope, content_allowed=True
    )
    assert second == first
    assert second_stats == RedactionStats()


def test_content_gating_drops_content_attributes() -> None:
    redactor = Redactor(_policy())
    payload = json.dumps({"role": "user", "content": "hello"})

    result, stats = redactor.redact_attributes(
        {
            "gen_ai.input.messages": payload,
            "gen_ai.output.messages": payload,
            "traceloop.entity.input": payload,
            "gen_ai.prompt.0.content": "hello",
            "gen_ai.usage.input_tokens": 15,
            "traceloop.prompt.template": "hi {name}",
        },
        get_association_registry().current(),
        content_allowed=False,
    )

    assert "gen_ai.input.messages" not in result
    assert "gen_ai.output.messages" not in result
    assert "traceloop.entity.input" not in result
    assert "gen_ai.prompt.0.content" not in result
    assert result["gen_ai.usage.input_tokens"] == 15
    assert result["traceloop.prompt.template"] == "hi {name}"
    assert stats.content_dropped == 4


def test_detectors_cover_phone_ipv4_bearer_api_key_and_secrets() -> None:
    policy = _policy(
        [
            {"name": "secret", "attribute": "notes", "detector": "secret_assignment", "action": "mask"},
            {"name": "bearer", "attribute": "auth", "detector": "bearer", "action": "mask"},
            {"name": "api", "attribute": "token", "detector": "api_key", "action": "mask"},
            {"name": "phone", "attribute": "phone", "detector": "phone_e164", "action": "mask"},
            {"name": "ip", "attribute": "ip", "detector": "ipv4", "action": "mask"},
        ]
    )
    redactor = Redactor(policy)
    scope = get_association_registry().current()

    result, stats = redactor.redact_attributes(
        {
            "notes": "password: hunter2secret",
            "auth": "Bearer abcdefghijklmno",
            "token": "sk-abcdefghijklmnop1234",
            "phone": "call +1 415-555-2671 now",
            "ip": "from 10.0.0.1 here",
        },
        scope,
        content_allowed=True,
    )

    assert result["notes"] == "password: [REDACTED:secret_assignment]"
    assert result["auth"] == "Bearer [REDACTED:bearer]"
    assert result["token"] == "[REDACTED:api_key]"
    assert result["phone"] == "call [REDACTED:phone_e164] now"
    assert result["ip"] == "from [REDACTED:ipv4] here"
    assert stats.values_masked == 5


@pytest.mark.parametrize(
    "payload,code",
    [
        (
            {"rules": [{"attribute": "a.*", "detector": "nope", "action": "mask"}]},
            "unknown_detector",
        ),
        (
            {"rules": [{"attribute": "**", "detector": "email", "action": "mask"}]},
            "bad_glob",
        ),
        (
            {
                "rules": [
                    {
                        "attribute": "a",
                        "detector": "email",
                        "action": "hash",
                        "salt": "short",
                    }
                ]
            },
            "weak_salt",
        ),
        ({"max_depth": 0}, "bad_depth"),
        ({"max_value_bytes": 10}, "bad_value_bytes"),
        ({"mask_token": "no-placeholder"}, "bad_mask_token"),
        ({"version": 0}, "bad_version"),
        (
            {
                "rules": [
                    {"attribute": "a", "detector": "email", "action": "nope"}
                ]
            },
            "unknown_action",
        ),
        (
            {
                "rules": [
                    {
                        "attribute": "a",
                        "detector": "email",
                        "action": "mask",
                        "keep_last": 4,
                    }
                ]
            },
            "bad_keep_last",
        ),
    ],
)
def test_policy_errors_expose_codes(payload: Mapping[str, object], code: str) -> None:
    with pytest.raises(RedactionPolicyError) as excinfo:
        RedactionPolicy.from_dict(payload)
    assert excinfo.value.code == code


def test_round_trip_and_fingerprint_stability() -> None:
    rules = [
        {
            "name": "r",
            "attribute": "gen_ai.input.messages",
            "detector": "email",
            "action": "mask",
        }
    ]
    first = _policy(rules)
    second = RedactionPolicy.from_dict(first.to_dict())
    assert first.fingerprint == second.fingerprint
    assert first == second

    changed = _policy(
        [
            {
                "name": "r",
                "attribute": "gen_ai.input.messages",
                "detector": "email",
                "action": "drop",
            }
        ]
    )
    assert changed.fingerprint != first.fingerprint


def test_reload_invalid_policy_keeps_previous() -> None:
    runtime = RedactionRuntime(_messages_policy())
    fingerprint = runtime.policy.fingerprint if runtime.policy else None

    assert runtime.reload_policy({"rules": [{"attribute": "a", "detector": "nope", "action": "mask"}]}) is False
    assert runtime.policy is not None
    assert runtime.policy.fingerprint == fingerprint
    assert runtime.snapshot().policy_reload_failures == 1


def test_throwing_detector_does_not_break_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(
        [
            {
                "name": "boom",
                "attribute": "custom.note",
                "detector": "email",
                "action": "mask",
            }
        ]
    )

    def boom(_: str) -> list[Any]:
        raise RuntimeError("detector exploded")

    monkeypatch.setitem(redactor_module.DETECTORS, "email", boom)
    redactor = Redactor(policy)

    result, stats = redactor.redact_attributes(
        {"custom.note": "alice@example.com", "other": "kept"},
        get_association_registry().current(),
        content_allowed=True,
    )

    assert stats.redaction_errors == 1
    # fail_closed defaults to True, so the offending attribute is dropped.
    assert "custom.note" not in result
    assert result["other"] == "kept"


@contextmanager
def _isolated_wrapper(
    exporter: InMemorySpanExporter, policy: Mapping[str, object] | None, **kwargs: Any
) -> Iterator[None]:
    saved = getattr(TracerWrapper, "instance", None)
    if saved is not None:
        del TracerWrapper.instance
    reset_privacy_state_for_tests()
    try:
        Traceloop.init(
            app_name="privacy-test",
            exporter=exporter,
            disable_batch=True,
            redaction_policy=policy,
            **kwargs,
        )
        yield
    finally:
        if hasattr(TracerWrapper, "instance"):
            del TracerWrapper.instance
        if saved is not None:
            TracerWrapper.instance = saved
        # Drop the policy so later tests cannot inherit an active runtime.
        reset_privacy_state_for_tests()


def test_init_installs_redaction_and_scan_state() -> None:
    exporter = InMemorySpanExporter()
    policy = _messages_policy().to_dict()
    with _isolated_wrapper(exporter, policy):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("redacted-span") as span:
            span.set_attribute(
                "gen_ai.input.messages",
                json.dumps({"role": "user", "content": "alice@example.com"}),
            )

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert (
            span.attributes[privacy_attributes.TRACELOOP_PRIVACY_SCAN_STATE]
            == privacy_attributes.SCAN_STATE_SCANNED
        )
        assert "[REDACTED:email]" in span.attributes["gen_ai.input.messages"]

        snapshot = get_privacy_snapshot()
        assert snapshot.policy_fingerprint == RedactionPolicy.from_dict(policy).fingerprint
        assert snapshot.redaction.values_masked >= 1


def test_repeated_on_end_marks_skipped() -> None:
    exporter = InMemorySpanExporter()
    with _isolated_wrapper(exporter, _messages_policy().to_dict()):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("repeat-span") as span:
            span.set_attribute(
                "gen_ai.input.messages",
                json.dumps({"content": "alice@example.com"}),
            )
        processor = TracerWrapper.instance._TracerWrapper__spans_processor
        readable = exporter.get_finished_spans()[0]
        processor.on_end(readable)
        assert (
            readable.attributes[privacy_attributes.TRACELOOP_PRIVACY_SCAN_STATE]
            == privacy_attributes.SCAN_STATE_SKIPPED
        )


def test_redaction_disabled_by_default_writes_nothing() -> None:
    exporter = InMemorySpanExporter()
    with _isolated_wrapper(exporter, None):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("plain-span") as span:
            span.set_attribute("gen_ai.input.messages", "alice@example.com")

        span = exporter.get_finished_spans()[0]
        assert privacy_attributes.TRACELOOP_PRIVACY_SCAN_STATE not in span.attributes
        assert span.attributes["gen_ai.input.messages"] == "alice@example.com"


def test_content_dropped_when_trace_content_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = InMemorySpanExporter()
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    with _isolated_wrapper(exporter, _messages_policy().to_dict()):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("content-span") as span:
            span.set_attribute(
                "gen_ai.input.messages",
                json.dumps({"content": "alice@example.com"}),
            )
            span.set_attribute("gen_ai.usage.input_tokens", 15)

        span = exporter.get_finished_spans()[0]
        assert "gen_ai.input.messages" not in span.attributes
        assert span.attributes["gen_ai.usage.input_tokens"] == 15
        assert get_privacy_snapshot().redaction.content_dropped >= 1


def test_env_policy_inline_json(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = InMemorySpanExporter()
    monkeypatch.setenv(
        "TRACELOOP_REDACTION_POLICY", json.dumps(_messages_policy().to_dict())
    )
    with _isolated_wrapper(exporter, None):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("env-span") as span:
            span.set_attribute(
                "gen_ai.input.messages",
                json.dumps({"content": "alice@example.com"}),
            )
        span = exporter.get_finished_spans()[0]
        assert "[REDACTED:email]" in span.attributes["gen_ai.input.messages"]


def test_env_policy_invalid_disables_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = InMemorySpanExporter()
    monkeypatch.setenv("TRACELOOP_REDACTION_POLICY", "{not-json")
    with _isolated_wrapper(exporter, None):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("env-invalid-span") as span:
            span.set_attribute("gen_ai.input.messages", "alice@example.com")
        span = exporter.get_finished_spans()[0]
        assert privacy_attributes.TRACELOOP_PRIVACY_SCAN_STATE not in span.attributes
        assert span.attributes["gen_ai.input.messages"] == "alice@example.com"
        assert get_privacy_snapshot().policy_reload_failures >= 1


def test_explicit_invalid_policy_raises(exporter) -> None:
    saved = getattr(TracerWrapper, "instance", None)
    if saved is not None:
        del TracerWrapper.instance
    try:
        with pytest.raises(RedactionPolicyError) as excinfo:
            Traceloop.init(
                exporter=exporter,
                disable_batch=True,
                redaction_policy={
                    "rules": [
                        {"attribute": "a.*", "detector": "nope", "action": "mask"}
                    ]
                },
            )
        assert excinfo.value.code == "unknown_detector"
    finally:
        if hasattr(TracerWrapper, "instance"):
            del TracerWrapper.instance
        if saved is not None:
            TracerWrapper.instance = saved


def test_redaction_errors_are_counted_on_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = InMemorySpanExporter()
    policy = _policy(
        [
            {
                "name": "boom",
                "attribute": "custom.note",
                "detector": "email",
                "action": "mask",
            }
        ],
        fail_closed=False,
    )

    def boom(_: str) -> list[Any]:
        raise RuntimeError("detector exploded")

    monkeypatch.setitem(redactor_module.DETECTORS, "email", boom)
    with _isolated_wrapper(exporter, policy.to_dict()):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("throwing-span") as span:
            span.set_attribute("custom.note", "alice@example.com")

        span = exporter.get_finished_spans()[0]
        assert span.name == "throwing-span"
        assert span.attributes["custom.note"] == "alice@example.com"
        assert get_privacy_snapshot().redaction.redaction_errors >= 1


def test_drop_rule_never_drops_scan_state() -> None:
    policy = _policy(
        [
            {
                "name": "drop-state",
                "attribute": "traceloop.privacy.*",
                "detector": "email",
                "action": "drop",
            }
        ]
    )
    redactor = Redactor(policy)
    state = privacy_attributes.SCAN_STATE_ARMED
    result, _ = redactor.redact_attributes(
        {
            privacy_attributes.TRACELOOP_PRIVACY_SCAN_STATE: state,
            "traceloop.privacy.extra": "alice@example.com",
        },
        get_association_registry().current(),
        content_allowed=True,
    )
    assert result[privacy_attributes.TRACELOOP_PRIVACY_SCAN_STATE] == state
    assert "traceloop.privacy.extra" not in result


def test_concurrent_redaction_counts_are_not_lost() -> None:
    import threading

    exporter = InMemorySpanExporter()
    policy = _policy(
        [
            {
                "name": "email-note",
                "attribute": "custom.note",
                "detector": "email",
                "action": "mask",
            }
        ]
    )
    saved_env = os.environ.get("TRACELOOP_TRACE_CONTENT")
    os.environ["TRACELOOP_TRACE_CONTENT"] = "false"
    try:
        with _isolated_wrapper(exporter, policy.to_dict()):
            before = get_privacy_snapshot().redaction
            tracer = trace.get_tracer(__name__)
            thread_count = 8
            spans_per_thread = 200
            failures: list[BaseException] = []

            def worker(index: int) -> None:
                try:
                    token = get_association_registry().bind({"user_id": index})
                    try:
                        for span_index in range(spans_per_thread):
                            with tracer.start_as_current_span(
                                f"concurrent-{index}-{span_index}"
                            ) as span:
                                span.set_attribute(
                                    "custom.note", "alice@example.com"
                                )
                                span.set_attribute(
                                    "gen_ai.input.messages",
                                    json.dumps({"content": "hi"}),
                                )
                    finally:
                        get_association_registry().reset(token)
                except BaseException as exc:  # pragma: no cover
                    failures.append(exc)

            threads = [
                threading.Thread(target=worker, args=(i,))
                for i in range(thread_count)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            assert not failures
            total = thread_count * spans_per_thread
            after = get_privacy_snapshot().redaction
            assert after.values_masked - before.values_masked == total
            assert after.content_dropped - before.content_dropped == total

            exported = [
                span
                for span in exporter.get_finished_spans()
                if span.name.startswith("concurrent-")
            ]
            assert len(exported) == total
            for span in exported:
                expected = int(span.name.split("-")[1])
                assert (
                    span.attributes[
                        f"{privacy_attributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
                    ]
                    == expected
                )
    finally:
        if saved_env is None:
            os.environ.pop("TRACELOOP_TRACE_CONTENT", None)
        else:
            os.environ["TRACELOOP_TRACE_CONTENT"] = saved_env


def test_span_postprocess_callback_sees_redacted_attributes() -> None:
    exporter = InMemorySpanExporter()
    seen: dict[str, Any] = {}

    def callback(span: Any) -> None:
        seen["messages"] = span.attributes["gen_ai.input.messages"]

    saved = getattr(TracerWrapper, "instance", None)
    if saved is not None:
        del TracerWrapper.instance
    reset_privacy_state_for_tests()
    try:
        Traceloop.init(
            app_name="privacy-test",
            exporter=exporter,
            disable_batch=True,
            redaction_policy=_messages_policy().to_dict(),
            span_postprocess_callback=callback,
        )
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("callback-span") as span:
            span.set_attribute(
                "gen_ai.input.messages",
                json.dumps({"content": "alice@example.com"}),
            )

        assert "[REDACTED:email]" in seen["messages"]
        exported = exporter.get_finished_spans()[0]
        assert "[REDACTED:email]" in exported.attributes["gen_ai.input.messages"]
    finally:
        instance = getattr(TracerWrapper, "instance", None)
        if instance is not None:
            processor = getattr(
                instance, "_TracerWrapper__spans_processor", None
            )
            if processor is not None:
                # The processor stays registered on the global tracer
                # provider; reset its on_end so later tests do not invoke
                # this test's callback.
                from opentelemetry.sdk.trace.export import SimpleSpanProcessor

                processor.on_end = SimpleSpanProcessor.on_end.__get__(
                    processor, SimpleSpanProcessor
                )
            del TracerWrapper.instance
        if saved is not None:
            TracerWrapper.instance = saved
        reset_privacy_state_for_tests()


def test_env_policy_file_path(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    exporter = InMemorySpanExporter()
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(json.dumps(_messages_policy().to_dict()), encoding="utf-8")
    monkeypatch.setenv("TRACELOOP_REDACTION_POLICY", str(policy_file))
    with _isolated_wrapper(exporter, None):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("file-policy-span") as span:
            span.set_attribute(
                "gen_ai.input.messages",
                json.dumps({"content": "alice@example.com"}),
            )
        span = exporter.get_finished_spans()[0]
        assert "[REDACTED:email]" in span.attributes["gen_ai.input.messages"]


def test_redaction_runtime_import_surface() -> None:
    # The tests reference the exported constants so they stay in sync.
    assert privacy_attributes.SCAN_STATE_SCANNED == "scanned"
    assert privacy_attributes.TRACELOOP_PRIVACY_SCAN_STATE == (
        "traceloop.privacy.scan.state"
    )
    assert RedactionAction.MASK.value == "mask"


def test_value_truncation_and_depth_limits() -> None:
    policy = _policy(
        [
            {
                "name": "notes",
                "attribute": "note",
                "detector": "email",
                "action": "mask",
            },
            {
                "name": "messages",
                "attribute": "gen_ai.output.messages",
                "detector": "email",
                "action": "mask",
            },
        ],
        max_depth=2,
        max_value_bytes=1024,
    )
    redactor = Redactor(policy)
    scope = get_association_registry().current()

    long_text = "alice@example.com " + ("x" * 5000)
    result, stats = redactor.redact_attributes(
        {"note": long_text}, scope, content_allowed=True
    )
    assert result["note"].startswith("[REDACTED:email] ")
    assert stats.truncated_values == 1

    deep = {"a": {"b": {"c": "alice@example.com"}}}
    result, stats = redactor.redact_attributes(
        {"gen_ai.output.messages": json.dumps(deep)}, scope, content_allowed=True
    )
    decoded = json.loads(result["gen_ai.output.messages"])
    assert decoded["a"]["b"]["c"] == "[TRUNCATED]"
    assert stats.truncated_values == 1


def test_policy_applies_rules_in_declaration_order() -> None:
    policy = _policy(
        [
            {
                "name": "first",
                "attribute": "note",
                "detector": "email",
                "action": "mask",
            },
            {
                "name": "second",
                "attribute": "note",
                "detector": "api_key",
                "action": "drop",
            },
        ]
    )
    result, _ = Redactor(policy).redact_attributes(
        {"note": "alice@example.com sk-abcdefgh12345678"},
        get_association_registry().current(),
        content_allowed=True,
    )
    # Only the first matching rule applies: email is masked, the API key is not.
    assert result["note"] == "[REDACTED:email] sk-abcdefgh12345678"


def test_unreadable_span_is_counted() -> None:
    runtime = RedactionRuntime(_messages_policy())

    class NoAttributes:
        pass

    stats = runtime.redact_span(NoAttributes())
    assert stats == RedactionStats()
    assert runtime.snapshot().unreadable_spans == 1
