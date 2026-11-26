"""Tests for association scopes bound through the OTel context."""

import asyncio
import threading
from typing import Any, List

import pytest
from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes

from traceloop.sdk.privacy import attributes as privacy_attributes
from traceloop.sdk.privacy.scope import (
    AssociationRegistry,
    get_association_registry,
)
from traceloop.sdk.decorators import task, workflow


def test_nested_bind_generation_is_lifo() -> None:
    registry = AssociationRegistry()

    first = registry.bind({"user_id": 1})
    second = registry.bind({"user_id": 2})

    assert first.scope.generation == 1
    assert first.previous_generation == 0
    assert second.scope.generation == 2

    assert registry.current().generation == 2

    assert registry.reset(second) is True
    assert registry.current().generation == 1

    assert registry.reset(first) is True
    assert registry.current().generation == 0


def test_out_of_order_reset_is_rejected() -> None:
    registry = AssociationRegistry()

    first = registry.bind({"user_id": 1})
    second = registry.bind({"user_id": 2})

    before = registry.snapshot()
    assert registry.reset(first) is False

    after = registry.snapshot()
    assert after.out_of_order_reset == before.out_of_order_reset + 1
    assert registry.current().generation == second.scope.generation

    assert registry.reset(second) is True


def test_bind_writes_typed_attributes_to_workflow_and_task(exporter) -> None:
    @workflow(name="privacy_scope_workflow")
    def run() -> None:
        inner()

    @task(name="privacy_scope_task")
    def inner() -> None:
        return None

    registry = get_association_registry()
    token = registry.bind({"user_id": 1, "user_name": "John Doe"})
    try:
        run()
    finally:
        registry.reset(token)

    spans = {span.name: span for span in exporter.get_finished_spans()}

    for span_name in ("privacy_scope_workflow.workflow", "privacy_scope_task.task"):
        span = spans[span_name]
        assert (
            span.attributes[f"{privacy_attributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"]
            == 1
        )
        assert (
            span.attributes[f"{privacy_attributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_name"]
            == "John Doe"
        )


def test_normalization_counters() -> None:
    registry = AssociationRegistry()

    before = registry.snapshot()
    long_value = "x" * 2000
    registry.bind(
        {
            "BAD KEY": 1,
            "container": [1, 2],
            "none_value": None,
            "long_value": long_value,
        }
    )
    after = registry.snapshot()

    assert after.invalid_keys == before.invalid_keys + 1
    assert (
        after.unsupported_association_values
        == before.unsupported_association_values + 2
    )
    assert after.truncated_strings == before.truncated_strings + 1

    scope = registry.current()
    values = dict(scope.properties)
    assert values["long_value"] == "x" * 1024
    assert "BAD KEY" not in values


def test_snapshot_id_is_stable_and_generation_sensitive() -> None:
    registry = AssociationRegistry()

    first = registry.bind({"user_id": 1, "session_id": "abc"})
    assert len(first.scope.snapshot_id) == 8

    # Same input at a different generation yields a different id.
    second = registry.bind({"user_id": 1, "session_id": "abc"})
    assert second.scope.snapshot_id != first.scope.snapshot_id

    assert registry.reset(second)
    third = registry.bind({"user_id": 1, "session_id": "abc"})
    # Same keys/values and same generation (2) reproduce the exact id.
    assert third.scope.snapshot_id == second.scope.snapshot_id


def test_allow_list_generation_and_content_decision() -> None:
    registry = AssociationRegistry()

    assert registry.current().content_decision == privacy_attributes.CONTENT_DECISION_INHERIT

    generation = registry.reload_allow_list(
        {
            "associationPropertyAllowList": [
                {"customer_id": "allowed"},
            ]
        }
    )
    assert generation == 1

    allowed = registry.bind({"customer_id": "allowed"})
    assert allowed.scope.content_decision == privacy_attributes.CONTENT_DECISION_ALLOW

    denied = registry.bind({"customer_id": "other"})
    assert denied.scope.content_decision == privacy_attributes.CONTENT_DECISION_DENY


def test_threads_do_not_share_association_scope(exporter) -> None:
    registry = get_association_registry()
    thread_count = 8
    spans_per_thread = 200
    failures: List[BaseException] = []
    tracer = trace.get_tracer(__name__)

    def worker(index: int) -> None:
        try:
            token = registry.bind({"user_id": index})
            try:
                for span_index in range(spans_per_thread):
                    with tracer.start_as_current_span(
                        f"privacy-thread-{index}-{span_index}"
                    ):
                        pass
            finally:
                assert registry.reset(token)
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures
    spans = [
        span
        for span in exporter.get_finished_spans()
        if span.name.startswith("privacy-thread-")
    ]
    assert len(spans) == thread_count * spans_per_thread
    for span in spans:
        expected = int(span.name.split("-")[2])
        assert (
            span.attributes[
                f"{privacy_attributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
            ]
            == expected
        )


@pytest.mark.asyncio
async def test_asyncio_tasks_do_not_cross_talk(exporter) -> None:
    registry = get_association_registry()
    base_generation = registry.current().generation
    tracer = trace.get_tracer(__name__)

    async def worker(index: int) -> None:
        token = registry.bind({"user_id": index})
        try:
            await asyncio.sleep(0)
            with tracer.start_as_current_span(f"privacy-async-{index}"):
                await asyncio.sleep(0)
            assert dict(registry.current().properties)["user_id"] == index
        finally:
            assert registry.reset(token)

    await asyncio.gather(*(worker(i) for i in range(32)))

    assert registry.current().generation == base_generation
    spans = {
        span.name: span
        for span in exporter.get_finished_spans()
        if span.name.startswith("privacy-async-")
    }
    assert len(spans) == 32
    for index in range(32):
        span = spans[f"privacy-async-{index}"]
        attr: Any = span.attributes[
            f"{privacy_attributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"
        ]
        assert attr == index


def test_set_association_properties_delegates_to_registry(exporter) -> None:
    from traceloop.sdk import Traceloop

    base_bind_count = get_association_registry().snapshot().bind_count

    @workflow(name="delegate_workflow")
    def run() -> None:
        Traceloop.set_association_properties({"user_id": 456})

    run()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    span = spans["delegate_workflow.workflow"]
    assert (
        span.attributes[f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"]
        == 456
    )
    assert get_association_registry().snapshot().bind_count == base_bind_count + 1
