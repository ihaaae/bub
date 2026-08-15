from __future__ import annotations

from pathlib import Path

import pytest

from bub.builtin.store import ForkTapeStore
from bub.builtin.tape import Tape
from bub.tape import AsyncTapeStoreAdapter, InMemoryTapeStore, TapeContext


@pytest.mark.asyncio
async def test_tape_fork_binds_temporary_fork_store_to_scoped_tape(tmp_path: Path) -> None:
    parent = InMemoryTapeStore()
    root = Tape(tmp_path, AsyncTapeStoreAdapter(parent), TapeContext()).scoped("test-tape")

    async with root.fork_tape(merge_back=True) as forked:
        first_store = forked.store

        assert isinstance(first_store, ForkTapeStore)
        assert first_store is not root.store

        await forked.append_event("step", {"value": 1})
        assert parent.read("test-tape") is None

    assert [entry.payload["name"] for entry in parent.read("test-tape") or []] == ["step"]

    async with root.fork_tape(merge_back=False) as forked:
        second_store = forked.store
        await forked.append_event("step", {"value": 2})

    assert isinstance(second_store, ForkTapeStore)
    assert second_store is not first_store
    assert [entry.payload["data"]["value"] for entry in parent.read("test-tape") or []] == [1]


@pytest.mark.asyncio
async def test_tape_info_reports_last_token_cache_hit_rate(tmp_path: Path) -> None:
    tape = Tape(tmp_path, AsyncTapeStoreAdapter(InMemoryTapeStore()), TapeContext()).scoped("test-tape")
    await tape.record_chat(
        run_id="run-1",
        system_prompt=None,
        new_messages=[],
        response_text=None,
        usage={
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "total_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 60},
        },
    )

    info = await tape.info()

    assert info.last_token_usage == 100
    assert info.last_token_cache_hit_rate == 0.75


@pytest.mark.asyncio
async def test_tape_info_omits_cache_hit_rate_when_usage_has_no_cache_details(tmp_path: Path) -> None:
    tape = Tape(tmp_path, AsyncTapeStoreAdapter(InMemoryTapeStore()), TapeContext()).scoped("test-tape")
    await tape.record_chat(
        run_id="run-1",
        system_prompt=None,
        new_messages=[],
        response_text=None,
        usage={"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
    )

    info = await tape.info()

    assert info.last_token_cache_hit_rate is None


@pytest.mark.asyncio
async def test_tape_cost_aggregates_usage_and_provider_reported_cost(tmp_path: Path) -> None:
    tape = Tape(tmp_path, AsyncTapeStoreAdapter(InMemoryTapeStore()), TapeContext()).scoped("test-tape")
    await tape.record_chat(
        run_id="run-1",
        system_prompt=None,
        new_messages=[],
        response_text=None,
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_tokens_details": {"cached_tokens": 40},
            "cost": 0.001,
        },
    )
    await tape.record_chat(
        run_id="run-2",
        system_prompt=None,
        new_messages=[],
        response_text=None,
        usage={
            "input_tokens": 50,
            "output_tokens": 5,
            "total_tokens": 55,
            "input_tokens_details": {"cached_tokens": 20},
            "cost": 0.002,
        },
    )

    cost = await tape.cost()

    assert cost.name == "test-tape"
    assert cost.cached_input_tokens == 60
    assert cost.uncached_input_tokens == 90
    assert cost.output_tokens == 15
    assert cost.cost == pytest.approx(0.003)


@pytest.mark.asyncio
async def test_tape_cost_marks_cost_unknown_when_provider_does_not_report_it(tmp_path: Path) -> None:
    tape = Tape(tmp_path, AsyncTapeStoreAdapter(InMemoryTapeStore()), TapeContext()).scoped("test-tape")
    await tape.record_chat(
        run_id="run-1",
        system_prompt=None,
        new_messages=[],
        response_text=None,
        usage={"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
    )

    cost = await tape.cost()

    assert cost.cached_input_tokens == 0
    assert cost.uncached_input_tokens == 8
    assert cost.output_tokens == 2
    assert cost.cost is None
