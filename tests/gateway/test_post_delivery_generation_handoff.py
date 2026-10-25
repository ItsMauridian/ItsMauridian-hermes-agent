"""Regression: a queued follow-up must not steal the previous turn's post-delivery callbacks.

``_process_message_background`` hands the session to a drain task when a follow-up is pending
(goal continuation, background-process or watch-pattern notification, a message sent while busy).
The drain task reuses the SAME interrupt ``Event`` object and stamps its own run generation on it
while the previous task is still unwinding (``_release_turn_marker`` awaits a durable write before
the finally block fires the post-delivery callback). Two things used to go wrong:

1. The finally block read the generation from the shared Event AFTER the next turn stamped it, so
   the previous turn popped with the wrong generation and its callbacks never fired.
2. The next turn's registration (newer generation) overwrote the previous turn's unfired slot.

The user-visible symptom is progress bubbles (tool lines, status lines) that are never cleaned up
after a turn that was followed by a queued turn. The previous turn's callbacks must fire exactly
once, for the generation that turn owned.
"""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text=None, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return {}


def _source(chat_id="42"):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm")


def _event(text, chat_id="42"):
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=_source(chat_id))


def _make_adapter():
    adapter = _StubAdapter(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)

    async def _send_with_retry(*args, **kwargs):
        return None

    adapter._send_with_retry = _send_with_retry
    return adapter


async def _run_two_turns(adapter, *, slow_unwind: bool):
    sk = build_session_key(_source())
    fired: list[str] = []
    second_done = asyncio.Event()

    async def handler(event):
        active = adapter._active_sessions[sk]
        if event.text == "M1":
            active._hermes_run_generation = 1
            adapter.register_post_delivery_callback(sk, lambda: fired.append("turn1"), generation=1)
            # A follow-up is already pending when the first turn finishes.
            adapter._pending_messages[sk] = _event("M2")
            return "first"
        active._hermes_run_generation = 2
        adapter.register_post_delivery_callback(sk, lambda: fired.append("turn2"), generation=2)
        second_done.set()
        return "second"

    adapter._message_handler = handler

    if slow_unwind:
        original = adapter._release_turn_marker

        async def slow_release(event):
            # The durable turn-marker write yields to the loop, so the drain task starts
            # before the first task reaches its finally block.
            await asyncio.sleep(0.05)
            return await original(event)

        adapter._release_turn_marker = slow_release

    await adapter.handle_message(_event("M1"))
    await asyncio.wait_for(second_done.wait(), timeout=5)
    # Let both tasks finish unwinding.
    for _ in range(100):
        task = adapter._session_tasks.get(sk)
        if (task is None or task.done()) and "turn2" in fired:
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.2)
    return fired


@pytest.mark.asyncio
async def test_previous_turn_callback_fires_when_followup_restamps_generation():
    fired = await _run_two_turns(_make_adapter(), slow_unwind=True)
    assert fired.count("turn1") == 1, fired
    assert fired.count("turn2") == 1, fired


@pytest.mark.asyncio
async def test_previous_turn_callback_fires_without_unwind_delay():
    fired = await _run_two_turns(_make_adapter(), slow_unwind=False)
    assert fired.count("turn1") == 1, fired
    assert fired.count("turn2") == 1, fired


@pytest.mark.asyncio
async def test_newer_generation_registration_keeps_older_unfired_lane():
    adapter = _make_adapter()
    fired: list[str] = []
    adapter.register_post_delivery_callback("s", lambda: fired.append("old"), generation=1)
    adapter.register_post_delivery_callback("s", lambda: fired.append("new"), generation=2)

    old = adapter.pop_post_delivery_callback("s", generation=1)
    new = adapter.pop_post_delivery_callback("s", generation=2)
    assert callable(old) and callable(new)
    for cb in (old, new):
        result = cb()
        if asyncio.iscoroutine(result):
            await result
    assert fired == ["old", "new"]


@pytest.mark.asyncio
async def test_stale_generation_cannot_register_after_newer_lane_exists():
    adapter = _make_adapter()
    adapter.register_post_delivery_callback("s", lambda: None, generation=5)
    adapter.register_post_delivery_callback("s", lambda: None, generation=3)
    assert adapter.pop_post_delivery_callback("s", generation=3) is None
    assert callable(adapter.pop_post_delivery_callback("s", generation=5))


@pytest.mark.asyncio
async def test_generation_pop_never_returns_another_generations_callback():
    adapter = _make_adapter()
    adapter.register_post_delivery_callback("s", lambda: None, generation=2)
    assert adapter.pop_post_delivery_callback("s", generation=1) is None
    assert callable(adapter.pop_post_delivery_callback("s", generation=2))
    assert adapter.pop_post_delivery_callback("s", generation=2) is None
