"""Regression: a busy follow-up (``/queue``, a message sent while the agent works) owes the same
post-delivery callbacks as a first message.

``dispatch_shared_busy`` runs the follow-up's own lifecycle: it awaits the message handler and
delivers the reply itself, so ``BasePlatformAdapter._process_message_background`` (whose finally
block fires the one-shot post-delivery callbacks) never wraps that turn. The progress-bubble cleanup
(``display.cleanup_progress``) and the background-review release are such callbacks, so before this
change every tool-progress bubble of a queued turn stayed in the chat forever.

The turn stamps its run generation on the event it ran; the lifecycle fires exactly that
generation's callbacks after the reply is delivered.
"""

import asyncio

import pytest

from gateway import session_ingress
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content=None, reply_to=None, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {}


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")


def _event(text="ping"):
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=_source())


@pytest.mark.asyncio
async def test_busy_followup_fires_its_own_callbacks_after_delivery():
    order: list = []
    holder = {}

    async def handler(event):
        adapter, sk = holder["adapter"], holder["sk"]
        event._hermes_run_generation = 7  # what _hmwa_prepare_turn stamps
        adapter.register_post_delivery_callback(sk, lambda: order.append(("cleanup", 7)), generation=7)
        return "reply"

    adapter = _StubAdapter(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
    sk = build_session_key(_source())
    holder.update(adapter=adapter, sk=sk)
    adapter._message_handler = handler

    async def fake_deliver(adapter_, event_, session_key_, response_):
        order.append(("delivered", response_))

    original = session_ingress.deliver_response
    session_ingress.deliver_response = fake_deliver
    try:
        await session_ingress.dispatch_shared_busy(adapter, _event(), sk)
        for task in list(adapter._background_tasks):
            await asyncio.wait_for(task, timeout=5)
    finally:
        session_ingress.deliver_response = original

    assert order == [("delivered", "reply"), ("cleanup", 7)], order


@pytest.mark.asyncio
async def test_busy_followup_leaves_other_generations_alone():
    order: list = []
    holder = {}

    async def handler(event):
        adapter, sk = holder["adapter"], holder["sk"]
        # A previous turn (generation 6) is still unwinding on its own lifecycle.
        adapter.register_post_delivery_callback(sk, lambda: order.append(("cleanup", 6)), generation=6)
        event._hermes_run_generation = 7
        adapter.register_post_delivery_callback(sk, lambda: order.append(("cleanup", 7)), generation=7)
        return "reply"

    adapter = _StubAdapter(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
    sk = build_session_key(_source())
    holder.update(adapter=adapter, sk=sk)
    adapter._message_handler = handler

    async def fake_deliver(adapter_, event_, session_key_, response_):
        order.append(("delivered", response_))

    original = session_ingress.deliver_response
    session_ingress.deliver_response = fake_deliver
    try:
        await session_ingress.dispatch_shared_busy(adapter, _event(), sk)
        for task in list(adapter._background_tasks):
            await asyncio.wait_for(task, timeout=5)
    finally:
        session_ingress.deliver_response = original

    assert ("cleanup", 7) in order and ("cleanup", 6) not in order, order
    # The older lane is still registered for the lifecycle that owns it.
    assert callable(adapter.pop_post_delivery_callback(sk, generation=6))


@pytest.mark.asyncio
async def test_busy_followup_without_a_returning_handler_fires_nothing():
    order: list = []

    async def handler(event):
        raise RuntimeError("handler blew up")

    adapter = _StubAdapter(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
    sk = build_session_key(_source())
    adapter._message_handler = handler
    adapter.register_post_delivery_callback(sk, lambda: order.append("other"), generation=3)

    async def _quiet(*args, **kwargs):
        return None

    adapter._notify_turn_error = _quiet
    with pytest.raises(RuntimeError):
        await session_ingress.dispatch_shared_busy(adapter, _event(), sk)
    assert order == []
