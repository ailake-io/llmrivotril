"""Small helpers for running synchronous work from async APIs."""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, TypeVar

ResultT = TypeVar("ResultT")


async def run_sync(func: Callable[..., ResultT], *args: Any, **kwargs: Any) -> ResultT:
    """Run ``func`` in a short-lived executor and release its worker promptly.

    ``asyncio.to_thread`` uses the event loop's process-lifetime default
    executor. That is usually fine for applications, but it can leave worker
    threads alive after a test loop is closed. A scoped executor keeps this
    bridge predictable and makes shutdown part of the call's lifecycle.
    """
    call = partial(func, *args, **kwargs)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="llmrivotril") as executor:
        future = executor.submit(call)
        # Polling the concurrent future keeps completion delivery on the
        # event-loop thread. This avoids a Python 3.13/pytest-asyncio edge
        # case where run_in_executor's cross-thread callback can be lost.
        while not future.done():
            await asyncio.sleep(0.001)
        return future.result()
