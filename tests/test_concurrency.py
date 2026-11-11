import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from llmrivotril.concurrency import CapacityLimiter


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_capacity(value):
    with pytest.raises(ValueError, match="max_concurrent_calls"):
        CapacityLimiter(value)


def test_thread_waiter_is_released_after_exception():
    limiter = CapacityLimiter(1)
    started, entered = Event(), Event()

    def worker():
        started.set()
        with limiter.slot():
            entered.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError), limiter.slot():
            future = pool.submit(worker)
            assert started.wait(2)
            assert not entered.wait(0.05)
            raise RuntimeError("release this slot")
        future.result(timeout=2)
    assert entered.is_set()


@pytest.mark.asyncio
async def test_async_waiter_cancellation_does_not_leak_capacity():
    limiter = CapacityLimiter(1)
    started = asyncio.Event()

    async def waiter():
        started.set()
        async with limiter.async_slot():
            pytest.fail("waiter acquired an occupied slot")

    with limiter.slot():
        task = asyncio.create_task(waiter())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def acquire_again():
        async with limiter.async_slot():
            pass

    await asyncio.wait_for(acquire_again(), timeout=2)


@pytest.mark.asyncio
async def test_async_tasks_share_capacity():
    limiter = CapacityLimiter(2)
    active = peak = 0

    async def worker():
        nonlocal active, peak
        async with limiter.async_slot():
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.001)
            active -= 1

    await asyncio.gather(*(worker() for _ in range(10)))
    assert peak == 2
