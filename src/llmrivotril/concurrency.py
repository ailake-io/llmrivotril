"""One per-agent capacity shared by synchronous and asynchronous calls."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from threading import Condition


class CapacityLimiter:
    def __init__(self, limit: int | None = None) -> None:
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ValueError("max_concurrent_calls must be a positive integer or None")
        self.limit = limit
        self._active = 0
        self._condition = Condition()

    def _try_acquire(self) -> bool:
        with self._condition:
            if self.limit is not None and self._active >= self.limit:
                return False
            self._active += 1
            return True

    def _release(self) -> None:
        with self._condition:
            self._active -= 1
            self._condition.notify()

    @contextmanager
    def slot(self) -> Iterator[None]:
        with self._condition:
            while not self._try_acquire():
                self._condition.wait()
        try:
            yield
        finally:
            self._release()

    @asynccontextmanager
    async def async_slot(self) -> AsyncIterator[None]:
        # No executor thread waits on a semaphore: cancellation while waiting
        # cannot acquire and strand a slot after the task has disappeared.
        while not self._try_acquire():
            await asyncio.sleep(0.01)
        try:
            yield
        finally:
            self._release()
