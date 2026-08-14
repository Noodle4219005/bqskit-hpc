"""This module implements the RuntimeTask class."""
from __future__ import annotations

import inspect
import logging
from typing import Any
from typing import Coroutine

import dill

from bqskit.runtime.address import RuntimeAddress

# Service classes for the worker's ready queue. Lower runs first.
#
# The queue is FIFO and non-preemptive, which forces a trade nobody wants:
# speculate shallowly and leave cores idle, or speculate deeply and put
# speculative work in front of the work that decides the answer. Measured on
# adder_8 at msz=4, that trade is 59.2% occupancy while LEAP is active on one
# node, or 0.87x wall once speculation gets ahead of the critical path.
#
# A service class breaks the trade rather than tuning a point on it:
# speculation may go as deep as memory allows -- and memory is absurdly cheap
# here, 1 GB of stored nodes being worth 77 CPU-hours of recomputation -- while
# critical work still starts the moment a core frees.
#
# Values are spaced so classes can be inserted between them later without
# renumbering anything already serialised into an in-flight task.
PRIORITY_CRITICAL = 0
"""Work the answer waits on. Everything defaults here."""

PRIORITY_SPECULATIVE = 10
"""Runahead: worth doing when a core would otherwise idle, never before."""


class RuntimeTask:
    """
    BQSKit Runtime's internal task structure.

    The task exists in two phases:
        1. In the initial phase, the task is a context-less function
        and arguments grouped with a return address. It is expected
        when the task is finished executing, to ship the result to
        the return address.

        2. When the task has been scheduled to run on a worker, it is
        initialized into a coroutine that can be executed in steps.
    """

    task_counter = 0
    """Per-process task id counter."""

    def __init__(
        self,
        fnargs: tuple[Any, Any, Any],
        return_address: RuntimeAddress,
        comp_task_id: int,
        breadcrumbs: tuple[RuntimeAddress, ...],
        logging_level: int | None = None,
        max_logging_depth: int = -1,
        task_name: str | None = None,
        log_context: dict[str, str] = {},
        priority: int = PRIORITY_CRITICAL,
        cost_hint: float = 0.0,
    ) -> None:
        """Create the task with a new id and return address."""
        RuntimeTask.task_counter += 1
        self.task_id = RuntimeTask.task_counter

        self.priority = priority
        """Service class for the worker's ready queue; lower runs first.

        Defaulted to PRIORITY_CRITICAL so every existing caller keeps exactly
        the behaviour it has today: with a single class in play, a queue
        ordered by (class, arrival) IS the FIFO it replaces.
        """

        self.cost_hint = cost_hint
        """Estimated processing cost used by optional pool ordering."""

        self.serialized_fnargs = dill.dumps(fnargs)
        self._fnargs: tuple[Any, Any, Any] | None = None
        self._name = fnargs[0].__name__ if task_name is None else task_name
        """Tuple of function pointer, arguments, and keyword arguments."""

        self.return_address = return_address
        """
        Where the result of this task should be sent.

        This doubles as a unique system-wide id for the task.
        """

        self.logging_level = logging_level or 0
        """Logs with levels >= to this get emitted, if None always emit."""

        self.comp_task_id = comp_task_id
        """The mailbox id of the this task's root task."""

        self.breadcrumbs = breadcrumbs
        """All of this task's parent tasks' addresses in order."""

        self.max_logging_depth = max_logging_depth
        """Logs are not emitted for tasks with this many or more parents."""

        self.coro: Coroutine[Any, Any, Any] | None = None
        """The coroutine containing this tasks code."""

        self.desired_box_id: int | None = None
        """When waiting on a mailbox, this stores that mailbox's id."""

        self.owned_mailboxes: list[int] = []
        """The mailbox ids that this task owns."""

        self.wake_on_next: bool = False
        """Set to true if this task should wake immediately on a result."""

        self.log_context: dict[str, str] = log_context
        """Additional context to be logged with this task."""

        self.msg_buffer: list[Any] = []

    @property
    def fnargs(self) -> tuple[Any, Any, Any]:
        """Return the function pointer, arguments, and keyword arguments."""
        if self._fnargs is None:
            self._fnargs = dill.loads(self.serialized_fnargs)
        assert self._fnargs is not None  # for type checker
        return self._fnargs

    def step(self, send_val: Any = None) -> Any:
        """Execute one step of the task."""
        if self.coro is None:
            raise RuntimeError('Task has not been initialized.')

        # Reset previously set await flags
        self.wake_on_next = False
        self.desired_box_id = None

        # Set logging level
        old_level = logging.getLogger().getEffectiveLevel()
        if (
            self.max_logging_depth < 0
            or len(self.breadcrumbs) <= self.max_logging_depth
        ):
            logging.getLogger().setLevel(self.logging_level)
        else:
            logging.getLogger().setLevel(100)

        # Execute a task step
        to_return = self.coro.send(send_val)

        # Reset logging
        logging.getLogger().setLevel(old_level)

        return to_return

    @property
    def unique_id(self) -> RuntimeAddress:
        """Return the task's system-wide unique id."""
        return self.return_address

    def start(self) -> None:
        """Initialize the task."""
        self.coro = self.run()

    def cancel(self) -> None:
        """Ask the coroutine to gracefully exit."""
        if self.coro is not None:
            # If this call to "close" raises a RuntimeError,
            # it is likely a blanket try/accept catching the
            # error used to stop the coroutine, preventing
            # it from stopping correctly.
            try:
                self.coro.close()
            except ValueError:
                # Coroutine is running and cannot be closed.
                pass
        else:
            raise RuntimeError('Task was cancelled with None coroutine.')

    async def run(self) -> Any:
        """Task coroutine wrapper."""
        if inspect.iscoroutinefunction(self.fnargs[0]):
            return await self.fnargs[0](*self.fnargs[1], **self.fnargs[2])
        return self.fnargs[0](*self.fnargs[1], **self.fnargs[2])

    def is_descendant_of(self, addr: RuntimeAddress) -> bool:
        """Return true if `addr` identifies a parent (or this) task."""
        return addr == self.return_address or addr in self.breadcrumbs

    def __str__(self) -> str:
        """Return a string representation of the task."""
        return f'{self._name}'

    def __repr__(self) -> str:
        """Return a string representation of the task."""
        return f'<RuntimeTask {self.unique_id} {self._name}>'
