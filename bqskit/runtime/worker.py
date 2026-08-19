"""This module implements BQSKit Runtime's Worker."""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process
from multiprocessing.connection import Client
from multiprocessing.connection import Connection
from queue import Empty
from queue import PriorityQueue
from queue import Queue
from threading import Lock
from threading import Thread
from typing import Any
from typing import Callable
from typing import cast
from typing import List
from typing import Sequence
from typing import Tuple

from bqskit.runtime import default_worker_port
from bqskit.runtime import set_blas_thread_counts
from bqskit.runtime.address import RuntimeAddress
from bqskit.runtime.future import RuntimeFuture
from bqskit.runtime.message import RuntimeMessage
from bqskit.runtime.result import RuntimeResult
from bqskit.runtime.task import PRIORITY_CRITICAL
from bqskit.runtime.task import RuntimeTask


_logger = logging.getLogger(__name__)

_TASKLOG_DIR = os.environ.get('BQSKIT_TASKLOG_DIR')
"""Directory for the per-step task log; unset disables the probe entirely.

Records (worker, start, end, service class, own-or-borrowed, block) so a
per-core timeline can be reconstructed. /proc/stat says a core was busy; only
this says with what.
"""

# Where a manager's occupancy broadcast is parked for the passes running on
# this worker. Read it with:
#
#     idle, total, stamp = get_runtime().get_cache().get(
#         '__bqskit_occupancy__', (None, None, 0.0))
#
# A missing key means "unknown", never "zero idle": the broadcast only exists
# in the detached runtime, and a pass that read absent-as-zero would turn
# itself off in every attached run.
_OCCUPANCY_KEY = '__bqskit_occupancy__'

# How many tasks a worker may have STARTED but not finished.
#
# A batch arriving at a worker starts exactly one task and delays the rest
# (see the SUBMIT_BATCH handler), and a delayed task is promoted only once the
# ready queue is completely empty. Those two rules together throttle the whole
# cluster: ForEachBlockPass hands out ~17 block-synthesis tasks per worker, the
# worker starts ONE, that block's instantiate children then keep the ready
# queue permanently non-empty, and the other 16 blocks never start. The number
# of blocks alive cluster-wide collapses from 1994 to roughly one per worker,
# so the machine runs on a fraction of the parallelism the circuit contains.
#
# Measured: 112 processes each at 21-44% of a core, uniformly -- not one
# saturated bottleneck but everybody starved.
#
# 1 reproduces the historical behaviour exactly, which is what keeps the digest
# gate meaningful. Raising it starts more parents, and parents are what
# GENERATE work for the rest of the cluster. The cost is memory, and in this
# domain memory is the cheap resource: 1 GB of stored search nodes is worth 77
# CPU-hours of recomputation.


@dataclass
class WorkerMailbox:
    """
    A mailbox on a worker is a final destination for a task's result.

    When a task is created, a mailbox is also created with an associated future.
    The parent task can await on the future, letting the worker's event loop
    know it is waiting on the associated result. When a result arrives, it is
    placed in the appropriate mailbox and the waiting task is placed into the
    ready queue.
    """
    expecting_single_result: bool = False
    expected_num_results: int = 0
    result: Any = None
    num_results: int = 0
    dest_addr: RuntimeAddress | None = None
    fresh_results: list[Any] | None = None

    @property
    def ready(self) -> bool:
        """Return true if the mailbox has all expected results."""
        return (
            self.num_results >= self.expected_num_results
            and self.num_results != 0
        )

    @property
    def has_task_waiting(self) -> bool:
        """Return True if a task is waiting on the result of this box."""
        return self.dest_addr is not None

    @staticmethod
    def new_mailbox(num_results: int | None = None) -> WorkerMailbox:
        """
        Create a new mailbox with `num_results` slots.

        If `num_results` is None (by default), then the mailbox will only have
        one slot and expect one result.
        """
        if num_results is None:
            return WorkerMailbox(True, 1)

        return WorkerMailbox(False, num_results, [None] * num_results)

    def get_new_results(self) -> list[tuple[int, Any]]:
        """Return and reset the results that have come in since previous
        call."""
        assert self.fresh_results is not None
        out = self.fresh_results
        self.fresh_results = []
        return out

    def deposit_result(self, result: RuntimeResult) -> None:
        """Store the result in the mailbox."""
        self.num_results += 1
        slot_id = result.return_address.mailbox_slot

        # Record as fresh result
        if self.fresh_results is None:
            self.fresh_results = []
        self.fresh_results.append((slot_id, result.result))

        if self.expecting_single_result:
            self.result = result.result
        else:
            self.result[slot_id] = result.result


class Worker:
    """
    BQSKit Runtime's Worker.

    BQSKit Runtime utilizes a dual-threaded worker to accept, execute,
    pause, spawn, resume, and complete tasks in a custom event loop built
    with python's async await mechanisms. Each worker receives and sends
    tasks and results to the greater system through a single duplex
    connection with a runtime server or manager. One thread performs
    work and sends outgoing messages, while the other thread handles
    incoming messages.

    At start-up, the worker receives an ID and waits for its first task.
    An executing task may use the `submit` and `map` methods to spawn child
    tasks and distribute them across the whole system. Once completed,
    those child tasks will have their results shipped back to the worker
    who created them. When a task awaits a child task, it is removed from
    the ready queue until the desired results come in.

    All created log records are shipped back to the client's process.
    This feature ensures compatibility with applications like jupyter
    that only print messages from the client process's stdout. Additionally,
    it allows BQSKit users seamless integration with the standard python
    logging module. From a user's perspective, they can configure any
    standard python logger from their process like usual and have the
    entire system honor that configuration. Lastly, we do support an
    additional logging option for maximum task depth. Tasks with more
    ancestors than the maximum logging depth will not produce any logs.

    Workers handle python errors by capturing and bubbling them up. For
    operating system-level crashes and errors -- such as seg-faults in
    client code -- the worker will attempt to print a stack trace and
    initiate a system-wide shutdown. However, note that these issues
    can go unhandled and cause the runtime to deadlock or crash.

    Workers perform very minimal scheduling of tasks. Newly created tasks
    are directly forwarded upwards to a manager or server, which in turn
    assigns them to workers. New tasks from above are commonly received
    in batches. In this case, all but one task from a batch is delayed.
    Delayed tasks are moved (in LIFO order) to the ready queue if a worker
    has no other work to complete. This mechanism encourages completing
    deeply-nested tasks first and prevents flooding the system with active
    tasks, which usually require much more memory than delayed ones.
    """

    def __init__(self, id: int, conn: Connection) -> None:
        """
        Initialize a worker with no tasks.

        Args:
            id (int): This worker's id.

            conn (Connection): This worker's duplex channel to a manager
                or a server.
        """
        self._id = id
        self._conn = conn

        self._tasks: dict[RuntimeAddress, RuntimeTask] = {}
        """Tracks all started, unfinished tasks on this worker."""

        self._delayed_tasks: list[RuntimeTask] = []
        self._tasklog_fh: Any = None
        """
        Store all delayed tasks in LIFO order.

        Delayed tasks have no context and are stored (more-or-less) as a
        function pointer together with the arguments. When it gets started, it
        consumes much more memory, so we delay the task start until necessary
        (at no cost)
        """

        self._ready_task_ids: PriorityQueue[
            tuple[int, int, RuntimeAddress]
        ] = PriorityQueue()
        """Tasks queued up for execution, ordered by service class.

        Entries are (priority, arrival, address). The arrival counter is what
        makes this a strict generalisation rather than a behaviour change: with
        every task at the same class -- which is every caller that does not ask
        for otherwise -- ordering by (class, arrival) IS the FIFO this replaces.
        It also keeps the tuples totally ordered, so PriorityQueue never has to
        compare two RuntimeAddress objects.
        """

        self._ready_seq = 0
        """Monotonic arrival counter; see _ready_task_ids."""

        self._cancelled_task_ids: set[RuntimeAddress] = set()
        """To ensure newly-received cancelled tasks are never started."""

        self._active_task: RuntimeTask | None = None
        """The currently executing task if one is running."""

        self._running = True
        """Controls if the event loop is running."""

        self._mailboxes: dict[int, WorkerMailbox] = {}
        """Map from mailbox ids to worker mailboxes."""

        self._mailbox_counter = 0
        """This count ensures every mailbox has a unique id."""

        self._cache: dict[str, Any] = {}
        """Local worker cache."""

        self.most_recent_read_submit: RuntimeAddress | None = None
        """Tracks the most recently processed submit message from above."""

        self.read_receipt_mutex = Lock()
        """
        A lock to ensure waiting messages's read receipt is correct.

        This lock enforces atomic update of `most_recent_read_submit` and
        task addition/enqueueing. This is necessary to ensure that the
        idle status is always correct.
        """

        # Send out every client emitted log message upstream
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = old_factory(*args, **kwargs)
            active_task = self._active_task
            if not record.name.startswith('bqskit.runtime'):
                if active_task is not None:
                    lvl = active_task.logging_level
                    if lvl is None or lvl <= record.levelno:
                        if lvl <= logging.DEBUG:
                            record.msg += f' [wid={self._id}'
                            items = active_task.log_context.items()
                            if len(items) > 0:
                                record.msg += ', '
                            con_str = ', '.join(f'{k}={v}' for k, v in items)
                            record.msg += con_str
                            record.msg += ']'
                        tid = active_task.comp_task_id
                        try:
                            serial = pickle.dumps(record)
                        except (pickle.PicklingError, TypeError):
                            serial = pickle.dumps((
                                record.name,
                                record.levelno,
                                record.getMessage(),
                            ))
                        self._conn.send((RuntimeMessage.LOG, (tid, serial)))
            return record

        logging.setLogRecordFactory(record_factory)

        # Start incoming thread
        self.incoming_thread = Thread(target=self.recv_incoming)
        self.incoming_thread.daemon = True
        self.incoming_thread.start()
        _logger.debug('Started incoming thread.')

        # Communicate that this worker is ready
        self._conn.send((RuntimeMessage.STARTED, self._id))

    def _loop(self) -> None:
        """Main worker event loop."""
        while self._running:
            try:
                self._try_step_next_ready_task()
            except Exception:
                self._running = False
                exc_info = sys.exc_info()
                error_str = ''.join(traceback.format_exception(*exc_info))
                _logger.error(error_str)
                try:
                    self._conn.send((RuntimeMessage.ERROR, error_str))
                except Exception:
                    pass

    def recv_incoming(self) -> None:
        """Continuously receive all incoming messages."""
        while self._running:
            # Receive message
            try:
                msg, payload = self._conn.recv()
            except Exception:
                _logger.debug('Crashed due to lost connection')
                if sys.platform == 'win32':
                    os.kill(os.getpid(), 9)
                else:
                    os.kill(os.getpid(), signal.SIGKILL)
                exit()

            _logger.debug(f'Received message {msg.name}.')
            _logger.log(1, f'Payload: {payload}')

            # Process message
            if msg == RuntimeMessage.SHUTDOWN:
                if sys.platform == 'win32':
                    os.kill(os.getpid(), 9)
                else:
                    os.kill(os.getpid(), signal.SIGKILL)

            elif msg == RuntimeMessage.SUBMIT:
                self.read_receipt_mutex.acquire()
                task = cast(RuntimeTask, payload)
                self.most_recent_read_submit = task.unique_id
                self._add_task(task)
                self.read_receipt_mutex.release()

            elif msg == RuntimeMessage.SUBMIT_BATCH:
                self.read_receipt_mutex.acquire()
                tasks = cast(List[RuntimeTask], payload)
                self.most_recent_read_submit = tasks[0].unique_id
                self._add_task(tasks.pop())  # Submit one task
                self._delayed_tasks.extend(tasks)  # Delay rest
                self.read_receipt_mutex.release()

            elif msg == RuntimeMessage.RESULT:
                result = cast(RuntimeResult, payload)
                self._handle_result(result)

            elif msg == RuntimeMessage.CANCEL:
                addr = cast(RuntimeAddress, payload)
                self._handle_cancel(addr)
                # TODO: preempt?

            elif msg == RuntimeMessage.COMMUNICATE:
                addrs, msg = cast(tuple[list[RuntimeAddress], Any], payload)
                self._handle_communicate(addrs, msg)

            elif msg == RuntimeMessage.IMPORTPATH:
                paths = cast(List[str], payload)
                for path in paths:
                    if path not in sys.path:
                        sys.path.append(path)

            elif msg == RuntimeMessage.OCCUPANCY:
                # This node's (idle, total) worker counts, throttled by the
                # manager. Parked in the worker cache because that is the only
                # channel a running pass can already read -- get_cache() is
                # part of RuntimeHandle, the occupancy is not.
                #
                # A pass that finds no key must behave as it did before: absent
                # is not "zero idle", it is "unknown", and treating unknown as
                # zero would silently disable every mechanism that sizes itself
                # from this number.
                # (idle_workers, total_workers, free_cores). The third field
                # is what a pass should size itself from: unassigned workers
                # and un-busy cores differ threefold in practice. Older
                # 2-tuples are accepted so a mixed-version node cannot crash a
                # worker, falling back to the worker count.
                if len(payload) >= 3:
                    idle, total, free = cast(
                        Tuple[int, int, int], payload[:3],
                    )
                else:
                    idle, total = cast(Tuple[int, int], payload)
                    free = idle
                self._cache[_OCCUPANCY_KEY] = (
                    idle, total, free, time.monotonic(),
                )

    def _add_task(self, task: RuntimeTask) -> None:
        """Start a task and add it to the loop."""
        self._tasks[task.return_address] = task
        task.start()
        self._enqueue_ready(task.return_address, task.priority)

    def _handle_result(self, result: RuntimeResult) -> None:
        """Insert result into appropriate mailbox and wake waiting task."""
        assert result.return_address.worker_id == self._id

        mailbox_id = result.return_address.mailbox_index
        if mailbox_id not in self._mailboxes:
            # If the mailbox has been dropped due to a cancel, ignore result
            return

        box = self._mailboxes[mailbox_id]
        box.deposit_result(result)

        if box.has_task_waiting:
            assert box.dest_addr is not None
            task = self._tasks[box.dest_addr]

            if task.wake_on_next or box.ready:
                # print(f'Worker {self._id} is waking task
                # {task.return_address}, with {task.wake_on_next=},
                # {box.ready=}')
                # Wake it, at the priority the task itself carries: a
                # speculative task that blocked and became runnable again is
                # still speculative, and re-admitting it as critical would let
                # it overtake the work it was meant to stay behind.
                woken = self._tasks.get(box.dest_addr)
                self._enqueue_ready(
                    box.dest_addr,
                    woken.priority if woken is not None else PRIORITY_CRITICAL,
                )
                box.dest_addr = None  # Prevent double wake

    def _handle_cancel(self, addr: RuntimeAddress) -> None:
        """
        Remove `addr` and its children tasks from this worker.

        Notes:
            Since `self._ready_task_ids' is a queue, it is more efficient
            to discard cancelled tasks when popping from it. Therefore, we
            do not do anything with `self._ready_task_ids` here.

            We also must make sure to call the `cancel` function of the
            tasks to make sure their coroutines are cleaned up.

            Also, we also don't need to send out cancel messages for
            cancelled children tasks since other workers can evaluate that
            for themselves using breadcrumbs and the original `addr` cancel
            message.
        """
        # TODO: Send update message?
        self._cancelled_task_ids.add(addr)

        # Remove all tasks that are children of `addr` from initialized tasks
        for key, task in self._tasks.items():
            if task.is_descendant_of(addr):
                task.cancel()
                for mailbox_id in self._tasks[key].owned_mailboxes:
                    self._mailboxes.pop(mailbox_id)
        self._tasks = {
            a: t for a, t in self._tasks.items()
            if not t.is_descendant_of(addr)
        }

        # Remove all tasks that are children of `addr` from delayed tasks
        self._delayed_tasks = [
            t for t in self._delayed_tasks
            if not t.is_descendant_of(addr)
        ]

    def _handle_communicate(
        self,
        addrs: list[RuntimeAddress],
        msg: Any,
    ) -> None:
        for task_addr in addrs:
            if task_addr not in self._tasks:
                continue

            self._tasks[task_addr].msg_buffer.append(msg)

    def _head_priority(self) -> int | None:
        """Service class of the most urgent ready task, or None if empty."""
        try:
            return self._ready_task_ids.queue[0][0]
        except IndexError:
            return None

    def _should_promote_delayed(self) -> bool:
        """Whether a delayed task should be started now.

        The original condition was `ready queue is empty`, which was correct
        only while every task shared one service class: a FIFO drains, so the
        queue does empty and delayed work does start.

        With service classes it is a starvation bug, and one that inverts the
        whole design. A batch puts ONE task in the ready queue and delays the
        rest, so a batch of CRITICAL tasks leaves N-1 of them in
        `_delayed_tasks` -- where, if speculation is sitting in the ready queue
        keeping it non-empty, they never start. Speculation would then block
        the critical path by OCCUPYING the queue rather than by being ahead of
        it, which is the opposite of what the priority is for.

        The condition that holds in both worlds: promote when the ready queue
        has nothing at least as urgent as the most urgent delayed task.
        """
        head = self._head_priority()
        if head is None:
            return True
        return head > min(t.priority for t in self._delayed_tasks)

    def _pop_best_delayed(self) -> RuntimeTask:
        """Take the most urgent delayed task, LIFO within its class.

        LIFO is load-bearing and predates this change: it completes
        deeply-nested tasks first, which keeps the number of STARTED tasks --
        and therefore memory -- down. Priority orders the classes; LIFO still
        orders within one.
        """
        best = min(t.priority for t in self._delayed_tasks)
        for i in range(len(self._delayed_tasks) - 1, -1, -1):
            if self._delayed_tasks[i].priority == best:
                return self._delayed_tasks.pop(i)
        raise AssertionError('non-empty delayed list has no minimum')

    def _enqueue_ready(self, addr: RuntimeAddress, priority: int) -> None:
        """Admit an address to the ready queue in its service class."""
        self._ready_seq += 1
        self._ready_task_ids.put((priority, self._ready_seq, addr))

    def _get_next_ready_task(self) -> RuntimeTask | None:
        """Return the next ready task if one exists, otherwise block."""
        while True:
            if self._delayed_tasks and self._should_promote_delayed():
                self._add_task(self._pop_best_delayed())
                continue

            # Critical section
            # Attempt to get a ready task. If none are available, message
            # the manager/server with a waiting message letting them
            # know the worker is idle. This needs to be atomic to prevent
            # the self.more_recent_read_submit from being updated after
            # catching the Empty exception, but before forming the payload.
            self.read_receipt_mutex.acquire()
            try:
                _prio, _seq, addr = self._ready_task_ids.get_nowait()

            except Empty:
                payload = (1, self.most_recent_read_submit)
                self._conn.send((RuntimeMessage.WAITING, payload))
                self.read_receipt_mutex.release()
                # Block for new message. Can release lock here since the
                # the `self.most_recent_read_submit` has been used.
                _prio, _seq, addr = self._ready_task_ids.get()

            else:
                self.read_receipt_mutex.release()

            # Handle a shutdown request that occured while waiting
            if not self._running:
                return None

            if addr in self._cancelled_task_ids or addr not in self._tasks:
                # When a task is cancelled on the worker it is not removed
                # from the ready queue because it is much cheaper to just
                # discard cancelled tasks as they come out.
                continue

            task = self._tasks[addr]

            if any(bcb in self._cancelled_task_ids for bcb in task.breadcrumbs):
                # If any of the selected tasks ancestor tasks are cancelled
                # then discard this one too. Each breadcrumb (bcb) is a
                # task address (unique system-wide task id) of an ancestor
                # task.
                # TODO: do I need to manually remove addr from self._tasks?
                continue

            return task

    def _try_step_next_ready_task(self) -> None:
        """Select a task to run, and advance it one step."""
        task = self._get_next_ready_task()

        if task is None:
            return

        try:
            self._active_task = task

            # One record per step: who ran, for how long, on whose work, at
            # which service class, and under which block. Without this the
            # only per-core signal is /proc/stat, which says a core was busy
            # but never says with what -- so "did the mechanism work" can only
            # be answered in aggregate, never per core over time.
            _t0 = time.monotonic() if _TASKLOG_DIR else 0.0

            # Perform a step of the task and get the future it awaits on
            future = task.step(self._get_desired_result(task))

            self._process_await(task, future)

        except StopIteration as e:
            self._process_task_completion(task, e.value)

        except Exception:
            assert self._active_task is not None  # for type checker

            # Bubble up errors
            exc_info = sys.exc_info()
            error_str = ''.join(traceback.format_exception(*exc_info))
            error_payload = (self._active_task.comp_task_id, error_str)
            self._conn.send((RuntimeMessage.ERROR, error_payload))

        finally:
            # In `finally`, not after `task.step`. A step that COMPLETES its
            # task raises StopIteration, which jumps straight past the old call
            # site -- so the log only ever held steps that awaited. A
            # speculative instantiate runs to completion in one step, so not one
            # of them was ever recorded: `pri` came back 0 for every row even on
            # the speculation arm, and `fn` held only the two long-lived
            # coroutines. The figure drawn from it had four legend entries and
            # one colour.
            if _TASKLOG_DIR:
                self._tasklog(task, _t0, time.monotonic())
            self._active_task = None

    def _tasklog(self, task: RuntimeTask, t0: float, t1: float) -> None:
        """Append one step record. Handle opened once, buffered, never per call.

        Four categories fall out of two fields the runtime already carries:
        `priority` (0 critical / 10 speculative) and `return_address.worker_id`
        (the worker whose work this is). Own vs borrowed is exactly whether
        that id is this worker's.

        Block identity is the FIRST breadcrumb with a real worker id, not
        `breadcrumbs[0]`. The outermost ancestor is the client, whose worker id
        is -1, so `breadcrumbs[0]` is the same string for every task in the run
        -- the first version collapsed a 20-block circuit to one block and made
        the boundaries in the figure invisible.
        """
        # Line-buffered, not block-buffered. Workers are stopped with SIGTERM
        # and never run an exit handler, so a 64 KB buffer is simply lost: the
        # first attempt produced 0 records from 39 worker files. One write
        # syscall per step costs about a microsecond against a step that runs
        # actual synthesis; the earlier disaster was opening a file per event,
        # not writing to an open one.
        if self._tasklog_fh is None:
            os.makedirs(_TASKLOG_DIR, exist_ok=True)
            self._tasklog_fh = open(
                os.path.join(_TASKLOG_DIR, f'task_{self._id}.jsonl'),
                'a', buffering=1,   # line-buffered: see below
            )
        owner = task.return_address.worker_id
        block = '-'
        for crumb in task.breadcrumbs:
            if crumb.worker_id != -1:
                block = (
                    f'{crumb.worker_id}:{crumb.mailbox_index}'
                    f':{crumb.mailbox_slot}'
                )
                break
        self._tasklog_fh.write(
            '{"w":%d,"t0":%.6f,"t1":%.6f,"pri":%d,"mine":%d,"blk":"%s",'
            '"fn":"%s","dep":%d}\n'
            % (self._id, t0, t1, task.priority, int(owner == self._id),
               block, task._name, len(task.breadcrumbs))
        )

    def _process_await(self, task: RuntimeTask, future: RuntimeFuture) -> None:
        """Process a task's await request."""
        if not isinstance(future, RuntimeFuture):
            raise RuntimeError('Can only await on a BQSKit RuntimeFuture.')

        if future.mailbox_id not in self._mailboxes:
            raise RuntimeError('Cannot await on a canceled task.')

        box = self._mailboxes[future.mailbox_id]

        # Let the mailbox know this task is waiting
        box.dest_addr = task.return_address
        task.desired_box_id = future.mailbox_id

        # if future._next_flag:
        #     # Set from Worker.next, implies the task wants the next result
        #     # if box.ready:
        #     #     m = 'Cannot wait for next results on a complete task.'
        #     #     raise RuntimeError(m)
        #     task.wake_on_next = True
        task.wake_on_next = future._next_flag
        # print(f'Worker {self._id} is waiting on task
        # {task.return_address}, with {task.wake_on_next=}')

        if box.ready:
            self._enqueue_ready(task.return_address, task.priority)

    def _process_task_completion(self, task: RuntimeTask, result: Any) -> None:
        """Package and send out task result."""
        assert task is self._active_task
        packaged_result = RuntimeResult(task.return_address, result, self._id)

        if task.return_address not in self._tasks:
            # print(f'Task was cancelled: {task.return_address},
            # {task.fnargs[0].__name__}')
            return

        if task.return_address.worker_id == self._id:
            self._handle_result(packaged_result)
            self._conn.send((RuntimeMessage.UPDATE, -1))
            # Let manager know this worker has one less task
            # without sending a result
        else:
            self._conn.send((RuntimeMessage.RESULT, packaged_result))

        # Remove task
        self._tasks.pop(task.return_address, None)

        # Cancel any open tasks.
        #
        # Iterate over a COPY. cancel() removes from owned_mailboxes, and
        # mutating the list being walked makes the loop skip the next entry.
        for mailbox_id in list(self._active_task.owned_mailboxes):
            # If task is complete, simply discard result
            if mailbox_id in self._mailboxes:
                if self._mailboxes[mailbox_id].ready:
                    self._mailboxes.pop(mailbox_id)
                    # Drop the ownership record too. Popping the mailbox while
                    # leaving its id here is what produced the dangling id that
                    # a later cancel() then indexed, killing the worker.
                    self._active_task.owned_mailboxes.remove(mailbox_id)
                    continue

            # Otherwise send a cancel message
            self.cancel(RuntimeFuture(mailbox_id))

    def _get_desired_result(self, task: RuntimeTask) -> Any:
        """Retrieve the task's desired result from the mailboxes."""
        if task.desired_box_id is None:
            return None

        box = self._mailboxes[task.desired_box_id]

        if task.wake_on_next:
            fresh_results = box.get_new_results()
            # assert len(fresh_results) > 0
            return fresh_results

        assert box.ready
        task.owned_mailboxes.remove(task.desired_box_id)
        return self._mailboxes.pop(task.desired_box_id).result

    def _get_new_mailbox_id(self) -> int:
        """Return a new unique mailbox id."""
        new_id = self._mailbox_counter
        self._mailbox_counter += 1
        return new_id

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        task_name: str | None = None,
        log_context: dict[str, str] = {},
        task_priority: int = PRIORITY_CRITICAL,
        cost_hint: float = 0.0,
        **kwargs: Any,
    ) -> RuntimeFuture:
        """Submit `fn` as a task to the runtime.

        `task_priority` selects the worker's service class. Named with the
        `task_` prefix because everything else in **kwargs is forwarded to
        `fn`, and a bare `priority` would collide with any callee that happens
        to take one.
        """
        assert self._active_task is not None

        if task_name is not None and not isinstance(task_name, str):
            raise RuntimeError('task_name must be a string.')

        if not isinstance(log_context, dict):
            raise RuntimeError('log_context must be a dictionary.')

        for k, v in log_context.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise RuntimeError(
                    'log_context must be a map from strings to strings.',
                )

        # Group fnargs together
        fnarg = (fn, args, kwargs)

        # Create a new mailbox
        mailbox_id = self._get_new_mailbox_id()
        self._mailboxes[mailbox_id] = WorkerMailbox.new_mailbox()
        self._active_task.owned_mailboxes.append(mailbox_id)

        # Create the task
        task = RuntimeTask(
            fnarg,
            RuntimeAddress(self._id, mailbox_id, 0),
            self._active_task.comp_task_id,
            self._active_task.breadcrumbs
            + (self._active_task.return_address,),
            self._active_task.logging_level,
            self._active_task.max_logging_depth,
            task_name,
            {**self._active_task.log_context, **log_context},
            task_priority,
            cost_hint,
        )

        # Submit the task (on the next cycle)
        self._conn.send((RuntimeMessage.SUBMIT, task))

        # Return future pointing to the mailbox
        return RuntimeFuture(mailbox_id)

    def map(
        self,
        fn: Callable[..., Any],
        *args: Any,
        task_name: Sequence[str | None] | str | None = None,
        log_context: Sequence[dict[str, str]] | dict[str, str] = {},
        task_priority: Sequence[int] | int = PRIORITY_CRITICAL,
        cost_hints: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> RuntimeFuture:
        """Map `fn` over the input arguments distributed across the runtime.

        `task_priority` selects the worker's service class. It may be a single
        value for the whole batch, or one value per task -- the same
        scalar-or-sequence shape `task_name`, `log_context` and `cost_hints`
        already use. Per-task is what lets a caller mix urgencies in ONE map,
        which matters because this runtime has no wait-any across futures: a
        caller that needs some of its tasks demoted cannot simply issue a
        second, lower-priority map and harvest it opportunistically.

        NO ESCALATION BY SPAWNING. A task's effective priority is the LEAST
        urgent of what it asked for and what its parent already has, so a
        demoted subtree stays demoted. Without this the demotion is cosmetic:
        ForEachBlockPass could dispatch a redundant block copy at low
        priority, and that copy's own LEAP would immediately dispatch its
        instantiate tasks at PRIORITY_CRITICAL and compete with real work
        anyway. See `RuntimeTask.priority`; the `task_` prefix keeps it out of
        the **kwargs that are forwarded to `fn`.
        """
        assert self._active_task is not None

        if task_name is None or isinstance(task_name, str):
            task_name = [task_name] * len(args[0])

        if len(task_name) != len(args[0]):
            raise RuntimeError(
                'task_name must be a string or a list of strings equal'
                'in length to the number of tasks.',
            )

        if isinstance(log_context, dict):
            log_context = [log_context] * len(args[0])

        if len(log_context) != len(args[0]):
            raise RuntimeError(
                'log_context must be a dictionary or a list of dictionaries'
                ' equal in length to the number of tasks.',
            )

        for context in log_context:
            for k, v in context.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    raise RuntimeError(
                        'log_context must be a map from strings to strings.',
                    )

        # Group fnargs together
        fnargs = []
        if len(args) == 1:
            for arg in args[0]:
                fnargs.append((fn, (arg,), kwargs))

        else:
            for subargs in zip(*args):
                fnargs.append((fn, subargs, kwargs))

        if len(fnargs) == 0:
            raise RuntimeError('Unable to map 0 tasks.')

        if isinstance(task_priority, int):
            task_priority = [task_priority] * len(fnargs)
        elif len(task_priority) != len(fnargs):
            raise ValueError(
                f'task_priority has length {len(task_priority)}, but '
                f'{len(fnargs)} tasks were created.',
            )
        # Larger number == less urgent, so max() is the demotion.
        _parent_priority = self._active_task.priority
        task_priority = [max(p, _parent_priority) for p in task_priority]

        if cost_hints is None:
            cost_hints = [0.0] * len(fnargs)
        elif len(cost_hints) != len(fnargs):
            raise ValueError(
                f'cost_hints has length {len(cost_hints)}, but '
                f'{len(fnargs)} tasks were created.',
            )

        # Create a new mailbox
        mailbox_id = self._get_new_mailbox_id()
        self._mailboxes[mailbox_id] = WorkerMailbox.new_mailbox(len(fnargs))
        self._active_task.owned_mailboxes.append(mailbox_id)

        # Create the tasks
        breadcrumbs = self._active_task.breadcrumbs
        breadcrumbs += (self._active_task.return_address,)
        tasks = [
            RuntimeTask(
                fnarg,
                RuntimeAddress(self._id, mailbox_id, i),
                self._active_task.comp_task_id,
                breadcrumbs,
                self._active_task.logging_level,
                self._active_task.max_logging_depth,
                task_name[i],
                {**self._active_task.log_context, **log_context[i]},
                task_priority[i],
                cost_hints[i],
            )
            for i, fnarg in enumerate(fnargs)
        ]

        # Submit the tasks
        self._conn.send((RuntimeMessage.SUBMIT_BATCH, tasks))

        # Return future pointing to the mailbox
        return RuntimeFuture(mailbox_id)

    def communicate(self, future: RuntimeFuture, msg: Any) -> None:
        """Send a message to the task associated with `future`."""
        assert self._active_task is not None
        assert future.mailbox_id in self._mailboxes

        num_slots = self._mailboxes[future.mailbox_id].expected_num_results
        addrs = [
            RuntimeAddress(self._id, future.mailbox_id, slot_id)
            for slot_id in range(num_slots)
        ]
        self._conn.send((RuntimeMessage.COMMUNICATE, (addrs, msg)))

    def get_messages(self) -> list[Any]:
        """Return all messages received by the worker for this task."""
        assert self._active_task is not None
        x = self._active_task.msg_buffer
        self._active_task.msg_buffer = []
        return x

    def cancel(self, future: RuntimeFuture) -> None:
        """Cancel all tasks associated with `future`.

        Tolerates a mailbox that is already gone. Indexing it unguarded
        raised KeyError inside the worker loop, which killed the worker and
        reached the client as "Server connection unexpectedly closed" --
        6 of 48 benchmark cells on 2026-08-19, in both arms.
        """
        assert self._active_task is not None
        mailbox = self._mailboxes.pop(future.mailbox_id, None)
        if future.mailbox_id in self._active_task.owned_mailboxes:
            self._active_task.owned_mailboxes.remove(future.mailbox_id)
        if mailbox is None:
            return
        num_slots = mailbox.expected_num_results
        addrs = [
            RuntimeAddress(self._id, future.mailbox_id, slot_id)
            for slot_id in range(num_slots)
        ]
        for addr in addrs:
            self._conn.send((RuntimeMessage.CANCEL, addr))

    def get_cache(self) -> dict[str, Any]:
        """
        Retrieve worker's local cache.

        Returns:
            (dict[str, Any]): The worker's local cache. This cache can be
                used to store large or unserializable objects within a
                worker process' memory. Passes on the same worker that use
                the same object can load the object from this cache. If
                there are multiple workers, those workers will load their
                own copies of the object into their own cache.
        """
        return self._cache

    async def next(self, future: RuntimeFuture) -> list[tuple[int, Any]]:
        """
        Wait for and return the next batch of results from a map task.

        Returns:
            (list[tuple[int, Any]]): A list of the results that arrived
                since the last time this was called. On the first call,
                all results that have arrived since the task started are
                returned. Each result is paired with the index of its
                arguments in the original map call.
        """
        # if future._done:
        if future.mailbox_id not in self._mailboxes:
            raise RuntimeError('Cannot wait on an already completed result.')

        future._next_flag = True
        next_result_batch = await future
        future._next_flag = False
        return next_result_batch


# Global variable containing reference to this process's worker object.
_worker = None


def start_worker(
    w_id: int | None,
    port: int,
    cpu: int | None = None,
    logging_level: int = logging.WARNING,
    num_blas_threads: int = 1,
    log_client: bool = False,
) -> None:
    """Start this process's worker."""
    if w_id is not None:
        # Ignore interrupt signals on workers, boss will handle it for us
        # If w_id is None, then we are being spawned separately.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # TODO: check what needs to be done on win

    # Set number of BLAS threads
    set_blas_thread_counts(num_blas_threads)

    # Enforce no default logging
    logging.lastResort = logging.NullHandler()
    logging.getLogger().handlers.clear()

    # Pin worker to cpu
    if cpu is not None:
        if sys.platform == 'win32':
            raise RuntimeError('Cannot pin worker to cpu on windows.')
        os.sched_setaffinity(0, [cpu])

    # Connect to manager
    max_retries = 7
    wait_time = .1
    conn: Connection | None = None
    family = 'AF_INET' if sys.platform == 'win32' else None
    for _ in range(max_retries):
        try:
            conn = Client(('localhost', port), family)
        except (ConnectionRefusedError, TimeoutError):
            time.sleep(wait_time)
            wait_time *= 2
        else:
            break

    if conn is None:
        raise RuntimeError('Unable to establish connection with manager.')

    # If id isn't provided, wait for assignment
    if w_id is None:
        msg, w_id = conn.recv()
        assert isinstance(w_id, int)
        assert msg == RuntimeMessage.STARTED

    # Set up runtime logging
    if not log_client:
        _runtime_logger = logging.getLogger('bqskit.runtime')
    else:
        _runtime_logger = logging.getLogger()
    _runtime_logger.propagate = False
    _runtime_logger.setLevel(logging_level)
    _handler = logging.StreamHandler()
    _handler.setLevel(0)
    _fmt_header = '%(asctime)s.%(msecs)03d - %(levelname)-8s |'
    _fmt_message = f' [wid={w_id}]: %(message)s'
    _fmt = _fmt_header + _fmt_message
    _formatter = logging.Formatter(_fmt, '%H:%M:%S')
    _handler.setFormatter(_formatter)
    _runtime_logger.addHandler(_handler)

    # Build and start worker
    global _worker
    _worker = Worker(w_id, conn)
    _worker._loop()


def get_worker() -> Worker:
    """Return a handle on this process's worker."""
    if _worker is None:
        raise RuntimeError('Worker has not been started.')
    return _worker


def _check_positive(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            '%s is an invalid positive int value' % value,
        )
    return ivalue


def start_worker_rank() -> None:
    """Entry point for spawning a rank of runtime worker processes."""
    parser = argparse.ArgumentParser(
        prog='bqskit-worker',
        description='Launch a rank of BQSKit runtime worker processes.',
    )
    parser.add_argument(
        'num_workers',
        type=_check_positive,
        help='The number of workers to spawn.',
    )
    parser.add_argument(
        '--cpus', '-c',
        nargs='+',
        type=int,
        help='Either one number or a list of numbers equal in length to the'
        ' number of workers. The workers will be pinned to specified logical'
        ' cpus. If a single-number is given, then all cpu indices are'
        ' enumerated starting at that number.',
    )
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=default_worker_port,
        help='The port the workers will try to connect to a manager on.',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Enable logging of increasing verbosity, either -v, -vv, or -vvv.',
    )
    parser.add_argument(
        '-l', '--log-client',
        action='store_true',
        help='Log messages from the client process.',
    )
    parser.add_argument(
        '-t', '--num_blas_threads',
        type=int,
        default=1,
        help='The number of threads to use in BLAS libraries.',
    )
    args = parser.parse_args()

    if args.cpus is not None:
        if len(args.cpus) == 1:
            cpus = [args.cpus[0] + i for i in range(args.num_workers)]

        elif len(args.cpus) == args.num_workers:
            cpus = args.cpus

        else:
            raise RuntimeError(
                'The specified logical cpus are invalid. Expected either'
                ' a single number or a list of numbers equal in length to'
                ' the number of workers.',
            )

    else:
        cpus = [None for _ in range(args.num_workers)]

    logging_level = [30, 20, 10, 1][min(args.verbose, 3)]

    if args.log_client and logging_level > 10:
        raise RuntimeError('Cannot log client messages without at least -vv.')

    # Spawn worker process
    procs = []
    for cpu in cpus:
        pargs = (
            None,
            args.port,
            cpu,
            logging_level,
            args.num_blas_threads,
            args.log_client,
        )
        procs.append(Process(target=start_worker, args=pargs))
        procs[-1].start()

    # Join them
    for proc in procs:
        proc.join()
