"""This module implements the NodeBase abstract class."""
from __future__ import annotations

import abc
import functools
import heapq
import logging
import os
import random
import selectors
import signal
import socket
import sys
import time
import traceback
from multiprocessing import Process
from multiprocessing.connection import Client
from multiprocessing.connection import Connection
from multiprocessing.connection import Listener
from queue import Queue
from threading import Thread
from types import FrameType
from typing import Any
from typing import cast
from typing import Sequence

from bqskit.runtime import default_manager_port
from bqskit.runtime import default_worker_port
from bqskit.runtime import set_blas_thread_counts
from bqskit.runtime.address import RuntimeAddress
from bqskit.runtime.direction import MessageDirection
from bqskit.runtime.message import RuntimeMessage
from bqskit.runtime.result import RuntimeResult
from bqskit.runtime.task import RuntimeTask
from bqskit.runtime.worker import start_worker


_logger = logging.getLogger(__name__)

# Seconds between occupancy broadcasts. The only consumer decides once per A*
# round -- hundreds of milliseconds -- so a tenth of a second is already finer
# than anything that reads it, while the idle count itself moves thousands of
# times a second.
_OCCUPANCY_INTERVAL = 0.1

_PIN_WORKERS = os.environ.get('BQSKIT_PIN_WORKERS', '0') != '0'

# Drains between pool statistics lines. Frequent enough to survive a SIGTERM
# that never runs an exit handler, rare enough not to flood the log.
_POOL_STAT_EVERY = int(os.environ.get('BQSKIT_POOL_STAT_EVERY', '2000'))


def _node_busy_fraction(prev: tuple[float, float] | None) -> tuple[
    float | None, tuple[float, float] | None,
]:
    """Busy fraction of this node's cores since `prev`, and a new snapshot.

    `num_idle_workers` counts workers with nothing ASSIGNED, which is not the
    same as cores doing nothing, and the gap is large: measured on adder_8 at
    msz=4, LEAP saw 15.5 idle workers out of 112 while 41% of the cores -- about
    46 of them -- were not computing. Sizing speculation from the worker count
    therefore under-committed the machine threefold and pinned occupancy at 59%.

    Reading /proc/stat costs one file read per broadcast and measures the thing
    the decision is actually about. On a node shared with another job it
    under-reports free capacity, which is the safe direction.
    """
    try:
        with open('/proc/stat', encoding='ascii') as fh:
            parts = fh.readline().split()
    except OSError:
        return None, prev
    if not parts or parts[0] != 'cpu':
        return None, prev
    v = [float(x) for x in parts[1:9]]
    while len(v) < 8:
        v.append(0.0)
    user, nice, system, idle, iowait, irq, softirq, steal = v
    # iowait is not work; counting it as busy would make a starved machine look
    # fed, which is the exact error this whole measurement exists to avoid.
    busy = user + nice + system + irq + softirq
    total = busy + idle + iowait + steal
    now = (busy, total)
    if prev is None:
        return None, now
    dbusy, dtotal = busy - prev[0], total - prev[1]
    if dtotal <= 0 or dbusy < 0:
        return None, now
    return min(1.0, dbusy / dtotal), now


class RuntimeEmployee:
    """Data structure for a boss's view of an employee."""

    def __init__(
        self,
        id: int,
        conn: Connection,
        total_workers: int,
        process: Process | None = None,
        is_manager: bool = False,
    ) -> None:
        """Construct an employee with all resources idle."""

        self.id = id
        """
        The ID of the employee.

        If this is a worker, then their unique worker id. If this is a manager,
        then their local id.
        """

        self.conn: Connection = conn
        self.total_workers = total_workers
        self.process = process
        self.num_tasks = 0
        self.num_idle_workers = total_workers
        self.is_manager = is_manager

        self.submit_cache: list[tuple[RuntimeAddress, int]] = []
        """
        Tracks recently submitted tasks by id and count.

        This is used to adjust the idle worker count when the employee sends a
        waiting message.
        """

    def initiate_shutdown(self) -> None:
        """Instruct employee to shutdown."""
        try:
            self.conn.send((RuntimeMessage.SHUTDOWN, None))
        except Exception:
            pass

    def complete_shutdown(self) -> None:
        """Ensure employee is shutdown and clean up resources."""
        if self.process is not None:
            self.process.join()

        self.process = None
        self.conn.close()

    def shutdown(self) -> None:
        """Initiate and complete shutdown."""
        self.initiate_shutdown()
        self.complete_shutdown()

    @property
    def recipient_string(self) -> str:
        """Return a string representation of the employee."""
        return f'{"Manager" if self.is_manager else "Worker"} {self.id}'

    @property
    def has_idle_resources(self) -> bool:
        return self.num_idle_workers > 0

    def get_num_of_tasks_sent_since(
        self,
        read_receipt: RuntimeAddress | None,
    ) -> int:
        """Return the number of tasks sent since the read receipt."""
        if read_receipt is None:
            return sum(count for _, count in self.submit_cache)

        for i, (addr, _) in enumerate(self.submit_cache):
            if addr == read_receipt:
                self.submit_cache = self.submit_cache[i:]
                return sum(count for _, count in self.submit_cache[1:])

        raise RuntimeError('Read receipt not found in submit cache.')


def sigint_handler(signum: int, _: FrameType | None, node: ServerBase) -> None:
    """Interrupt the node."""
    if not node.running:
        return

    node.running = False
    node.terminate_hotline.send(b'\0')
    _logger.info('Server interrupted.')


class ServerBase:
    """Base class for all non-worker process nodes in the BQSKit Runtime."""

    def __init__(self) -> None:
        """Initialize a runtime node component."""

        self.lower_id_bound = 0
        self.upper_id_bound = int(2 ** 30)
        """
        The node starts with an ID range from 0 -> 2^30. ID ranges are then
        assigned to managers by evenly splitting this range.

        Managers then recursively split their range when connecting the sub-
        managers. Finally, workers are assigned specific ids from within this
        range.
        """

        self.running = True
        """True while the node is running."""

        self.sel = selectors.DefaultSelector()
        """Used to efficiently idle and wake when communication is ready."""

        p, self.terminate_hotline = socket.socketpair()
        self.sel.register(p, selectors.EVENT_READ, MessageDirection.SIGNAL)
        """Terminate hotline is used to unblock select while running."""

        self.employees: list[RuntimeEmployee] = []
        """Tracks this node's employees, which are managers or workers."""

        self.conn_to_employee_dict: dict[Connection, RuntimeEmployee] = {}
        """Used to find the employee associated with a message."""

        self._pool: list[tuple[int, float, int, RuntimeTask]] = []
        self._pool_seq = 0
        self._pool_cursor = 0

        # Declared here, unconditionally, for the reason job 1024323 taught:
        # an undeclared counter is not a missing statistic, it is a KeyError or
        # an AttributeError that propagates out and kills the run while Slurm
        # still reports COMPLETED 0:0.
        self._pool_drains = 0
        # Placement stalls: a drain that ended with tasks still pooled and
        # nowhere to put them. Recorded WITH the idle count this level
        # believed at that moment, because the three explanations are only
        # distinguishable together:
        #   blocked, believed idle == 0   -> the credit is spent or stale
        #   blocked, believed idle  > 0   -> eligibility rejected every target
        #   never blocked                 -> supply, and no placement fix helps
        # The worker -> boss hop. handle_waiting discounts a report by the
        # tasks sent since the reporter's read receipt, so a worker that says
        # "I am idle" can be recorded as busy. That is correct when a task is
        # genuinely in flight to it and wrong if the discount outlives the
        # flight, and only counting tells those apart.
        self._waiting_msgs = 0
        self._waiting_zeroed = 0
        self._waiting_unaccounted_sum = 0
        self._pool_blocked = 0
        self._pool_blocked_depth_sum = 0
        self._pool_blocked_idle_sum = 0
        self._pool_depth_sum = 0
        self._pool_depth_max = 0
        # How often the owner-first branch could fire, and did. The owner is
        # the worker that submitted these children and is still running them,
        # so it is not idle at dispatch time -- this counts whether that makes
        # the preference unreachable in practice.
        self._pool_owner_eligible = 0
        self._pool_owner_hit = 0

        self._cpu_snapshot: tuple[float, float] | None = None
        """Previous (busy, total) jiffies, for the free-core measurement."""

        self._last_occupancy_bcast = 0.0
        """Throttle state for broadcast_occupancy.

        Starts at zero rather than the current time so the first idle-count
        change publishes immediately: a run that never fills its workers would
        otherwise wait a tenth of a second before any pass could learn there
        was room to speculate into.
        """

        # Servers do not need blas threads
        set_blas_thread_counts(1)

        # Safely and immediately exit on interrupt signals
        handle = functools.partial(sigint_handler, node=self)
        signal.signal(signal.SIGINT, handle)

        # Start outgoing thread
        self.outgoing: Queue[tuple[Connection, RuntimeMessage, Any]] = Queue()
        self.outgoing_thread = Thread(target=self.send_outgoing, daemon=True)
        self.outgoing_thread.start()
        _logger.info('Started outgoing thread.')

    def connect_to_managers(self, ipports: Sequence[tuple[str, int]]) -> None:
        """Connect to all managers given by endpoints in `ipports`."""
        d = len(ipports)
        self.step_size = (self.upper_id_bound - self.lower_id_bound) // d

        # Establish connections with all managers
        manager_conns = []
        for i, (ip, port) in enumerate(ipports):
            lb = self.lower_id_bound + (i * self.step_size)
            ub = min(
                self.lower_id_bound + ((i + 1) * self.step_size),
                self.upper_id_bound,
            )
            manager_conns.append(self.connect_to_manager(ip, port, lb, ub))
            _logger.info(f'Connected to manager {i} at {ip}:{port}.')
            _logger.debug(f'Gave bounds {lb=} and {ub=} to manager {i}.')

        # Wait for started messages from all managers and register them
        self.total_workers = 0
        for i, conn in enumerate(manager_conns):
            msg, num_workers = conn.recv()
            assert msg == RuntimeMessage.STARTED
            self.employees.append(
                RuntimeEmployee(
                    i,
                    conn,
                    num_workers,
                    is_manager=True,
                ),
            )
            self.conn_to_employee_dict[conn] = self.employees[-1]
            self.sel.register(
                conn,
                selectors.EVENT_READ,
                MessageDirection.BELOW,
            )
            _logger.info(f'Registered manager {i} with {num_workers=}.')
            self.total_workers += num_workers
        self.num_idle_workers = self.total_workers

        _logger.info(f'Node has {self.total_workers} total workers.')

        # Tell each manager how many workers this node commands in total. A
        # manager cannot otherwise know how large its siblings are, and that
        # figure is the denominator it needs to retain only a fair share of a
        # batch submitted from below (see Manager.send_up_or_schedule_tasks).
        # This must wait until every manager has reported, hence its position
        # after the registration loop rather than inside connect_to_manager.
        for conn in manager_conns:
            self.outgoing.put(
                (conn, RuntimeMessage.STARTED, self.total_workers),
            )

    def connect_to_manager(
        self,
        ip: str,
        port: int,
        lb: int,
        ub: int,
    ) -> Connection:
        """
        Connect to a manager at the endpoint given by `ip` and `port`.

        Args:
            ip (str): The IP address where the manager is expected to be
                listening.

            port (int): The port number on which the manager is expected
                to be listening.

            lb (int): The ID lower bound to send to the manager.

            ub (int): The ID upper bound to send to the manager.
        """
        max_retries = 5
        wait_time = .25

        for _ in range(max_retries):
            try:
                conn = Client((ip, port))
            except ConnectionRefusedError:
                time.sleep(wait_time)
                wait_time *= 2
            else:
                conn.send((RuntimeMessage.CONNECT, (lb, ub)))
                return conn

        raise RuntimeError(f'Manager connection refused at {ip}:{port}')

    def spawn_workers(
        self,
        num_workers: int = -1,
        port: int = default_worker_port,
        logging_level: int = logging.WARNING,
        num_blas_threads: int = 1,
    ) -> None:
        """
        Spawn worker processes.

        Args:
            num_workers (int): The number of workers to spawn. If -1,
                then spawn as many workers as CPUs on the system.
                (Default: -1).

            port (int): The port this server will listen for workers on.
                Default can be found in the
                :obj:`~bqskit.runtime.default_worker_port` global variable.

            logging_level (int): The logging level for the workers.

            num_blas_threads (int): The number of threads to use in BLAS
                libraries. (Default: 1).
        """
        if num_workers == -1:
            oscount = os.cpu_count()
            num_workers = oscount if oscount else 1

        if self.lower_id_bound + num_workers >= self.upper_id_bound:
            raise RuntimeError('Insufficient id range for workers.')

        # Create and start all worker processes
        #
        # BQSKit's worker already knows how to pin itself (worker.py, `cpu`
        # argument) but nothing ever passed one, so workers float and the
        # manager competes with 112 of them for the core it needs to run the
        # select() relay on.
        pin = _PIN_WORKERS
        ncpu = os.cpu_count() or 1
        overflowed = False
        procs = {}
        for i in range(num_workers):
            w_id = self.lower_id_bound + i
            kwargs = {
                'logging_level': logging_level,
                'num_blas_threads': num_blas_threads,
            }
            if pin:
                cpu = i
                if cpu < ncpu:
                    kwargs['cpu'] = cpu
                elif not overflowed:
                    overflowed = True
                    _logger.warning(
                        'Only %d cores for %d workers; workers '
                        'from index %d left unpinned.',
                        ncpu, num_workers, i,
                    )
            procs[w_id] = Process(
                target=start_worker,
                args=(w_id, port),
                kwargs=kwargs,
            )
            procs[w_id].daemon = True
            procs[w_id].start()
            _logger.debug(f'Stated worker process {i}.')

        if pin:
            _logger.info('%d workers pinned 1:1, no core reserved.', num_workers)

        # Listen for the worker connections
        family = 'AF_INET' if sys.platform == 'win32' else None
        listener = Listener(('localhost', port), family, backlog=num_workers)
        conns = [listener.accept() for _ in range(num_workers)]
        listener.close()

        # Organize all workers into the employees data structure
        temp_reorder = {}
        for i, conn in enumerate(conns):
            msg, w_id = conn.recv()
            assert msg == RuntimeMessage.STARTED
            employee = RuntimeEmployee(w_id, conn, 1, procs[w_id])
            temp_reorder[w_id - self.lower_id_bound] = employee
            self.conn_to_employee_dict[conn] = employee

        # The employess list needs to be sorted according to the IDs
        for i in range(num_workers):
            self.employees.append(temp_reorder[i])

        # Register employee communication
        for employee in self.employees:
            self.sel.register(
                employee.conn,
                selectors.EVENT_READ,
                MessageDirection.BELOW,
            )
            _logger.debug(f'Registered worker {employee.id}.')

        self.step_size = 1
        self.total_workers = num_workers
        self.num_idle_workers = num_workers
        _logger.info(f'Node has spawned {num_workers} workers.')

    def connect_to_workers(
        self,
        num_workers: int = -1,
        port: int = default_worker_port,
    ) -> None:
        """
        Connect to worker processes.

        Args:
            num_workers (int): The number of workers to expect. If -1,
                then expect as many workers as CPUs on the system.
                (Default: -1).

            port (int): The port this server will listen for workers on.
                Default can be found in the
                :obj:`~bqskit.runtime.default_worker_port` global variable.
        """
        if num_workers == -1:
            oscount = os.cpu_count()
            num_workers = oscount if oscount else 1

        _logger.info(f'Expecting {num_workers} worker connections.')

        if self.lower_id_bound + num_workers >= self.upper_id_bound:
            raise RuntimeError('Insufficient id range for workers.')

        # Listen for the worker connections
        family = 'AF_INET' if sys.platform == 'win32' else None
        listener = Listener(('localhost', port), family, backlog=num_workers)
        conns = [listener.accept() for _ in range(num_workers)]
        listener.close()

        for i, conn in enumerate(conns):
            w_id = self.lower_id_bound + i
            self.outgoing.put((conn, RuntimeMessage.STARTED, w_id))
            employee = RuntimeEmployee(w_id, conn, 1)
            self.employees.append(employee)
            self.conn_to_employee_dict[conn] = employee

        # Register employee communication
        for employee in self.employees:
            w_id = employee.id
            assert employee.conn.recv() == (RuntimeMessage.STARTED, w_id)
            self.sel.register(
                employee.conn,
                selectors.EVENT_READ,
                MessageDirection.BELOW,
            )
            _logger.info(f'Registered worker {w_id}.')

        self.step_size = 1
        self.total_workers = num_workers
        self.num_idle_workers = num_workers
        _logger.info(f'Node has connected to {num_workers} workers.')

    def listen_once(self, ip: str, port: int) -> Connection:
        """Listen on `ip`:`port` for a connection and return on first one."""
        family = 'AF_INET' if sys.platform == 'win32' else None
        listener = Listener((ip, port), family)
        conn = listener.accept()
        listener.close()
        return conn

    def send_outgoing(self) -> None:
        """Outgoing thread forwards messages as they are created.

        One thread, one FIFO, every destination -- so a large SUBMIT_BATCH is
        head-of-line blocking for whatever sits behind it, and the pool's
        service-class ordering does not survive past the queue. Both are fixable
        (per-destination queues served round-robin, i.e. virtual output queues),
        but only worth fixing if the wire is actually the constraint, so the
        depth is sampled here and reported rather than assumed.
        """
        while True:
            outgoing = self.outgoing.get()

            if not self.running:
                # NodeBase's handle_shutdown will put a dummy value in the
                # queue to wake the thread up so it can exit safely.
                # Hence the node.running check now rather than in the
                # while condition.
                break

            if outgoing[0].closed:
                continue

            try:
                outgoing[0].send((outgoing[1], outgoing[2]))
            except (EOFError, ConnectionResetError):
                self.handle_disconnect(outgoing[0])
                _logger.warning('Connection reset while sending message.')
                continue

            if _logger.isEnabledFor(logging.DEBUG):
                to = self.get_to_string(outgoing[0])
                _logger.debug(f'Sent message {outgoing[1].name} to {to}.')

            if outgoing[1] == RuntimeMessage.SUBMIT_BATCH:
                _logger.log(1, f'[{outgoing[2][0]}] * {len(outgoing[2])}\n')
            else:
                _logger.log(1, f'{outgoing[2]}\n')

            self.outgoing.task_done()

    def _start_placement_probe(self) -> None:
        """Sample (pool depth, idle workers) on a FIXED PERIOD, off-thread.

        Placement idle is defined as: work is available AND capacity is
        available AND the two have not been matched. That is a conjunction of
        two live quantities, so it can only be measured by reading both at the
        same instant, on a clock that has nothing to do with either.

        Two earlier attempts measured proxies instead and both were invalid.
        `_pool_blocked` counts drains that ended with no employee holding an
        idle worker -- but that is the loop's own exit condition, so it fires
        whenever the pool is deeper than the free capacity, and the companion
        `_pool_blocked_idle_sum` is then identically zero by construction, not
        by observation. The other read was taken inside the drain, which is
        triggered by the very events being measured.

        This thread samples on its own clock, so `both_positive` is the honest
        quantity: the fraction of samples in which tasks were queued while
        workers sat idle.
        """
        _dir = os.environ.get('BQPROF_PLACEMENT_DIR')
        if not _dir:
            return
        import json as _json
        import threading as _th

        period = float(os.environ.get('BQPROF_PLACEMENT_PERIOD', '0.05'))
        path = f'{_dir}/placement_{os.getpid()}.jsonl'

        def _loop() -> None:
            n = both = pool_pos = idle_pos = 0
            depth_sum = idle_sum = 0
            t0 = time.time()
            with open(path, 'w') as fh:
                while self.running:
                    time.sleep(period)
                    depth = len(self._pool)
                    idle = self.num_idle_workers
                    n += 1
                    depth_sum += depth
                    idle_sum += idle
                    pool_pos += depth > 0
                    idle_pos += idle > 0
                    both += (depth > 0 and idle > 0)
                    if n % 200 == 0:
                        fh.write(_json.dumps({
                            't': round(time.time() - t0, 2), 'samples': n,
                            'pool_positive': pool_pos, 'idle_positive': idle_pos,
                            'both_positive': both,
                            'depth_mean': round(depth_sum / n, 3),
                            'idle_mean': round(idle_sum / n, 3),
                            'total_workers': self.total_workers,
                        }) + '\n')
                        fh.flush()

        t = _th.Thread(target=_loop, daemon=True, name='placement-probe')
        t.start()
        _logger.info('placement probe on, period %.3fs -> %s', period, path)

    def run(self) -> None:
        """Main loop."""
        _logger.info(f'{self.__class__.__name__} running...')
        self._start_placement_probe()

        try:
            while self.running:
                # Wait for messages
                events = self.sel.select()  # Say that 5 times fast

                for key, _ in events:
                    # Unpack message, payload, and direction.
                    conn = cast(Connection, key.fileobj)
                    direction = cast(MessageDirection, key.data)

                    # If interrupted by signal, shutdown and exit
                    if direction == MessageDirection.SIGNAL:
                        _logger.debug('Received interrupt signal.')
                        self.handle_shutdown()
                        return

                    # Unpack and Log message
                    try:
                        msg, payload = conn.recv()
                    except (EOFError, ConnectionResetError):
                        self.handle_disconnect(conn)
                        continue
                    log = f'Received message {msg.name} from {direction.name}.'
                    _logger.debug(log)
                    if msg == RuntimeMessage.SUBMIT_BATCH:
                        _logger.log(1, f'[{payload[0]}] * {len(payload)}\n')
                    else:
                        _logger.log(1, f'{payload}\n')

                    # Handle message
                    self.handle_message(msg, direction, conn, payload)

        except Exception:
            exc_info = sys.exc_info()
            error_str = ''.join(traceback.format_exception(*exc_info))
            _logger.error(error_str)
            self.handle_system_error(error_str)

        finally:
            self.handle_shutdown()

    @abc.abstractmethod
    def handle_message(
        self,
        msg: RuntimeMessage,
        direction: MessageDirection,
        conn: Connection,
        payload: Any,
    ) -> None:
        """
        Process the message coming from `direction`.

        Args:
            msg (RuntimeMessage): The message type to handle.

            direction (MessageDirection): The direction the message came from.

            conn (Connection): The connection object where this came from.

            payload (Any): The message data.
        """

    @abc.abstractmethod
    def handle_system_error(self, error_str: str) -> None:
        """
        Handle an error in runtime code as opposed to client code.

        This is called when an error arises in runtime code not in a
        RuntimeTask's coroutine code.
        """

    @abc.abstractmethod
    def get_to_string(self, conn: Connection) -> str:
        """Return a string representation of the connection."""

    def handle_shutdown(self) -> None:
        """Shutdown the node and release resources."""
        # Stop running
        _logger.info('Shutting down node.')
        # At INFO because it answers a standing question rather than tracing a
        # run: max depth 1 means the pool never held a backlog, and therefore
        # that ordering it -- LPT, heaviest-first, anything -- cannot change a
        # placement. Emitted before `running = False` so a node killed by
        # SIGTERM mid-shutdown has already said it.
        if self._pool_drains:
            _logger.info(
                'pool depth: %d drains, mean %.2f, max %d, %d left undrained',
                self._pool_drains,
                self._pool_depth_sum / self._pool_drains,
                self._pool_depth_max,
                len(self._pool),
            )
        self.running = False

        # Instruct employees to shutdown
        for employee in self.employees:
            employee.initiate_shutdown()

        for employee in self.employees:
            employee.complete_shutdown()

        self.employees.clear()
        _logger.debug('Shutdown employees.')

        # Close selector
        self.sel.close()
        _logger.debug('Cleared selector.')

        # Close outgoing thread
        if self.outgoing_thread.is_alive():
            self.outgoing.put(b'\0')  # type: ignore
            self.outgoing_thread.join()
            _logger.debug('Joined outgoing thread.')
            assert not self.outgoing_thread.is_alive()

    def handle_disconnect(self, conn: Connection) -> None:
        """Remove `conn` from the server."""
        self.sel.unregister(conn)
        conn.close()

        # If one of my employees crashed/shutdown/disconnected, I shutdown
        if conn in self.conn_to_employee_dict:
            self.handle_shutdown()

    def assign_tasks(
        self,
        tasks: Sequence[RuntimeTask],
    ) -> list[list[RuntimeTask]]:
        """
        Go through the tasks and assign each one to an employee.

        Strategy:
            - Assign first to idle workers evenly and randomly
            - Employees with more idle workers get prioritized
            - Assign remaining tasks to employees with the fewest tasks
        """
        # assignments will contain the assigned task list for each employee
        assignments: list[list[RuntimeTask]] = [[] for _ in self.employees]

        # Every employee's id repeated as many time as it has idle workers:
        idle_id_repeated_list: list[int] = sum(
            (
                [i] * e.num_idle_workers
                for i, e in enumerate(self.employees)
            ), [],
        )

        # Shuffle to reduce chance of inefficiency described in handle_waiting
        random.shuffle(idle_id_repeated_list)

        # Assign tasks to idle workers in random order
        for idle_employee_id, task in zip(idle_id_repeated_list, tasks):
            assignments[idle_employee_id].append(task)

        # Check if there are more tasks to be assigned
        num_remaining_tasks = len(tasks) - len(idle_id_repeated_list)

        if num_remaining_tasks <= 0:
            return assignments

        remaining_tasks = list(tasks[-num_remaining_tasks:])

        # Sort the employees by how many tasks they have
        ntasks = sorted([
            (
                e.num_tasks + len(assignments[i]),  # Consider idle assignments
                random.random(),  # Random value for tie breaker
                i,
            )
            for i, e in enumerate(self.employees)
        ])

        while len(remaining_tasks) > 0:
            num_tasks, r, employee_id = ntasks[0]
            assignments[employee_id].append(remaining_tasks.pop())
            ntasks[0] = (num_tasks + 1, r, employee_id)

            # Maintain sorted order by swapping up the updated count
            idx = 0
            while idx + 1 < len(ntasks) and ntasks[idx] > ntasks[idx + 1]:
                ntasks[idx + 1], ntasks[idx] = ntasks[idx], ntasks[idx + 1]
                idx += 1

        return assignments

    def _drain_pool(self) -> None:
        """Hand pooled tasks to employees, preferring three levels in order.

        Tasks retain their service class first, then prefer the employee that
        owns their parent, and finally any other employee with an idle worker.
        A task is delivered only when its recipient has an idle worker; this is
        the difference between one shared pool and N private queues.
        """
        if not self._pool:
            return
        # Pool depth on entry. This decides whether "heaviest task first" can
        # do anything at all: LPT reorders a BACKLOG, and job 1023530 measured
        # it at 0.997x for the reason that no backlog ever formed -- 21 blocks
        # against 112 workers, all placed by the first drain, so there was no
        # order to change. Two things have changed since: parallel multistart
        # submits M tasks per block, and BQSKIT_SCAN_LOOKAHEAD submits up to 32,
        # so a single block can now offer 32 at once.
        #
        # If _pool_depth_max stays at 1 the question is closed for good. If it
        # runs deep, the next step is NOT switching LPT on -- cost_hint is set
        # in exactly one place (foreach.py, for blocks), so every inner task
        # carries 0.0 and LPT would sort ~30 items while leaving the other 99%
        # tied. Populating cost_hint for the inner tasks comes first.
        self._pool_drains += 1
        _depth_in = len(self._pool)
        self._pool_depth_sum += _depth_in
        if _depth_in > self._pool_depth_max:
            self._pool_depth_max = _depth_in
        # Emitted DURING the run, every _POOL_STAT_EVERY drains, not at
        # shutdown. The first version logged in handle_shutdown and printed
        # nothing at all: srun kills the manager with SIGTERM, base.py only
        # registers a SIGINT handler, so handle_shutdown never runs and
        # "Shutting down node" appears zero times in a completed job's log.
        if self._pool_drains % _POOL_STAT_EVERY == 0:
            _logger.info(
                'pool: %d drains, depth mean %.2f max %d, '
                'owner %d/%d (%.1f%%), blocked %d (%.1f%%) '
                'at depth %.1f believing %.2f idle; '
                'waiting %d msgs, %d zeroed (%.1f%%), unaccounted mean %.2f',
                self._pool_drains,
                self._pool_depth_sum / self._pool_drains,
                self._pool_depth_max,
                self._pool_owner_hit,
                self._pool_owner_eligible,
                100.0 * self._pool_owner_hit
                / max(1, self._pool_owner_eligible),
                self._pool_blocked,
                100.0 * self._pool_blocked / self._pool_drains,
                self._pool_blocked_depth_sum / max(1, self._pool_blocked),
                self._pool_blocked_idle_sum / max(1, self._pool_blocked),
                self._waiting_msgs,
                self._waiting_zeroed,
                100.0 * self._waiting_zeroed / max(1, self._waiting_msgs),
                self._waiting_unaccounted_sum / max(1, self._waiting_msgs),
            )
        batches: dict[int, tuple[RuntimeEmployee, list[RuntimeTask]]] = {}
        progress = True
        while self._pool and progress:
            progress = False
            _cls, _cost, _seq, task = heapq.heappop(self._pool)
            target = None
            owner = None
            if self.is_my_worker(task.return_address.worker_id):
                owner = self.get_employee_responsible_for(
                    task.return_address.worker_id,
                )
                self._pool_owner_eligible += 1
                if owner.num_idle_workers > 0:
                    target = owner
                    self._pool_owner_hit += 1
            if target is None:
                # Round-robin from a rotating cursor rather than a scan from
                # employee 0: a fixed start biases every placement toward the
                # low-numbered employees, which is the imbalance the pool is
                # here to remove.
                n = len(self.employees)
                for offset in range(n):
                    e = self.employees[(self._pool_cursor + offset) % n]
                    if e is owner:
                        continue
                    if e.num_idle_workers > 0:
                        target = e
                        self._pool_cursor = (
                            self._pool_cursor + offset + 1
                        ) % n
                        break
            if target is None:
                # Nobody has room; put it back and stop.
                heapq.heappush(self._pool, (_cls, _cost, _seq, task))
                self._pool_blocked += 1
                self._pool_blocked_depth_sum += len(self._pool)
                self._pool_blocked_idle_sum += sum(
                    e.num_idle_workers for e in self.employees
                )
                break
            batches.setdefault(id(target), (target, []))[1].append(task)
            target.num_idle_workers -= 1
            target.num_tasks += 1
            progress = True

        # Coalesce before touching the wire. The relay is one select() loop per
        # level, so a drain that places many tasks avoids one pickle per task.
        # Order within a batch is preserved, so service-class ordering survives.
        for employee, batch in batches.values():
            if len(batch) == 1:
                self.outgoing.put(
                    (employee.conn, RuntimeMessage.SUBMIT, batch[0]),
                )
            else:
                self.outgoing.put(
                    (employee.conn, RuntimeMessage.SUBMIT_BATCH, batch),
                )
            # Same read-receipt accounting assign_tasks does: without it a
            # WAITING that races an in-flight placement is credited as idle
            # twice.
            employee.submit_cache.append((batch[0].unique_id, len(batch)))

        self.num_idle_workers = sum(
            e.num_idle_workers for e in self.employees
        )

    def pooled_idle_workers(self) -> int:
        """Return idle capacity to advertise upward, net of backlog."""
        return max(0, self.num_idle_workers - len(self._pool))

    # Job 1023530 measured this key at 0.997x not because ordering was
    # useless, but because 21 blocks for 112 workers all dispatched on the
    # first drain, leaving no queue to reorder. The inner-task cost hints and
    # expanded speculative capacity now make that queue exist.
    def schedule_tasks(self, tasks: Sequence[RuntimeTask]) -> None:
        """Add tasks to the shared pool and dispatch any that fit."""
        if len(tasks) == 0:
            return
        for task in tasks:
            self._pool_seq += 1
            heapq.heappush(
                self._pool,
                (task.priority, -task.cost_hint, self._pool_seq, task),
            )
        self._drain_pool()

    def send_result_down(self, result: RuntimeResult) -> None:
        """Send the `result` to the appropriate employee."""
        dest_worker_id = result.return_address.worker_id

        if not self.is_my_worker(dest_worker_id):
            raise RuntimeError('Cannot send result to unmanaged worker.')

        employee = self.get_employee_responsible_for(dest_worker_id)
        self.outgoing.put((employee.conn, RuntimeMessage.RESULT, result))

    def is_my_worker(self, worker_id: int) -> bool:
        """Return true if `worker_id` is one of my workers (recursively)."""
        employee_id = (worker_id - self.lower_id_bound) // self.step_size
        return 0 <= employee_id < len(self.employees)

    def get_employee_responsible_for(self, worker_id: int) -> RuntimeEmployee:
        """Return the employee that manages `worker_id`."""
        employee_id = (worker_id - self.lower_id_bound) // self.step_size
        return self.employees[employee_id]

    def broadcast(self, msg: RuntimeMessage, payload: Any) -> None:
        """Broadcast a cancel message to my employees."""
        for employee in self.employees:
            self.outgoing.put((employee.conn, msg, payload))

    def handle_importpath(self, paths: list[str]) -> None:
        """Update the system path with the given paths."""
        for path in paths:
            if path not in sys.path:
                sys.path.append(path)
        self.broadcast(RuntimeMessage.IMPORTPATH, paths)

    def handle_waiting(
        self,
        conn: Connection,
        new_idle_count: int,
        read_receipt: RuntimeAddress | None,
    ) -> None:
        """
        Record that an employee is idle with nothing to do.

        There is a race condition that is corrected here. If an employee sends a
        waiting message at the same time that its boss sends it a task, the
        boss's idle count will eventually be incorrect. To fix this, every
        waiting message sent by an employee is accompanied by a read receipt of
        the latest batch of tasks it has processed. The boss can then adjust the
        idle count by the number of tasks sent since the read receipt.
        """
        employee = self.conn_to_employee_dict[conn]
        unaccounted_task = employee.get_num_of_tasks_sent_since(read_receipt)
        adjusted_idle_count = max(new_idle_count - unaccounted_task, 0)
        self._waiting_msgs += 1
        self._waiting_unaccounted_sum += unaccounted_task
        if new_idle_count > 0 and adjusted_idle_count == 0:
            self._waiting_zeroed += 1

        old_count = employee.num_idle_workers
        employee.num_idle_workers = adjusted_idle_count
        self.num_idle_workers += (adjusted_idle_count - old_count)
        assert 0 <= self.num_idle_workers <= self.total_workers
        self._drain_pool()
        self.broadcast_occupancy()

    def broadcast_occupancy(self) -> None:
        """Tell the workers directly below how much of this node is idle.

        Lives on ServerBase rather than on Manager because the attached server
        -- `Compiler(num_workers=N)` -- owns its workers directly, with no
        manager in between. Without this the signal would exist only in the
        detached runtime, and every single-node A/B would silently fall back to
        the estimator while appearing to test the measured path.

        Sent only to employees that are workers. A manager receiving OCCUPANCY
        from above would have no handler for it, and its own workers already
        get the count from it rather than from here.

        Throttled: the idle count moves on every task start and finish,
        thousands per second, while the only consumer -- a pass sizing its own
        speculation -- reads it once per A* round. The messages are node-local,
        so they never reach the single-threaded server relay.
        """
        now = time.monotonic()
        if now - self._last_occupancy_bcast < _OCCUPANCY_INTERVAL:
            return
        self._last_occupancy_bcast = now

        # Free CORES, not unassigned workers. See _node_busy_fraction: the two
        # differ threefold in practice, and it is the core count that says how
        # much extra work the machine can absorb.
        busy_frac, self._cpu_snapshot = _node_busy_fraction(self._cpu_snapshot)
        if busy_frac is None:
            free_cores = self.num_idle_workers
        else:
            free_cores = max(
                self.num_idle_workers,
                int(round(self.total_workers * (1.0 - busy_frac))),
            )
        payload = (self.num_idle_workers, self.total_workers, free_cores)
        for employee in self.employees:
            if not employee.is_manager:
                self.outgoing.put(
                    (employee.conn, RuntimeMessage.OCCUPANCY, payload),
                )


def parse_ipports(ipports_str: Sequence[str]) -> list[tuple[str, int]]:
    """Parse command line ip and port inputs."""
    ipports = []
    for ipport_group in ipports_str:
        for ipport in ipport_group.split(','):
            if ipport.strip() == '':
                continue
            comps = ipport.strip().split(':')

            if len(comps) == 1:
                ip, port = comps[0], str(default_manager_port)
                # Expect only managers to be listening on these ips
                # so default port is manager's default port.

            elif len(comps) == 2:
                ip, port = comps

            else:
                raise ValueError(f'Invalid manager address: {ipport}.')

            if not (0 <= int(port) < 65536):
                raise ValueError(f'Invalid port number: {ipport}')

            ipports.append((ip, int(port)))
    return ipports


def import_tests_package() -> None:
    """
    Import tests package recursively during detached architecture testing.

    This should only be run by the CI test suite from the root bqskit folder.

    credit: https://www.youtube.com/watch?v=t43zBsVcva0
    """
    sys.path.append(os.path.join(os.getcwd()))
    import tests
    import pkgutil
    for mod in pkgutil.walk_packages(tests.__path__, f'{tests.__name__}.'):
        __import__(mod.name, fromlist=['_trash'])
