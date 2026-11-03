"""Single-owner, paced control loop independent of network/model request latency.

Only the thread running ``run`` (or ``run_once`` in tests) calls dispatch,
idle_tick or publish. HTTP/model threads submit commands and read snapshots.
A skill generator must yield exactly once after each physical control tick.
"""
from concurrent.futures import Future
from dataclasses import dataclass
from collections.abc import Iterator
from pathlib import Path
import json
import math
import queue
import threading
import time


class RuntimeClosedError(RuntimeError):
    pass


class RuntimeBusyError(ValueError):
    pass


@dataclass
class Request:
    command: str
    payload: dict
    result: Future


class ControlScheduler:
    def __init__(self, dispatch, idle_tick, should_tick, *, control_hz=60.,
                 publish=None, on_fault=None, timing_path=None, realtime=True,
                 clock=time.monotonic, max_commands_per_tick=16):
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError('control_hz must be positive and finite')
        self.dispatch, self.idle_tick, self.should_tick = dispatch, idle_tick, should_tick
        self.publish, self.on_fault = publish, on_fault
        self.period = 1./control_hz
        self.realtime, self.clock = realtime, clock
        self.max_commands_per_tick = max_commands_per_tick
        self.requests = queue.Queue()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.owner = None
        self.active = None
        self.closed = False
        self.fault = None
        self.started = None
        self.control_started = None
        self.last_logged_fault = None
        self.metrics = dict(control_hz=control_hz, realtime=realtime, control_ticks=0,
            idle_ticks=0, skill_ticks=0, completed_commands=0, rejected_commands=0,
            missed_budgets=0, max_cycle_wall_s=0., max_control_wall_s=0.,
            simulated_control_s=0., wall_elapsed_s=0., control_wall_elapsed_s=0., observed_control_hz=0.)
        self.timing = None
        if timing_path is not None:
            path = Path(timing_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.timing = path.open('w', buffering=1)

    def submit(self, command, payload=None):
        """Thread-safe; this never calls the simulator or model."""
        future = Future()
        with self.lock:
            if self.closed or self.stop_event.is_set():
                future.set_exception(RuntimeClosedError('Online runtime is closed'))
            else:
                self.requests.put(Request(command, dict(payload or {}), future))
        return future

    def snapshot(self):
        with self.lock:
            active = self.active
            return dict(self.metrics, active_command=active[0].command if active else None,
                        fault=self.fault, stopped=self.stop_event.is_set(), closed=self.closed,
                        pending_commands=self.requests.qsize())

    def _assert_owner(self):
        ident = threading.get_ident()
        if self.owner is None:
            self.owner = ident
        elif self.owner != ident:
            raise RuntimeError('Simulation control must remain on its owner thread')

    def _reject(self, request, error):
        if not request.result.done():
            request.result.set_exception(error)
        self.metrics['rejected_commands'] += 1

    def _finish(self, request, result):
        if not request.result.done():
            request.result.set_result(result)
        self.metrics['completed_commands'] += 1

    def run_once(self):
        """Drain bounded commands and execute at most one physical control tick."""
        self._assert_owner()
        if self.closed or self.stop_event.is_set():
            return
        cycle_start = self.clock()
        if self.started is None:
            self.started = cycle_start
        for _ in range(self.max_commands_per_tick):
            try:
                request = self.requests.get_nowait()
            except queue.Empty:
                break
            if request.result.cancelled():
                continue
            if request.command == 'close':
                self._finish(request, {'closed': True})
                self.stop_event.set()
                break
            if self.active is not None:
                self._reject(request, RuntimeBusyError('A skill is executing; wait for its receipt'))
                continue
            try:
                result = self.dispatch(request.command, request.payload)
                if isinstance(result, Iterator):
                    self.active = (request, result)
                else:
                    self._finish(request, result)
            except Exception as exc:
                self._reject(request, exc)
        kind, control_wall = None, 0.
        if not self.stop_event.is_set() and self.fault is None:
            tick_start = self.clock()
            try:
                if self.active is not None:
                    request, iterator = self.active
                    try:
                        next(iterator)
                        kind = 'skill'
                    except StopIteration as result:
                        self.active = None
                        self._finish(request, result.value)
                    except Exception as exc:
                        self.active = None
                        self._reject(request, exc)
                        if not isinstance(exc, ValueError):
                            raise
                elif self.should_tick():
                    self.idle_tick()
                    kind = 'idle'
            except Exception as exc:
                self.fault = f'{type(exc).__name__}: {exc}'
                if self.on_fault:
                    self.on_fault(exc)
            control_wall = self.clock()-tick_start
        # publish runs on the scene owner; readers only see its cached copy.
        if self.publish:
            self.publish()
        elapsed = self.clock()-cycle_start
        if kind:
            if self.control_started is None:
                self.control_started = cycle_start
            self.metrics['control_ticks'] += 1
            self.metrics[kind+'_ticks'] += 1
            active_wall = max(self.clock()-self.control_started, 0.)
            self.metrics['control_wall_elapsed_s'] = active_wall
            self.metrics['observed_control_hz'] = self.metrics['control_ticks']/active_wall if active_wall else 0.
        self.metrics['max_cycle_wall_s'] = max(self.metrics['max_cycle_wall_s'], elapsed)
        self.metrics['max_control_wall_s'] = max(self.metrics['max_control_wall_s'], control_wall)
        late = elapsed > self.period
        self.metrics['missed_budgets'] += int(late)
        self.metrics['simulated_control_s'] = self.metrics['control_ticks']*self.period
        wall = max(self.clock()-self.started, 0.)
        self.metrics['wall_elapsed_s'] = wall
        new_fault = self.fault and self.fault != self.last_logged_fault
        if self.timing is not None and (kind or late or new_fault):
            self.timing.write(json.dumps(dict(control_tick=self.metrics['control_ticks'], kind=kind,
                cycle_wall_s=elapsed, control_wall_s=control_wall, missed_budget=late,
                active_command=self.active[0].command if self.active else None,
                wall_elapsed_s=wall, fault=self.fault), allow_nan=False)+'\n')
            self.last_logged_fault = self.fault

    def run(self):
        """Pace without catch-up bursts; slow work reduces measured wall-clock Hz."""
        self._assert_owner()
        try:
            while not self.stop_event.is_set():
                start = self.clock()
                self.run_once()
                if self.realtime:
                    # Command arrival never bypasses the physical control period.
                    self.stop_event.wait(max(0., self.period-(self.clock()-start)))
                elif self.active is None and (self.fault or not self.should_tick()):
                    self.stop_event.wait(.002)
        finally:
            self.close()

    def close(self):
        """Owner-thread teardown cancels pending work and closes active generators."""
        self._assert_owner()
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.stop_event.set()
        if self.active:
            request, iterator = self.active
            self.active = None
            try:
                close = getattr(iterator, 'close', None)
                if close:
                    close()
            except Exception as exc:
                self.fault = f'Skill cancellation failed: {type(exc).__name__}: {exc}'
            finally:
                self._reject(request, RuntimeClosedError('Runtime closed during skill'))
        while True:
            try:
                self._reject(self.requests.get_nowait(), RuntimeClosedError('Runtime closed before command'))
            except queue.Empty:
                break
        if self.timing is not None:
            self.timing.close()
