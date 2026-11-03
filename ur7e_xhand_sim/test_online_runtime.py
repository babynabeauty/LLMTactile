"""CPU-only concurrency tests: no Genesis, GPU, Astra account or checkpoint."""
import threading
import time
import unittest

from ur7e_xhand_sim.online_runtime import ControlScheduler, RuntimeBusyError, RuntimeClosedError


class FakeTrial:
    def __init__(self):
        self.ticks = 0
        self.revision = 0
        self.threads = set()
        self.finished = False
        self.generator_closed = False

    def tick(self):
        self.threads.add(threading.get_ident())
        self.ticks += 1

    def dispatch(self, command, payload):
        self.threads.add(threading.get_ident())
        if command == 'observe':
            return {'observation_id': str(self.revision), 'step': self.ticks}
        if command != 'act':
            raise ValueError('Unknown command')
        return self.execute(payload)

    def execute(self, payload):
        if payload['observation_id'] != str(self.revision):
            raise ValueError('Stale observation')
        self.revision += 1
        try:
            for _ in range(payload.get('ticks', 3)):
                self.tick()
                yield
        finally:
            self.generator_closed = True
        return {'observation_id': str(self.revision), 'step': self.ticks}


class RuntimeTest(unittest.TestCase):
    def test_delayed_astra_does_not_stop_control_or_expire_decision(self):
        trial = FakeTrial()
        scheduler = ControlScheduler(trial.dispatch, trial.tick, lambda: not trial.finished)
        owner = threading.Thread(target=scheduler.run)
        owner.start()
        try:
            initial = scheduler.submit('observe').result(timeout=2.)
            # Represents the network/model wait in a different thread.
            time.sleep(.12)
            later = scheduler.snapshot()
            self.assertGreaterEqual(later['idle_ticks'], 4)
            self.assertGreater(trial.ticks, initial['step'])
            # Physical ticks advance, the decision revision remains valid.
            result = scheduler.submit('act', {'observation_id': initial['observation_id']}).result(timeout=2.)
            self.assertEqual(result['observation_id'], '1')
            self.assertEqual(scheduler.snapshot()['skill_ticks'], 3)
            self.assertEqual(trial.threads, {owner.ident})
            self.assertNotIn(threading.get_ident(), trial.threads)
        finally:
            scheduler.submit('close').result(timeout=2.)
            owner.join(timeout=2.)
        self.assertFalse(owner.is_alive())
        self.assertTrue(scheduler.snapshot()['closed'])
        with self.assertRaises(RuntimeClosedError):
            scheduler.submit('observe').result()

    def test_old_plan_rejected_but_control_and_new_plan_continue(self):
        trial = FakeTrial()
        scheduler = ControlScheduler(trial.dispatch, trial.tick, lambda: True)
        old = scheduler.submit('act', {'observation_id': '0', 'ticks': 1})
        scheduler.run_once()
        scheduler.run_once()
        self.assertEqual(old.result()['observation_id'], '1')
        stale = scheduler.submit('act', {'observation_id': '0'})
        scheduler.run_once()
        with self.assertRaisesRegex(ValueError, 'Stale observation'):
            stale.result()
        self.assertIsNone(scheduler.snapshot()['fault'])
        fresh = scheduler.submit('act', {'observation_id': '1', 'ticks': 1})
        scheduler.run_once()
        scheduler.run_once()
        self.assertEqual(fresh.result()['observation_id'], '2')
        scheduler.close()

    def test_busy_rejection_and_close_cancel_active_skill(self):
        trial = FakeTrial()
        scheduler = ControlScheduler(trial.dispatch, trial.tick, lambda: True)
        active = scheduler.submit('act', {'observation_id': '0', 'ticks': 100})
        scheduler.run_once()
        busy = scheduler.submit('act', {'observation_id': '1'})
        scheduler.run_once()
        with self.assertRaises(RuntimeBusyError):
            busy.result()
        close = scheduler.submit('close')
        pending = scheduler.submit('observe')
        scheduler.run_once()
        scheduler.close()
        self.assertEqual(close.result(), {'closed': True})
        with self.assertRaises(RuntimeClosedError):
            active.result()
        with self.assertRaises(RuntimeClosedError):
            pending.result()
        self.assertTrue(trial.generator_closed)
        self.assertEqual(trial.ticks, 2)

    def test_control_fault_stops_physics_but_close_still_works(self):
        faults = []
        def fail():
            raise RuntimeError('fake guard stop')
        scheduler = ControlScheduler(lambda *_: None, fail, lambda: True, on_fault=faults.append)
        scheduler.run_once()
        scheduler.run_once()
        self.assertEqual(len(faults), 1)
        self.assertIn('fake guard stop', scheduler.snapshot()['fault'])
        closed = scheduler.submit('close')
        scheduler.run_once()
        self.assertEqual(closed.result(), {'closed': True})
        scheduler.close()

    def test_slow_cycle_counts_miss_without_catchup_ticks(self):
        now, ticks = [0.], []
        def slow_tick():
            ticks.append(now[0])
            now[0] += .05
        scheduler = ControlScheduler(lambda *_: None, slow_tick, lambda: True, clock=lambda: now[0])
        scheduler.run_once()
        self.assertEqual(len(ticks), 1)
        self.assertEqual(scheduler.snapshot()['missed_budgets'], 1)
        self.assertEqual(scheduler.snapshot()['control_ticks'], 1)
        self.assertAlmostEqual(scheduler.snapshot()['observed_control_hz'], 20.)
        scheduler.run_once()
        self.assertEqual(len(ticks), 2)
        scheduler.close()

    def test_owner_cannot_move_to_a_network_thread(self):
        scheduler = ControlScheduler(lambda *_: None, lambda: None, lambda: False)
        scheduler.run_once()
        error = []
        def wrong_thread():
            try:
                scheduler.run_once()
            except Exception as exc:
                error.append(exc)
        worker = threading.Thread(target=wrong_thread)
        worker.start()
        worker.join()
        self.assertEqual(len(error), 1)
        self.assertIn('owner thread', str(error[0]))
        scheduler.close()


if __name__ == '__main__':
    unittest.main()
