from lbtoolbox.monitor import Stat, ReportMonitor, WatchMonitor
from lbtoolbox.monitor_srv import Store, Collector, CollectorThread, QueryHandler, QueryHandlerThread

import unittest
import time
import numpy as np

def mktopic():
    return ''.join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"), 8))

class TestStat(unittest.TestCase):
    def testPickling(self):
        s1 = Stat()
        s2 = Stat()

        s2.update_from_pickle(s1.pickle_from(0))
        self.assertEqual(len(s2.vals), 0)

        s1.record(np.random.randn(3), 1, 1)
        s2.update_from_pickle(s1.pickle_from(0))
        self.assertEqual(len(s2.vals), 1)
        np.testing.assert_array_almost_equal(s2.vals, s1.vals)
        np.testing.assert_array_almost_equal(s2.e, s1.e)
        np.testing.assert_array_almost_equal(s2.b, s1.b)

        s1.record(np.random.randn(3), 1, 1)
        s1.record(np.random.randn(3), 1, 2)
        s2.update_from_pickle(s1.pickle_from(0))
        self.assertEqual(len(s2.vals), 3)
        np.testing.assert_array_almost_equal(s2.vals, s1.vals)
        np.testing.assert_array_almost_equal(s2.e, s1.e)
        np.testing.assert_array_almost_equal(s2.b, s1.b)

        s1.record(np.random.randn(3), 1, 3)
        s1.record(np.random.randn(3), 1, 3)
        s2.update_from_pickle(s1.pickle_from(s2.b[-1]))
        self.assertEqual(len(s2.vals), 5)
        np.testing.assert_array_almost_equal(s2.vals, s1.vals)
        np.testing.assert_array_almost_equal(s2.e, s1.e)
        np.testing.assert_array_almost_equal(s2.b, s1.b)

class TestCollector(unittest.TestCase):

    def setUp(self):
        self.s = Store()

        # Kinda hacky, but whatever
        self.addr = "ipc:///tmp/lbmon-testing-report-{}.ipc".format(np.random.rand())

        # TODO: How to test both without huge copy-paste?
        self.threaded = True
        if self.threaded:
            self.c = CollectorThread(self.s, self.addr)
            self.c.start()
        else:
            self.c = Collector(self.s, self.addr)

        self.topic = mktopic()
        self.r = ReportMonitor("test", self.topic, self.addr)

        # Let networking settle, even IPC needs it!
        time.sleep(0.5)

    def tearDown(self):
        if self.threaded:
            self.c.stop = True
            self.c.join()

    def testEpochLoss(self):
        self.r.elosses([3,2,1])
        self.r.elosses(np.array([1, 0.8, 0.6, 0.4, 0.2]))
        self.r.elosses([0.01, 0.01])

        if self.threaded:
            time.sleep(1)
        else:
            for _ in range(3):
                self.c.tick()

        # Verify the internals of the monitor!
        mon = self.s.get("test", self.topic)
        self.assertEqual(len(mon._online_losses), 3)
        np.testing.assert_array_equal(mon._online_losses[0], [3,2,1])
        np.testing.assert_array_almost_equal(mon._online_losses[1], [1, 0.8, 0.6, 0.4, 0.2])
        np.testing.assert_array_almost_equal(mon._online_losses[2], [0.01,0.01])

    def testLoss(self):
        self.r.epoch()
        for l in [3,2,1]:
            self.r.loss(l)

        self.r.epoch()
        for l in [1, 0.8, 0.6, 0.4, 0.2]:
            self.r.loss(l)

        self.r.epoch()
        self.r.loss(0.01)
        self.r.loss(0.01)

        if self.threaded:
            time.sleep(1)
        else:
            for _ in range(10):
                self.c.tick()

        # Verify the internals of the monitor!
        mon = self.s.get("test", self.topic)
        self.assertEqual(len(mon._online_losses), 3)
        np.testing.assert_array_equal(mon._online_losses[0], [3,2,1])
        np.testing.assert_array_almost_equal(mon._online_losses[1], [1, 0.8, 0.6, 0.4, 0.2])
        np.testing.assert_array_almost_equal(mon._online_losses[2], [0.01,0.01])

    def testEpochStat(self):
        # Tests sending stats as numbers every batch.
        self.r.elosses([3,2,1])
        self.r.stat("val err", 0.5)
        self.r.elosses(np.array([1, 0.8, 0.6, 0.4, 0.2]))
        self.r.stat("val err", 0.3)
        self.r.elosses([0.01, 0.01])
        self.r.stat("val err", 0.1)

        if self.threaded:
            time.sleep(1)
        else:
            for _ in range(3):
                self.c.tick()

        # Verify the internals of the monitor!
        stat = self.s.get("test", self.topic)._stats["val err"]
        self.assertListEqual(stat.vals, [0.5, 0.3, 0.1])
        self.assertListEqual(stat.e, [1,2,3])
        self.assertListEqual(stat.b, [3,8,10])

    def testStat(self):
        # Tests sending stats as numpy arrays every update.
        self.r.epoch()
        for l in [3,2,1]:
            self.r.loss(l)
            self.r.stat("grad", np.array([0.1, 0.2]))

        self.r.epoch()
        for l in [1, 0.8, 0.6, 0.4, 0.2]:
            self.r.loss(l)
            self.r.stat("grad", np.array([0.2, 0.1]))

        self.r.epoch()
        self.r.loss(0.01)
        self.r.stat("grad", np.array([0.3, 0.3]))
        self.r.loss(0.01)
        self.r.stat("grad", np.array([0.4, 0.4]))

        if self.threaded:
            time.sleep(1)
        else:
            for _ in range(10):
                self.c.tick()

        # Verify the internals of the monitor!
        stat = self.s.get("test", self.topic)._stats["grad"]
        self.assertEqual(len(stat.vals), 10)
        np.testing.assert_array_almost_equal(stat.vals[0], [0.1, 0.2])
        np.testing.assert_array_almost_equal(stat.vals[1], [0.1, 0.2])
        np.testing.assert_array_almost_equal(stat.vals[2], [0.1, 0.2])
        np.testing.assert_array_almost_equal(stat.vals[3], [0.2, 0.1])
        np.testing.assert_array_almost_equal(stat.vals[4], [0.2, 0.1])
        np.testing.assert_array_almost_equal(stat.vals[5], [0.2, 0.1])
        np.testing.assert_array_almost_equal(stat.vals[6], [0.2, 0.1])
        np.testing.assert_array_almost_equal(stat.vals[7], [0.2, 0.1])
        np.testing.assert_array_almost_equal(stat.vals[8], [0.3, 0.3])
        np.testing.assert_array_almost_equal(stat.vals[9], [0.4, 0.4])
        self.assertListEqual(stat.e, [1,1,1,2,2,2,2,2,3,3])
        self.assertListEqual(stat.b, [1,2,3,4,5,6,7,8,9,10])

    def testArchival(self):
        self.r.elosses([3,2,1])

        # New monitor with same name/topic should archive old
        self.r = ReportMonitor("test", self.topic, self.addr)
        self.r.elosses([9,8,7])

        if self.threaded:
            time.sleep(1)
        else:
            for _ in range(3):
                self.c.tick()

        # Verify that the old stuff is gone.
        mon = self.s.get("test", self.topic)
        self.assertEqual(len(mon._online_losses), 1)
        np.testing.assert_array_equal(mon._online_losses[0], [9,8,7])

        # TODO: Verify that the old one has been archived??


class TestQueryHandler(unittest.TestCase):

    def setUp(self):
        self.s = Store()

        # Kinda hacky, but whatever
        addr = "ipc:///tmp/lbmon-testing-watch-{}.ipc".format(np.random.rand())

        # NOTE: This test can't work non-threaded because
        #       we'd need to call `tick` in the middle of `WatchMonitor.update`.
        self.q = QueryHandlerThread(self.s, addr)
        self.q.start()

        self.topic = mktopic()
        self.mon = self.s.get("test", self.topic)
        self.w = WatchMonitor("test", self.topic, addr, timeoutms=2000)

        # Let networking settle, even IPC needs it!
        time.sleep(0.5)

    def tearDown(self):
        self.q.stop = True
        self.q.join()

    def testEpochLoss(self):
        # Should get nothing yet!
        self.w.update()
        self.assertEqual(len(self.w._online_losses), 0)

        # Getting one.
        self.mon.elosses([3,2,1])

        self.w.update()
        self.assertEqual(len(self.w._online_losses), 1)
        np.testing.assert_array_equal(self.w._online_losses[0], [3,2,1])

        # Getting two at a time
        self.mon.elosses(np.array([1, 0.8, 0.6, 0.4, 0.2]))
        self.mon.elosses([0.01, 0.01])

        self.w.update()
        self.assertEqual(len(self.w._online_losses), 3)
        np.testing.assert_array_equal(self.w._online_losses[0], [3,2,1])
        np.testing.assert_array_almost_equal(self.w._online_losses[1], [1, 0.8, 0.6, 0.4, 0.2])
        np.testing.assert_array_almost_equal(self.w._online_losses[2], [0.01,0.01])

    def testLoss(self):
        # Should get nothing yet!
        self.w.update()
        self.assertEqual(len(self.w._online_losses), 0)

        # Getting one, partially.
        self.mon.epoch()
        self.mon.loss(3)
        self.mon.loss(2)

        self.w.update()
        self.assertEqual(len(self.w._online_losses), 1)
        np.testing.assert_array_equal(self.w._online_losses[0], [3,2])

        # Getting the second part of it and some more.
        self.mon.loss(1)
        self.mon.epoch()
        self.mon.loss(1)
        self.mon.loss(0.8)

        self.w.update()
        self.assertEqual(len(self.w._online_losses), 2)
        np.testing.assert_array_equal(self.w._online_losses[0], [3,2,1])
        np.testing.assert_array_almost_equal(self.w._online_losses[1], [1, 0.8])

        # Getting the remainder of one without crossing epochs
        self.mon.loss(0.6)
        self.mon.loss(0.4)
        self.mon.loss(0.2)

        self.w.update()
        self.assertEqual(len(self.w._online_losses), 2)
        np.testing.assert_array_equal(self.w._online_losses[0], [3,2,1])
        np.testing.assert_array_almost_equal(self.w._online_losses[1], [1, 0.8, 0.6, 0.4, 0.2])

        # Getting only the epoch.
        # TODO: This currently only syncs because `mon` is the same instance `q`.
        #       If it was a `ReportingMonitor` this wouldn't send.
        self.mon.epoch()

        self.w.update()
        self.assertEqual(len(self.w._online_losses), 3)
        np.testing.assert_array_equal(self.w._online_losses[0], [3,2,1])
        np.testing.assert_array_almost_equal(self.w._online_losses[1], [1, 0.8, 0.6, 0.4, 0.2])
        np.testing.assert_array_equal(self.w._online_losses[2], [])

        # And last part fully.
        self.mon.loss(0.01)
        self.mon.loss(0.01)

        self.w.update()
        self.assertEqual(len(self.w._online_losses), 3)
        np.testing.assert_array_equal(self.w._online_losses[0], [3,2,1])
        np.testing.assert_array_almost_equal(self.w._online_losses[1], [1, 0.8, 0.6, 0.4, 0.2])
        np.testing.assert_array_almost_equal(self.w._online_losses[2], [0.01,0.01])

    def testEpochStat(self):
        s = self.w._stats["val err"]

        # Should get nothing yet!
        self.w.update()
        self.assertEqual(len(s.vals), 0)
        self.assertEqual(len(s.e), 0)
        self.assertEqual(len(s.b), 0)

        # Getting one.
        self.mon.elosses([3,2,1])
        self.mon.stat("val err", 0.5)

        self.w.update()
        np.testing.assert_array_almost_equal(s.vals, [0.5])
        np.testing.assert_array_almost_equal(s.e, [1])
        np.testing.assert_array_almost_equal(s.b, [3])

        # Getting two at a time
        self.mon.elosses(np.array([1, 0.8, 0.6, 0.4, 0.2]))
        self.mon.stat("val err", 0.3)
        self.mon.elosses([0.01, 0.01])
        self.mon.stat("val err", 0.1)

        self.w.update()
        np.testing.assert_array_almost_equal(s.vals, [0.5, 0.3, 0.1])
        np.testing.assert_array_equal(s.e, [1, 2, 3])
        np.testing.assert_array_equal(s.b, [3, 8, 10])

    def testStat(self):
        s = self.w._stats["grad"]

        # Should get nothing yet!
        self.w.update()
        self.assertEqual(len(s.vals), 0)
        self.assertEqual(len(s.e), 0)
        self.assertEqual(len(s.b), 0)

        # Getting one, partially.
        self.mon.epoch()
        self.mon.loss(3)
        self.mon.stat("grad", np.array([0.10, 0.20]))
        self.mon.loss(2)
        self.mon.stat("grad", np.array([0.11, 0.21]))

        self.w.update()
        np.testing.assert_array_almost_equal(s.vals[0], [0.10, 0.20])
        np.testing.assert_array_almost_equal(s.vals[1], [0.11, 0.21])
        np.testing.assert_array_equal(s.e, [1,1])
        np.testing.assert_array_equal(s.b, [1,2])

        # Getting the second part of it and some more.
        self.mon.loss(1)
        self.mon.stat("grad", np.array([0.12, 0.22]))
        self.mon.epoch()
        self.mon.loss(1)
        self.mon.stat("grad", np.array([0.20, 0.10]))
        self.mon.loss(0.8)
        self.mon.stat("grad", np.array([0.21, 0.11]))

        self.w.update()
        np.testing.assert_array_almost_equal(s.vals[0], [0.10, 0.20])
        np.testing.assert_array_almost_equal(s.vals[1], [0.11, 0.21])
        np.testing.assert_array_almost_equal(s.vals[2], [0.12, 0.22])
        np.testing.assert_array_almost_equal(s.vals[3], [0.20, 0.10])
        np.testing.assert_array_almost_equal(s.vals[4], [0.21, 0.11])
        np.testing.assert_array_equal(s.e, [1,1,1,2,2])
        np.testing.assert_array_equal(s.b, [1,2,3,4,5])

        # Getting the remainder of one without crossing epochs
        self.mon.loss(0.6)
        self.mon.stat("grad", np.array([0.22, 0.12]))
        self.mon.loss(0.4)
        self.mon.stat("grad", np.array([0.23, 0.13]))
        self.mon.loss(0.2)
        self.mon.stat("grad", np.array([0.24, 0.14]))

        self.w.update()
        np.testing.assert_array_almost_equal(s.vals[0], [0.10, 0.20])
        np.testing.assert_array_almost_equal(s.vals[1], [0.11, 0.21])
        np.testing.assert_array_almost_equal(s.vals[2], [0.12, 0.22])
        np.testing.assert_array_almost_equal(s.vals[3], [0.20, 0.10])
        np.testing.assert_array_almost_equal(s.vals[4], [0.21, 0.11])
        np.testing.assert_array_almost_equal(s.vals[5], [0.22, 0.12])
        np.testing.assert_array_almost_equal(s.vals[6], [0.23, 0.13])
        np.testing.assert_array_almost_equal(s.vals[7], [0.24, 0.14])
        np.testing.assert_array_equal(s.e, [1,1,1,2,2,2,2,2])
        np.testing.assert_array_equal(s.b, [1,2,3,4,5,6,7,8])

        # Getting only the epoch.
        # TODO: This currently only syncs because `mon` is the same instance `q`.
        #       If it was a `ReportingMonitor` this wouldn't send.
        self.mon.epoch()

        self.w.update()
        np.testing.assert_array_almost_equal(s.vals[0], [0.10, 0.20])
        np.testing.assert_array_almost_equal(s.vals[1], [0.11, 0.21])
        np.testing.assert_array_almost_equal(s.vals[2], [0.12, 0.22])
        np.testing.assert_array_almost_equal(s.vals[3], [0.20, 0.10])
        np.testing.assert_array_almost_equal(s.vals[4], [0.21, 0.11])
        np.testing.assert_array_almost_equal(s.vals[5], [0.22, 0.12])
        np.testing.assert_array_almost_equal(s.vals[6], [0.23, 0.13])
        np.testing.assert_array_almost_equal(s.vals[7], [0.24, 0.14])
        np.testing.assert_array_equal(s.e, [1,1,1,2,2,2,2,2])
        np.testing.assert_array_equal(s.b, [1,2,3,4,5,6,7,8])

        # And last part fully.
        self.mon.loss(0.01)
        self.mon.stat("grad", np.array([0.3, 0.33]))
        self.mon.loss(0.01)
        self.mon.stat("grad", np.array([0.4, 0.44]))

        self.w.update()
        np.testing.assert_array_almost_equal(s.vals[0], [0.10, 0.20])
        np.testing.assert_array_almost_equal(s.vals[1], [0.11, 0.21])
        np.testing.assert_array_almost_equal(s.vals[2], [0.12, 0.22])
        np.testing.assert_array_almost_equal(s.vals[3], [0.20, 0.10])
        np.testing.assert_array_almost_equal(s.vals[4], [0.21, 0.11])
        np.testing.assert_array_almost_equal(s.vals[5], [0.22, 0.12])
        np.testing.assert_array_almost_equal(s.vals[6], [0.23, 0.13])
        np.testing.assert_array_almost_equal(s.vals[7], [0.24, 0.14])
        np.testing.assert_array_almost_equal(s.vals[8], [0.3, 0.33])
        np.testing.assert_array_almost_equal(s.vals[9], [0.4, 0.44])
        np.testing.assert_array_equal(s.e, [1,1,1,2,2,2,2,2,3,3])
        np.testing.assert_array_equal(s.b, [1,2,3,4,5,6,7,8,9,10])


if __name__ == '__main__':
    unittest.main()
