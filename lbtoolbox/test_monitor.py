from lbtoolbox.monitor import *
import unittest
import time
import numpy as np

def mktopic():
    return ''.join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"), 8))

def statseq(s1, s2):
    for v1, v2 in zip(s1.vals, s2.vals):
        np.testing.assert_array_almost_equal(v1, v2)
    np.testing.assert_array_equal(s1.e, s2.e)
    np.testing.assert_array_equal(s1.b, s2.b)

class TestStat(unittest.TestCase):
    def testScalar(self):
        s = Stat()
        s.record(3, 1, 2)
        s.record(3.5, 1, 3)

        s2 = Stat()
        s2.__setstate__(s.__getstate__())
        statseq(s, s2)

    def testArray(self):
        s = Stat()
        s.record(np.random.randn(3), 1, 2)
        s.record(np.random.randn(2), 1, 3)

        s2 = Stat()
        s2.__setstate__(s.__getstate__())
        statseq(s, s2)

class TestMonitor(unittest.TestCase):

    def _perepoch(self, m):
        l1 = [3,2,1]
        m.elosses(l1)
        as1 = np.random.randn(5) ; m.stat("arraystat", as1)
        ns1 = np.random.rand()   ; m.stat("numstat", ns1)
        l2 = [0.9, 0.8]
        m.elosses(l2)
        as2 = np.random.randn(5) ; m.stat("arraystat", as2)
        ns2 = np.random.rand()   ; m.stat("numstat", ns2)
        l3 = [0.7, 0.5, 0.45, 0.4]
        m.elosses(l3)
        as3 = np.random.randn(7) ; m.stat("arraystat", as3)
        ns3 = np.random.rand()   ; m.stat("numstat", ns3)

        self.assertEqual(m._e(), 3)
        self.assertEqual(m._b(), len(l1) + len(l2) + len(l3))
        np.testing.assert_array_equal(m['arraystat'].e, [1,2,3])
        np.testing.assert_array_equal(m['arraystat'].b, [len(l1), len(l1) + len(l2), len(l1) + len(l2) + len(l3)])
        np.testing.assert_array_equal(m['arraystat'].vals[0], as1)
        np.testing.assert_array_equal(m['arraystat'].vals[1], as2)
        np.testing.assert_array_equal(m['arraystat'].vals[2], as3)
        np.testing.assert_array_equal(m['numstat'].e, [1,2,3])
        np.testing.assert_array_equal(m['numstat'].b, [len(l1), len(l1) + len(l2), len(l1) + len(l2) + len(l3)])
        self.assertEqual(m['numstat'].vals[0], ns1)
        self.assertEqual(m['numstat'].vals[1], ns2)
        self.assertEqual(m['numstat'].vals[2], ns3)

    def _perbatch(self, m):
        l1 = [3,2,1]
        m.epoch()
        for l in l1:
            m.loss(l)
        as1 = np.random.randn(5) ; m.stat("arraystat", as1)
        ns1 = np.random.rand()   ; m.stat("numstat", ns1)
        l2 = [0.9, 0.8]
        as2 = [np.random.randn(5), np.random.randn(5)]
        ns2 = [np.random.rand(), np.random.rand()]
        m.epoch()
        for l, as_, ns in zip(l2, as2, ns2):
            m.loss(l)
            m.stat("arraystat", as_)
            m.stat("numstat", ns)
        l3 = [0.7, 0.5, 0.45, 0.4]
        m.epoch()
        for l in l3:
            m.loss(l)
        as3 = np.random.randn(7) ; m.stat("arraystat", as3)
        ns3 = np.random.rand()   ; m.stat("numstat", ns3)

        self.assertEqual(m._e(), 3)
        self.assertEqual(m._b(), len(l1) + len(l2) + len(l3))
        np.testing.assert_array_equal(m['arraystat'].e, [1,2,2,3])
        np.testing.assert_array_equal(m['arraystat'].b, [len(l1), len(l1)+1, len(l1)+2, len(l1) + len(l2) + len(l3)])
        np.testing.assert_array_equal(m['arraystat'].vals[0], as1)
        np.testing.assert_array_equal(m['arraystat'].vals[1], as2[0])
        np.testing.assert_array_equal(m['arraystat'].vals[2], as2[1])
        np.testing.assert_array_equal(m['arraystat'].vals[3], as3)
        np.testing.assert_array_equal(m['numstat'].e, [1,2,2,3])
        np.testing.assert_array_equal(m['numstat'].b, [len(l1), len(l1)+1, len(l1)+2, len(l1) + len(l2) + len(l3)])
        self.assertEqual(m['numstat'].vals[0], ns1)
        self.assertEqual(m['numstat'].vals[1], ns2[0])
        self.assertEqual(m['numstat'].vals[2], ns2[1])
        self.assertEqual(m['numstat'].vals[3], ns3)

    def testPerEpoch(self):
        self._perepoch(Monitor())
        self._perepoch(ReportMonitor("test", mktopic()))

    def testPerBatch(self):
        self._perbatch(Monitor())
        self._perbatch(ReportMonitor("test", mktopic()))


class TestReportingWatching(unittest.TestCase):

    def testPerEpoch(self):
        topic = mktopic()
        m = ReportMonitor("test", topic)
        w = WatchMonitor("test", topic)

        # Some time for the connection to happen, or messages will get lost.
        time.sleep(1)

        # SAME AS ABOVE
        l1 = [3,2,1]
        m.elosses(l1)
        as1 = np.random.randn(5) ; m.stat("arraystat", as1)
        ns1 = np.random.rand()   ; m.stat("numstat", ns1)
        l2 = [0.9, 0.8]
        m.elosses(l2)
        as2 = np.random.randn(5) ; m.stat("arraystat", as2)
        ns2 = np.random.rand()   ; m.stat("numstat", ns2)
        l3 = [0.7, 0.5, 0.45, 0.4]
        m.elosses(l3)
        as3 = np.random.randn(7) ; m.stat("arraystat", as3)
        ns3 = np.random.rand()   ; m.stat("numstat", ns3)

        # We checked the reporting monitor previously, now just the watcher.

        # Let's give him a little time!
        time.sleep(2)

        # Now to the watcher!
        w.stats("arraystat", "numstat")
        w.update()

        self.assertEqual(w._e(), m._e())
        self.assertEqual(w._b(), m._b())
        statseq(w['arraystat'], m['arraystat'])
        statseq(w['numstat'], m['numstat'])

    def testPerBatch(self):
        topic = mktopic()
        m = ReportMonitor("test", topic)
        w = WatchMonitor("test", topic)

        # Some time for the connection to happen, or messages will get lost.
        time.sleep(1)

        # SAME AS ABOVE
        l1 = [3,2,1]
        m.epoch()
        for l in l1:
            m.loss(l)
        as1 = np.random.randn(5) ; m.stat("arraystat", as1)
        ns1 = np.random.rand()   ; m.stat("numstat", ns1)
        l2 = [0.9, 0.8]
        as2 = [np.random.randn(5), np.random.randn(5)]
        ns2 = [np.random.rand(), np.random.rand()]
        m.epoch()
        for l, as_, ns in zip(l2, as2, ns2):
            m.loss(l)
            m.stat("arraystat", as_)
            m.stat("numstat", ns)
        l3 = [0.7, 0.5, 0.45, 0.4]
        m.epoch()
        for l in l3:
            m.loss(l)
        as3 = np.random.randn(7) ; m.stat("arraystat", as3)
        ns3 = np.random.rand()   ; m.stat("numstat", ns3)

        # We checked the reporting monitor previously, now just the watcher.

        # Let's give him a little time!
        time.sleep(2)

        # Now to the watcher!
        w.stats("arraystat", "numstat")
        w.update()

        self.assertEqual(w._e(), m._e())
        self.assertEqual(w._b(), m._b())
        statseq(w['arraystat'], m['arraystat'])
        statseq(w['numstat'], m['numstat'])

    def testStress(self):
        topic = mktopic()
        m = ReportMonitor("test", topic)
        w = WatchMonitor("test", topic)

        # Some time for the connection to happen, or messages will get lost.
        time.sleep(1)

        ls = np.random.randn(1000)
        ss = np.random.randn(1000//5)
        As = np.random.randn(1000//10, 123, 456)
        for i, l in enumerate(ls):
            if i % 100 == 0:
                m.epoch()
            m.loss(l)
            if i % 5 == 0:
                m.stat("numstat", ss[i//5])
            if i % 10 == 0:
                m.stat("arraystat", As[i//10])

            time.sleep(np.random.rand()/100)

        # Let's give him a little time!
        time.sleep(2)

        # Now to the watcher!
        w.stats("arraystat", "numstat")
        w.update()

        self.assertEqual(w._e(), m._e())
        self.assertEqual(w._b(), m._b())
        statseq(w['arraystat'], m['arraystat'])
        statseq(w['numstat'], m['numstat'])

# TODO: Use case 2: stats per batch, sometimes hit epoch.
if __name__ == '__main__':
    print("NOTE: a monitor server needs to be running during the test.")
    unittest.main()
