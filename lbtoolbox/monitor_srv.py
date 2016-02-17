import sys
import time
from collections import defaultdict
from threading import Thread

import nnpy

from monitor import RWMonitor


# TODO:
# - Persistent storage.
# - Mathematical Queries (mean, quantile, ...).


class Store:
    def __init__(self):
        # TODO back by e.g. sqlite3
        # See http://stackoverflow.com/a/18622264/2366315
        self.store = defaultdict(dict)

    def get(self, name, topic):
        if topic in self.store[name]:
            return self.store[name][topic]
        else:
            mon = RWMonitor(name, topic)
            self.store[name][topic] = mon
            return mon

    def archive(self, name, topic):
        # TODO: archive instead of delete?
        #       for this, might want to record starting datetime in c'tor.
        if topic in self.store[name]:
            del self.store[name][topic]


class Collector(object):
    def __init__(self, store, *addrs):
        self.store = store

        self.sub = nnpy.Socket(nnpy.AF_SP, nnpy.SUB)
        for addr in addrs:
            self.sub.bind(addr)
        self.sub.setsockopt(nnpy.SUB, nnpy.SUB_SUBSCRIBE, '')
        self.sub.setsockopt(nnpy.SOL_SOCKET, nnpy.RCVTIMEO, 1000) # in ms
        # TODO? Default is 1024kB.
        # self.sub.setsockopt(nnpy.SOL_SOCKET, nnpy.RCVMAXSIZE, -1)

    def tick(self):
        try:
            msg = self.sub.recv()
        except AssertionError:
            if nnpy.nanomsg.nn_errno() == nnpy.ETIMEDOUT:
                return
            raise

        proj, desc, stat, data = msg.split(b'/', 3)
        proj, desc = proj.decode('utf-8'), desc.decode('utf-8')
        if stat == b'start':
            self.store.archive(proj, desc)
        else:
            mon = self.store.get(proj, desc)
            mon.recv(stat, data)


class CollectorThread(Collector, Thread):
    def __init__(self, store, *addrs):
        Collector.__init__(self, store, *addrs)
        Thread.__init__(self)
        self.stop = False

    def run(self):
        while not self.stop:
            self.tick()


class QueryHandler(object):
    def __init__(self, store, *addrs):
        self.store = store

        self.rep = nnpy.Socket(nnpy.AF_SP, nnpy.REP)
        for addr in addrs:
            self.rep.bind(addr)
        self.rep.setsockopt(nnpy.SOL_SOCKET, nnpy.RCVTIMEO, 1000) # in ms
        self.rep.setsockopt(nnpy.SOL_SOCKET, nnpy.SNDTIMEO, 1000) # in ms

    def tick(self):
        try:
            msg = self.rep.recv()
        except AssertionError:
            if nnpy.nanomsg.nn_errno() == nnpy.ETIMEDOUT:
                return
            raise

        proj, desc, query = msg.split(b'/', 2)
        proj, desc = proj.decode('utf-8'), desc.decode('utf-8')

        mon = self.store.get(proj, desc)
        ans = mon.query(query)
        self.rep.send(ans)


class QueryHandlerThread(QueryHandler, Thread):
    def __init__(self, store, *addrs):
        QueryHandler.__init__(self, store, *addrs)
        Thread.__init__(self)
        self.stop = False

    def run(self):
        while not self.stop:
            self.tick()


if __name__ == "__main__":
    store = Store()
    default_addrs = ["tcp://127.0.0.1:1337", "ipc:///tmp/lbmon-coll.ipc"]
    c = CollectorThread(store, *(sys.argv[1:] or default_addrs))
    c.start()

    default_addrs = ["tcp://127.0.0.1:31337", "ipc:///tmp/lbmon-repo.ipc"]
    r = QueryHandlerThread(store, *(sys.argv[1:] or default_addrs))
    r.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # TODO: see http://zguide.zeromq.org/py:interrupt
        r.stop = True
        c.stop = True
        r.join()
        c.join()
