from collections import defaultdict

try:
    import cPickle as pickle
except ImportError:
    import pickle

import struct
from io import BytesIO
import numpy as np
from numbers import Real


def np2bytes(what):
    io = BytesIO()
    np.save(io, what)
    return io.getvalue()


def bytes2np(b):
    return np.load(BytesIO(b))


class Stat:
    def __init__(self):
        self.vals = []
        self.e, self.b = [], []

    def record(self, val, e, b):
        # Insert in such a way that `b` is sorted ascending.
        # Start finding the right place from the end as records are most likely
        # to come in the correct order.
        idx = 0
        for idx in range(len(self.b), 0, -1):
            if self.b[idx-1] <= b:
                break
        self.vals.insert(idx, val)
        self.e.insert(idx, e)
        self.b.insert(idx, b)

    def __getstate__(self):
        return [np2bytes(self.vals), np2bytes(self.e), np2bytes(self.b)]

    def __setstate__(self, state):
        self.vals = bytes2np(state[0]).tolist()
        self.e = bytes2np(state[1]).tolist()
        self.b = bytes2np(state[2]).tolist()

    def pickle_from(self, b):
        # Special-case for potentially frequent "all":
        if b == 0:
            return pickle.dumps([np2bytes(self.vals), np2bytes(self.e), np2bytes(self.b)])

        # Since most queries are expected to be about the latest stats,
        # we start searching from the end instead of binary searching.
        # This may be optimized even more (at the expense of more code)
        # by jumping to (b/self.b[-1])*len(self.b) and searching.
        idx = 0
        for idx in range(len(self.b), 0, -1):
            if self.b[idx-1] <= b:
                break
        return pickle.dumps([
            np2bytes(self.vals[idx:]),
            np2bytes(self.e[idx:]),
            np2bytes(self.b[idx:])
        ])

    def update_from_pickle(self, pick):
        vals, e, b = map(bytes2np, pickle.loads(pick))

        # Nothing new fast-lane.
        if len(b) == 0:
            return

        # Similar to the above, but we need to get to the first of `b`s.
        idx = 0
        for idx in range(len(self.b), -1, -1):
            if idx == 0 or self.b[idx-1] < b[0]:
                break
        self.vals = self.vals[:idx] + list(vals)
        self.e = self.e[:idx] + list(e)
        self.b = self.b[:idx] + list(b)


class Monitor:
    def __init__(self):
        self._online_losses = []
        # `e` is len of the above, `b` is sum of lens of the above.
        self._stats = defaultdict(Stat)

    def elosses(self, losses):
        """
        Store a list of all the losses of the current epoch.
        Also advances the epoch by one!
        """
        self._online_losses.append(losses)

    def loss(self, loss):
        """
        Append one batch-loss to the current epoch's list of losses.
        """
        if self._e() == 0:  # Special case for the epochless use-case.
            self.epoch()
        self._online_losses[-1].append(loss)

    def epoch(self):
        """
        Switches to the next epoch. Only call when using `bloss` as
        `elosses` does this automatically.
        """
        self._online_losses.append([])

    def stat(self, name, val):
        """
        Record `val` for the statistic `name` at the current point in time.
        """
        self._stats[name].record(val, self._e(), self._b())

    def _e(self):
        return len(self._online_losses)

    def _b(self):
        # TODO: optimize, maybe
        return sum(map(len, self._online_losses))

    def _b_in_e(self):
        return len(self._online_losses[-1]) if self._online_losses else 0

    def stats(self, *names):
        """Special shortcut syntax, now just for my comfort!

        Adds elements of the last argument using names of the formers.
        Best explained by example:

        stats("a", "b", "c", (1,2,3))

        corresponds to stats("a", 1) ; stats("b", 2) ; stats("c", r)
        """
        assert len(names)-1 == len(names[-1])
        for name, val in zip(names[:-1], names[-1]):
            self.bstat(name, val)

    def __getitem__(self, stat):
        return self._stats[stat]


try:
    import nnpy

    def name2b(name):
        # TODO: Warn!?
        return name.replace("/", "-").encode("utf-8")


    class RWMonitor(Monitor):
        def __init__(self, project, desc):
            super(RWMonitor, self).__init__()

            self.prefix = name2b(project) + b"/" + name2b(desc) + b"/"

        def _fmt_start(self):
            return self.prefix + b"start/0"

        # Send/receive updates
        ###

        def _fmt_loss(self, e, b_in_e, loss):
            return self.prefix + b'loss/' + struct.pack('IIf', e, b_in_e, loss)

        def _parse_loss(self, msgbody):
            e, b_in_e, loss = struct.unpack('IIf', msgbody)
            return e, b_in_e, loss

        def _recv_loss(self, e, b_in_e, loss):
            # NOTE: this method fills up potentially empty entries
            #       because messages might arrive out-of-order!
            e = e-1  # Epochs start counting at 1, arrays at 0

            while len(self._online_losses) <= e:
                self._online_losses.append([])
            while len(self._online_losses[e]) <= b_in_e:
                self._online_losses[e].append(np.nan)
            self._online_losses[e][b_in_e] = loss

        def _fmt_elosses(self, e, losses):
            e = str(e).encode('utf-8')
            return self.prefix + b'losses/' + e + b'/' + np2bytes(losses)

        def _parse_elosses(self, msgbody):
            e, losses = msgbody.split(b'/', 1)
            return int(e), bytes2np(losses)

        def _recv_elosses(self, e, losses):
            # NOTE: this method fills up potentially empty entries
            #       because messages might arrive out-of-order!
            while len(self._online_losses) <= e:
                self._online_losses.append([])
            self._online_losses[e] = losses

        def _fmt_stat(self, name, e, b, val):
            # TODO: Could optimize for space if `isinstance(val, Real)`
            name = name2b(name)
            e = str(e).encode('utf-8')
            b = str(b).encode('utf-8')
            return self.prefix + b'/'.join([name, e, b, np2bytes(val)])

        def _parse_stat(self, name, msgbody):
            e, b, val = msgbody.split(b'/', 2)
            return name.decode('utf-8'), int(e), int(b), bytes2np(val)

        def _recv_stat(self, name, e, b, val):
            self._stats[name].record(val, e, b)

        def recv(self, msg, body):
            if msg == b'loss':
                self._recv_loss(*self._parse_loss(body))
            elif msg == b'losses':
                self._recv_elosses(*self._parse_elosses(body))
            else:
                self._recv_stat(*self._parse_stat(msg, body))

        # Query information
        ###

        def query(self, query):
            if b'/' not in query:
                query += b"/1" # Epochs start at 1. But it doesn't matter.

            name, from_ = query.split(b'/')
            from_ = max(0, int(from_)-1)  # Epochs start at 1 but array indices at 0!

            if name == b"losses":
                return pickle.dumps([np2bytes(l) for l in self._online_losses[from_:]])
            else:
                stat = self._stats[name.decode('utf-8')]
                return stat.pickle_from(from_)


    class ReportMonitor(RWMonitor):
        def __init__(self, project, desc, addr="tcp://127.0.0.1:1337"):
            super(ReportMonitor, self).__init__(project, desc)

            self.sock = nnpy.Socket(nnpy.AF_SP, nnpy.PUB)
            self.sock.connect(addr)

            # TODO: Send message to archive existing (with date?) and start new.
            self.sock.send(self._fmt_start())

        def loss(self, loss):
            # TODO: optimize, maybe batch.
            msg = self._fmt_loss(self._e(), self._b_in_e(), loss)
            Monitor.loss(self, loss)
            self.sock.send(msg)

        def elosses(self, losses):
            msg = self._fmt_elosses(self._e(), losses)
            Monitor.elosses(self, losses)
            self.sock.send(msg)

        def epoch(self):
            super(ReportMonitor, self).epoch()
            # TODO: Possibly actually send something here!

        def stat(self, name, val):
            msg = self._fmt_stat(name, self._e(), self._b(), val)
            Monitor.stat(self, name, val)
            self.sock.send(msg)


    class WatchMonitor(RWMonitor):
        def __init__(self, project, desc, addr="tcp://127.0.0.1:31337", timeoutms=-1):
            super(WatchMonitor, self).__init__(project, desc)

            self.sock = nnpy.Socket(nnpy.AF_SP, nnpy.REQ)
            self.sock.connect(addr)

            self.sock.setsockopt(nnpy.SOL_SOCKET, nnpy.RCVTIMEO, timeoutms)
            self.sock.setsockopt(nnpy.SOL_SOCKET, nnpy.SNDTIMEO, timeoutms)
            self.sock.setsockopt(nnpy.SOL_SOCKET, nnpy.RCVMAXSIZE, -1)

        def update(self):
            # Ask for losses of currently last known epoch and following if any.
            # TODO: Might not be optimal in the epochless case!
            e = str(self._e()).encode('utf-8')
            try:
                self.sock.send(self.prefix + b'losses/' + e)
                losses = pickle.loads(self.sock.recv())
            except AssertionError:
                if nnpy.nanomsg.nn_errno() == nnpy.ETIMEDOUT:
                    return
                raise
            if not self._online_losses:
                self._online_losses = [bytes2np(l) for l in losses]
            else:
                if losses:
                    self._online_losses[-1] = bytes2np(losses[0])
                for l in losses[1:]:
                    self._online_losses.append(bytes2np(l))

            for name, stat in self._stats.items():
                try:
                    from_ = stat.b[-1] if stat.b else 0
                    self.sock.send(self.prefix + name2b(name) + b'/' + str(from_).encode('utf-8'))
                    stat.update_from_pickle(self.sock.recv())
                except AssertionError:
                    if nnpy.nanomsg.nn_errno() == nnpy.ETIMEDOUT:
                        return
                    raise

        def stat(self, name):
            # Mark as "we want to request it on next update"
            return self._stats[name]

        def stats(self, *names):
            return tuple(map(self.stat, names))

        # TODO: Given the following, it may make sense not to subclass?
        def elosses(self, *a, **kw):
            raise NotImplementedError("WatchMonitor can only watch!!")

        def loss(self, *a, **kw):
            raise NotImplementedError("WatchMonitor can only watch!!")

        def epoch(self, *a, **kw):
            raise NotImplementedError("WatchMonitor can only watch!!")

except ImportError:
    print("Warning: no nnpy found!")
