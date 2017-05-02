from time import time
from collections import OrderedDict


class Chrono:
    def __init__(self):
        self.timings = OrderedDict()

    def measure(self, what):
        return Timer(lambda t: self._done(what, t))

    def _done(self, what, t):
        self.timings.setdefault(what, []).append(t)

    def times(self, what):
        return self.timings[what]

    def avgtime(self, what, dropfirst=False):
        timings = self.timings[what]
        if dropfirst and len(timings) > 1:
            timings = timings[1:]
        return sum(timings)/len(timings)

    def __str__(self, fmt='{:{w}.5f}', dropfirst=False):
        avgtimes = {k: self.avgtime(k, dropfirst) for k in self.timings}
        l = max(map(len, avgtimes))
        w = max(len(fmt.format(v, w=0)) for v in avgtimes.values())
        return '\n'.join(("{:{l}s}: "+fmt+"s").format(k,v,l=l,w=w) for k,v in sorted(avgtimes.items(), key=lambda t: t[1], reverse=True))


class Timer:
    def __init__(self, donecb):
        self.cb = donecb

    def __enter__(self):
        self.t0 = time()

    def __exit__(self, exc_type, exc_value, traceback):
        t = time() - self.t0
        self.cb(t)


class PrintTimer(Timer):
    def __init__(self, msg, *a, **kw):
        Timer.__init__(self, lambda t: print("Done in {:.3g}s".format(t), flush=True))
        print(msg.format(*a, **kw), end='', flush=True)
