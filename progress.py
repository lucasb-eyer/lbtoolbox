#!/usr/bin/env python

import uuid

# Currently, this is ipy only!
from IPython.display import HTML, Javascript, display, clear_output

try:  # Py >= 3.3
    from time import monotonic as monotime
except ImportError:
    from time import time as monotime

class Progressbar(object):
    code = """
        <div style="position: relative; border: 1px solid black; width:500px; text-align:center">
          <div id="{id}fill" style="background-color:#6BD1FB; width:{pct:.2%};">&nbsp;</div>
          <div id="{id}text" style="position: absolute; top: 0; left: 0; right: 0;">{pct:.2%}</div>
        </div>
        """
    upcode = "$('div#{id}fill').width('{pct:.2%}'); $('div#{id}text').text('{pct:.2%}');"

    # Set clearing=False if your output also contains other IPython display objects.
    # Set clearing=True if you'll have a huge number of updates over a long period of time.
    # Set clearing=True to decrease memory usage.
    # Increase mindt to decrease CPU load.
    def __init__(self, a=1.0, b=None, mindt=1./30, clearing=True):
        self.divid = str(uuid.uuid4())
        self.zero = a if b is not None else 0.0
        self.hundred = b if b is not None else a
        self.curr = self.zero  # For relative update
        self.clearing = clearing
        self.t = 0
        self.mindt = mindt
        if not clearing:
            display(HTML(Progressbar.code.format(id=self.divid, pct=0)))
        self.update(self.curr)

    def update(self, val=None):
        if val is None:
            self.curr += 1
            val = self.curr

        # Do not update too often, or the browsers will go crazy!
        # Except for the last update.
        t = monotime()
        if abs(self.hundred - val) > 1e-4 and t - self.t < self.mindt:
            return
        self.t = t

        v = float(val - self.zero)/(self.hundred - self.zero)
        # Better than the js stuff, since the js stuff keeps adding and thus
        # makes the browser insanely slow if there are >> 1k updates.
        if self.clearing:
            #clear_output(stdout=False, stderr=False, other=True)
            clear_output(wait=True)
            display(HTML(Progressbar.code.format(id=self.divid, pct=v)))
        else:
            display(Javascript(Progressbar.upcode.format(id=self.divid, pct=v)))
