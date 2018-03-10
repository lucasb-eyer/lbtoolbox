import lbtoolbox.util as lbu

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole?
            return True
        elif shell == 'TerminalInteractiveShell':  # Terminal running IPython?
            return False
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


if isnotebook():
    from ipywidgets import IntProgress as _IntProgress
    from IPython.display import display
    ProgressBar = _IntProgress
else:
    import sys as _sys


    def display(pb):
        pb.render()


    class ProgressBar:
        def __init__(self, val=None, max=None, description=None):
            self._val = val
            self._max = max
            self._desc = description
            self._fmt = "{desc}, {v}/{max} ({pct:.2%})"

        @property
        def value(self):
            return self._val

        @value.setter
        def value(self, val):
            self._val = val
            self.render()

        @property
        def max(self):
            return self._max

        @max.setter
        def max(self, max):
            self._max = max
            self.render()

        @property
        def description(self):
            return self._desc

        @description.setter
        def description(self, desc):
            self._desc = desc
            self.render()

        def render(self):
            pct = float(self.value)/float(self.max) if self.value is not None and self.max is not None else float('NaN')
            _sys.stdout.write("\r" + self._fmt.format(desc=self.description, v=self.value, max=self.max, pct=pct))
            _sys.stdout.flush()


def update_progress(pb, val=None, max=None, description=None, inc=None):
    if pb is None or pb is False:
        return
    elif pb is True:
        pb = ProgressBar(val, max=max)
        display(pb)

    if val is not None:
        pb.value = val
    elif inc is not None:
        pb.value += inc

    if max is not None:
        pb.max = max

    if description is not None:
        if hasattr(pb, '_lucas_prefix'):
            pb.description = pb._lucas_prefix + description
        else:
            pb.description = description

    return pb


def progress(*iterables, tot=None, description=None, pb=True):
    # Find the length of any of them if possible.
    if tot is None:
        for i in iterables:
            try:
                tot = max(len(i), tot or 0.0)
            except TypeError:
                pass

    pb = update_progress(pb, val=0, max=tot, description=description)
    for things in zip(*iterables):
        yield lbu.maybetuple(things)
        update_progress(pb, inc=1)
