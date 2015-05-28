#!/usr/bin/env python3

import time
import sys

import numpy as np

from lbtoolbox.util import printnow
from lbtoolbox.thutil import save_model, load_model
from lbtoolbox.plotting import plot_training, plot_cost


class SimpleTrainer(object):

    def __init__(self, optim, scorer, loadfrom=None, postepoch=None, progressout=sys.stdout):
        self.optim = optim
        self.scorer = scorer
        self.postepoch = postepoch or _default_printprogress
        self.progressout = progressout

        if not loadfrom:
            self.trnlls, self.trcosts, self.trepochs, self.trts = [], [], [], []
            self.vanlls, self.vaerrs, self.vaepochs, self.vats = [], [], [], []
            self.bestva = float('inf')
            self.bestvanll = float('inf')
            self.ebest = 0
            self.e = 0
            self.t = 0
        else:
            if not loadfrom.endswith(".npz"):
                loadfrom += ".npz"

            with np.load(loadfrom) as f:
                self.trnlls, self.trcosts, self.trepochs, self.trts = list(f['trnlls']), list(f['trcosts']), list(f['trepochs']), list(f['trts'])
                self.vanlls, self.vaerrs, self.vaepochs, self.vats = list(f['vanlls']), list(f['vaerrs']), list(f['vaepochs']), list(f['vats'])

                self.bestva = min(self.vaerrs)
                self.bestvanll = min(self.vanlls)
                self.ebest = np.argmin(self.vaerrs)
                self.t = f['t']


    def fit(self, Xtr, ytr, Xva, yva, epochs,
            lr0,
            lr_decay=None,
            aug=None,
            min_nll_diff=1e-3,
            valid_freq=1,
            full_valid_freq=1,
            savename='/tmp/model',
            shuf=None):

        # This is quite a small hack for percents in the default post-epoch reporting.
        self.last_yva_len = len(yva)

        lr = lr0
        tstart = time.clock()
        try:
          for e in range(epochs):
            # This arrangement is a bit weird.

            # We first validate (and potentially test), then train.
            # This is because `train` computes the nll+cost on the trainset
            # for free as a side-product of training. But it computes the
            # value of it before the weight update.

            do_valid = False
            if self.e % valid_freq == 0:
                do_valid = True
                fast = True
            if self.e % full_valid_freq == 0:
                do_valid = True
                fast = False

            if do_valid:
                t0 = time.clock()
                nll, err = self.valid_epoch(Xva, yva, aug=aug, fast=fast)
                t1 = time.clock()

                # Store for posteriority.
                self.vanlls.append(nll)
                self.vaerrs.append(err)
                self.vaepochs.append(self.e)
                self.vats.append(t1-t0)

            # Save the model that performs best on the validation set.
            # Best is: "significantly" lower nll.
            if self.vanlls[-1] < self.bestvanll - min_nll_diff:
                self.ebest = self.e
                self.bestva = self.vaerrs[-1]
                self.bestvanll = self.vanlls[-1]
                save_model(self.optim.model, savename)

            t0 = time.clock()
            kw = dict(aug=aug, shuf=shuf)
            if lr is not None:
                kw['lrate'] = lr
            cost, nll = self.fit_epoch(Xtr, ytr, **kw)
            t1 = time.clock()

            # Store for posteriority.
            self.trnlls.append(nll)
            self.trcosts.append(cost)
            self.trepochs.append(self.e)
            self.trts.append(t1-t0)

            # Call whatever is registered as post-epoch callback.
            self.postepoch(self)

            # And only now did we advance an epoch.
            # That is because the cost and nll computed by the training step
            # are those *before* updating the gradient.
            self.e += 1

            if lr is not None and lr_decay is not None:
                lr *= lr_decay

        except KeyboardInterrupt:
            # Just stop the loop on keyboard interrupt.
            # But also reload the best model we had.
            printnow(self.progressout, "Interrupted.")
            if True: #input("Reload best model? [y/n]") == 'y':
                load_model(self.optim.model, savename)
                printnow(self.progressout, "Reloaded!")

        tend = time.clock()
        self.t += tend-tstart

        printnow(self.progressout, 'Best in validation@{ebest}e ({t:.1f}s in total, i.e. {spe:.1f}s/e)\n',
            ebest = self.ebest,
            t = tend - tstart,
            spe = (tend - tstart)/e
        )


    def fit_epoch(self, X, y, **kw):
        fitret = self.optim.fit_epoch(X, y, **kw)
        self.optim.finalize(X, y, **kw)
        return fitret


    def valid_epoch(self, X, y, **kw):
        return self.scorer(X, y, **kw)


    def report_training(self, start=None, end=None, nva=None):
        start = start if start is not None else 0
        end = end if end is not None else self.e+1
        epochs = []
        for e in range(start, end):
            try:
                etr = self.trepochs.index(e)
                s_tr = _report_train(self.trcosts[etr], self.trnlls[etr], self.trts[etr])
            except ValueError:
                s_tr = None
            try:
                eva = self.vaepochs.index(e)
                s_va = _report_score(self.vanlls[eva], self.vaerrs[eva], self.vats[eva], nva)
            except ValueError:
                s_va = None
            epochs.append(_report_epoch(e, s_tr, s_va))
        return '\n'.join(epochs)


    def plot_training(self, mistakes=True, **kwargs):
        # We skip the 0-th epoch in the plots.
        return plot_training(valid_errs=self.vaerrs[1:], valid_epochs=self.vaepochs[1:], **kwargs)


    def plot_cost(self, **kwargs):
        # We skip the 0-th epoch in the plots.
        return plot_cost(train_costs=self.trnlls[1:], valid_costs=self.vanlls[1:],
                         train_epochs=self.trepochs[1:], valid_epochs=self.vaepochs[1:], **kwargs)


    def save_training(self, fname):
        if not fname.endswith(".npz"):
            fname += ".npz"

        np.savez(fname,
            trnlls=self.trnlls, trcosts=self.trcosts, trepochs=self.trepochs, trts=self.trts,
            vanlls=self.vanlls, vaerrs=self.vaerrs, vaepochs=self.vaepochs, vats=self.vats,
            t=self.t,
        )


def _report_train(cost, nll, t):
    s = 'train cost: {:.3f}, NLL: {:.3f}'.format(cost, nll)
    if t is not None:
        s += ' ({:.1f}s)'.format(t)
    return s


def _report_score(nll, err, t, n):
    s = 'NLL: {:.3f}'.format(nll)
    if n is not None:
        s += ', err: {:=6.2%}'.format(err/n)
    else:
        s += ', err: {}'.format(err)
    if t is not None:
        s += ' ({:.1f}s)'.format(t)
    return s


def _report_epoch(e, s_tr, sva, isbest):
    s = '{e:=3}: '.format(e=e)
    s += s_tr or 'No training'
    if sva is not None:
        s += ', val ' + sva
    if isbest:
        s += '**'
    return s


def _default_printprogress(self):
    # This just formats all of them nicely together and prints it.
    s_tr = _report_train(self.trcosts[-1], self.trnlls[-1], self.trts[-1])
    s_va = _report_score(self.vanlls[-1], self.vaerrs[-1], self.vats[-1], self.last_yva_len) if self.vaepochs[-1] == self.e else None
    printnow(self.progressout, "{}\n", _report_epoch(self.e, s_tr, s_va, self.e == self.ebest))
