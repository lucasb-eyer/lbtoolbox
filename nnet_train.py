#!/usr/bin/env python3

import time
import sys

import numpy as np

from lbtoolbox.thutil import save_model, load_model
from lbtoolbox.plotting import plot_training, plot_cost

class Trainer(object):

    def __init__(self, optim, loadfrom=None):
        self.optim = optim

        if not loadfrom:
            self.trnlls, self.trcosts, self.trepochs, self.trts = [], [], [], []
            self.vanlls, self.vaerrs, self.vaepochs, self.vats = [], [], [], []
            self.tenlls, self.teerrs, self.teepochs, self.tets = [], [], [], []
            self.bestva = float('inf')
            self.bestvanll = float('inf')
            self.ebest = 0
            self.echeck = 0
            self.e = 0
            self.t = 0
        else:
            if not loadfrom.endswith(".npz"):
                loadfrom += ".npz"

            with np.load(loadfrom) as f:
                self.trnlls, self.trcosts, self.trepochs, self.trts = list(f['trnlls']), list(f['trcosts']), list(f['trepochs']), list(f['trts'])
                self.vanlls, self.vaerrs, self.vaepochs, self.vats = list(f['vanlls']), list(f['vaerrs']), list(f['vaepochs']), list(f['vats'])
                self.tenlls, self.teerrs, self.teepochs, self.tets = list(f['tenlls']), list(f['teerrs']), list(f['teepochs']), list(f['tets'])

                self.bestva = min(self.vaerrs)
                self.bestvanll = min(self.vanlls)
                self.ebest = np.argmin(self.vaerrs)
                self.echeck = self.e = self.trepochs[-1]
                self.t = f['t']


    def fit(self, Xtr, ytr, Xva, yva, Xte=None, yte=None,
            lr0=1,
            aug=None,
            patience=10,
            recover_patience=5,
            lr_decrease=10,
            lr_decay=None,
            max_reducs=4,
            min_nll_diff=1e-3,
            max_epochs=1000,
            valid_freq=1,
            test_freq=1,
            skip_initial=False,
            skip_final=False,
            savename='/tmp/model'):
        lr = lr0
        nreducs = 0
        tstart = time.clock()
        try:
          for e in range(max_epochs):
            # This arrangement is a bit weird.

            # We first validate (and potentially test), then train.
            # This is because `train` computes the nll+cost on the trainset
            # for free as a side-product of training. But it computes the
            # value of it before the weight update.

            if self.e % valid_freq == 0:
                t0 = time.clock()
                nll, err = self.valid_epoch(Xva, yva, aug=aug)
                t1 = time.clock()

                # Store for posteriority.
                self.vanlls.append(nll)
                self.vaerrs.append(err)
                self.vaepochs.append(self.e)
                self.vats.append(t1-t0)
                sva = self.report_score(nll, err, t1 - t0, len(yva))
            else:
                sva = None

            if self.e % test_freq == 0 and Xte is not None and test_freq is not None and not (skip_initial and e == 0):
                t0 = time.clock()
                nll, err = self.test_epoch(Xte, yte, aug=aug)
                t1 = time.clock()

                # Store for posteriority.
                self.tenlls.append(nll)
                self.teerrs.append(err)
                self.teepochs.append(self.e)
                self.tets.append(t1-t0)
                ste = self.report_score(nll, err, t1 - t0, len(yte))
            else:
                ste = None

            # Save the model that performs best on the validation set.
            # Best is: fewer errors or same errors but significantly lower nll.
            if self.vaerrs[-1] < self.bestva or (self.vaerrs[-1] == self.bestva and self.vanlls[-1] < self.bestvanll - min_nll_diff):
                self.bestva = self.vaerrs[-1]
                self.bestvanll = self.vanlls[-1]
                self.echeck = self.ebest = self.e
                save_model(self.optim.model, savename)

            # If we're not getting better for some time, reload the best
            # model so far and decrease learning-rate.
            if self.e - self.echeck > patience:
                lr /= lr_decrease
                nreducs += 1
                t0 = time.clock()
                load_model(self.optim.model, savename)
                t1 = time.clock()
                print("Best model reloaded from {e} in {t:.2f}s; lr decayed for the {n}. time, to {lr}".format(e=self.ebest, t=t1-t0, n=nreducs, lr=lr))
                self.echeck = self.e - (patience-recover_patience) - 1

            # If we decayed enough, it means we're not going to get
            # any better, really.
            if nreducs > max_reducs:
                print("Learning-rate dropped {} times. I just lost all my patience.".format(nreducs))
                break

            t0 = time.clock()
            cost, nll = self.fit_epoch(Xtr, ytr, lr, aug=aug)
            t1 = time.clock()

            # Store for posteriority.
            self.trnlls.append(nll)
            self.trcosts.append(cost)
            self.trepochs.append(self.e)
            self.trts.append(t1-t0)
            s_tr = self.report_train(cost, nll, t1 - t0)

            # This just formats all of them nicely together and prints it.
            print(self.report_epoch(self.e, s_tr, sva, ste))
            sys.stdout.flush()

            # And only now did we advance an epoch.
            # That is because the cost and nll computed by the training step
            # are those *before* updating the gradient.
            self.e += 1

            if lr_decay is not None:
                lr *= lr_decay

        except KeyboardInterrupt:
            # Just stop the loop on keyboard interrupt.
            # But also reload the best model we had.
            print("Interrupted. Reloading best model and stopping.")
            sys.stdout.flush()
            load_model(self.optim.model, savename)

        tend = time.clock()
        self.t += tend-tstart

        print('Best in validation@{ebest}e ({t:.1f}s in total, i.e. {spe:.1f}s/e)'.format(
            ebest = self.ebest,
            t = tend - tstart,
            spe = (tend - tstart)/e
        ))
        sys.stdout.flush()

        if Xte is not None and not skip_final:
            t0 = time.clock()
            nll, err = self.test_epoch(Xte, yte, aug=aug)
            t1 = time.clock()
            print('Final test', self.report_score(nll, err, t1-t0, len(yte)), ' i.e. {:.2%} score'.format(1 - err/len(yte)))
            sys.stdout.flush()


    def fit_epoch(self, X, y, lr, aug):
        return self.optim.fit_epoch(X, y, lr, aug)


    def valid_epoch(self, X, y, aug):
        return self.optim.score_epoch(X, y, aug, fast=True)


    def test_epoch(self, X, y, aug):
        return self.optim.score_epoch(X, y, aug, fast=False)


    def report_train(self, cost, nll, t):
        s = 'train cost: {:.3f}, NLL: {:.3f}'.format(cost, nll)
        if t is not None:
            s += ' ({:.1f}s)'.format(t)
        return s


    def report_score(self, nll, err, t, n):
        s = 'NLL: {:.3f}'.format(nll)
        if n is not None:
            s += ', err: {:=6.2%}'.format(err/n)
        else:
            s += ', err: {}'.format(err)
        if t is not None:
            s += ' ({:.1f}s)'.format(t)
        return s


    def report_epoch(self, e, s_tr, sva, ste):
        s = '{e:=3}: '.format(e=e)
        s += s_tr or 'No training'
        if sva is not None:
            s += ', val ' + sva
        if ste is not None:
            s += ', test ' + ste
        if self.ebest == e:
            s += '**'
        return s


    def report_training(self, start=None, end=None, nva=None, nte=None):
        start = start if start is not None else 0
        end = end if end is not None else self.e+1
        epochs = []
        for e in range(start, end):
            try:
                etr = self.trepochs.index(e)
                s_tr = self.report_train(self.trcosts[etr], self.trnlls[etr], self.trts[etr])
            except ValueError:
                s_tr = None
            try:
                eva = self.vaepochs.index(e)
                s_va = self.report_score(self.vanlls[eva], self.vaerrs[eva], self.vats[eva], nva)
            except ValueError:
                s_va = None
            try:
                ete = self.teepochs.index(e)
                s_te = self.report_score(self.tenlls[ete], self.teerrs[ete], self.tets[ete], nte)
            except ValueError:
                s_te = None
            epochs.append(self.report_epoch(e, s_tr, s_va, s_te))
        return '\n'.join(epochs)


    def plot_training(self, mistakes=True, **kwargs):
        # We skip the 0-th epoch in the plots.
        return plot_training(None, self.vaerrs[1:], self.teerrs[1:] if mistakes == True else mistakes,
                             None, self.vaepochs[1:], self.teepochs[1:], **kwargs)


    def plot_cost(self, cost=True, **kwargs):
        # We skip the 0-th epoch in the plots.
        return plot_cost(self.trnlls[1:], self.vanlls[1:], self.tenlls[1:] if cost == True else cost,
                         self.trepochs[1:], self.vaepochs[1:], self.teepochs[1:], **kwargs)

    def save_training(self, fname):
        if not fname.endswith(".npz"):
            fname += ".npz"

        np.savez(fname,
            trnlls=self.trnlls, trcosts=self.trcosts, trepochs=self.trepochs, trts=self.trts,
            vanlls=self.vanlls, vaerrs=self.vaerrs, vaepochs=self.vaepochs, vats=self.vats,
            tenlls=self.tenlls, teerrs=self.teerrs, teepochs=self.teepochs, tets=self.tets,
            t=self.t,
        )
