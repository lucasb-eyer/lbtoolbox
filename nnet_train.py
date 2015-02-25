#!/usr/bin/env python3

import time
import sys

from lbtoolbox.thutil import save_model, load_model
from lbtoolbox.plotting import plot_training, plot_cost

class Trainer(object):

    def __init__(self, optim):
        self.optim = optim
        self.trnlls, self.trcosts, self.trepochs = [], [], []
        self.vanlls, self.vaerrs, self.vaepochs = [], [], []
        self.tenlls, self.teerrs, self.teepochs = [], [], []
        self.bestva = float('inf')
        self.bestvanll = float('inf')
        self.ebest = 0
        self.echeck = 0
        self.e = 0


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
                print("Best model reloaded from {e} in {t:.2f}s; lr decayed for the {n}th time, to {lr}".format(e=self.ebest, t=t1-t0, n=nreducs, lr=lr))
                self.echeck = self.e - recover_patience - 1

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

            s = '{e:=3}: '.format(e=self.e)
            s += self.report_train(cost, nll, t1 - t0)
            if sva is not None:
                s += ', val ' + sva
            if ste is not None:
                s += ', test ' + ste
            if self.ebest == self.e:
                s += '**'
            print(s)
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


    def plot_training(self, mistakes=True, **kwargs):
        # We skip the 0-th epoch in the plots.
        return plot_training(None, self.vaerrs[1:], self.teerrs[1:] if mistakes == True else mistakes,
                             None, self.vaepochs[1:], self.teepochs[1:], **kwargs)


    def plot_cost(self, cost=True, **kwargs):
        # We skip the 0-th epoch in the plots.
        return plot_cost(self.trnlls[1:], self.vanlls[1:], self.tenlls[1:] if cost == True else cost,
                         self.trepochs[1:], self.vaepochs[1:], self.teepochs[1:], **kwargs)
