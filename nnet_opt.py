#!/usr/bin/env python3

import theano as th
import theano.tensor as T
import numpy as np

from lbtoolbox.util import batched_padded


class BatchGradDescent(object):

    def __init__(self, model, nllclip=(1e-15, 1-1e-15)):
        # Keep in mind for later.
        self.model = model
        self.nllclip = nllclip

        # Compile theano functions for the various measures of a model's goodness.
        self.fn_pred_proba = th.function(inputs=[], outputs=model.pred_proba)
        self.fn_errors = th.function(inputs=[], outputs=model.errors)
        self.fn_cost = th.function(inputs=[], outputs=model.cost)
        self.fn_nll = th.function(inputs=[], outputs=model.nll)

        # And one for computing all of the above jointly, much faster.
        self.fn_eval = th.function(inputs=[], outputs=[model.nll, model.errors, model.pred_proba])

        # And finally the training part.
        # TODO: Could still output cost, as it's computed anyways...
        # Fuck transfercost of 1 number?
        self.sh_learningrate = T.scalar('lrate')
        g = T.grad(cost=model.cost, wrt=model.params)
        self.fn_train = th.function(
            inputs=[self.sh_learningrate],
            outputs=[model.cost, model.nll],
            updates=[(p, p - self.sh_learningrate * gp) for p, gp in zip(model.params, g)]
        )


    def fit_epoch(self, X, y, learning_rate, aug=None):
        costs, nlls = [], []
        for _, bx, by in batched_padded(self.model.batchsize, X, y):
            if aug:
                bx = aug.augbatch_train(bx)
            self.model.sh_x.set_value(th.sandbox.cuda.CudaNdarray(bx))
            self.model.sh_y.set_value(by)
            cost, nll = self.fn_train(learning_rate)
            costs.append(cost)
            nlls.append(nll)
        return sum(costs)/len(costs), sum(nlls)/len(nlls)


    def score_epoch(self, X, y, aug=None, fast=False):
        nlls = []
        errs = 0
        # Go through the training in minibatches.
        # As long as there's https://github.com/Theano/Theano/issues/2464
        # we need to pad the last batch.
        for bs, bx, by in batched_padded(self.model.batchsize, X, y):
            if aug:
                # In the case of augmentation, average all outputs of the
                # augmenters. ("Return of the Devil in the Details", Chatfield, Simonyan, Vevaldi & Zisserman)
                ppreds = []
                for bx_aug in aug.augbatch_pred(bx, fast):
                    self.model.sh_x.set_value(th.sandbox.cuda.CudaNdarray(bx_aug))
                    # Due to the same bug as above, we need to cut off the
                    # padded part of the last batch.
                    ppreds.append(self.fn_pred_proba()[:bs])
                p_y_given_x = sum(ppreds)/len(ppreds)
                nll = -np.mean(np.log(np.clip(p_y_given_x[np.arange(bs), by[:bs]], *self.nllclip)))
                err = np.sum(np.argmax(p_y_given_x[:bs], axis=1) != by[:bs])
            else:
                self.model.sh_x.set_value(th.sandbox.cuda.CudaNdarray(bx))
                self.model.sh_y.set_value(by)
                nll, err, pred = self.fn_eval()
            nlls.append(nll)
            errs += err
        return sum(nlls)/len(nlls), errs

