#!/usr/bin/env python3

import theano as th
import theano.tensor as T
import numpy as np

from lbtoolbox.util import batched_padded


class MinibatchOptimizer(object):

    def __init__(self, model, nllclip):
        # Keep in mind for later.
        self.model = model
        self.nllclip = nllclip

        # Compile theano functions for the various measures of a model's goodness.
        self.fn_pred_proba = th.function(inputs=[], outputs=model.pred_proba)
        self.fn_errors = th.function(inputs=[], outputs=model.errors)
        self.fn_cost = th.function(inputs=[], outputs=model.cost)
        self.fn_nll = th.function(inputs=[], outputs=model.nll)

        # And one for computing all of the above jointly, much faster.
        # Except cost, which is practically useless for evaluation.
        self.fn_eval = th.function(inputs=[], outputs=[model.nll, model.errors, model.pred_proba])

        # The training function is what changes throughout different methods,
        # so the specific subclasses will need to compile that.
        # They all share the fact of having a learning-rate, though.
        self.sh_learningrate = T.scalar('lrate')


    def fit_epoch(self, X, y, learning_rate, aug=None):
        costs, nlls = [], []
        # Go through the training in minibatches.
        # As long as there's https://github.com/Theano/Theano/issues/2464
        # we need to pad the last batch.
        for _, bx, by in batched_padded(self.model.batchsize, X, y):
            # Potentially generate a new augmentation on-the-fly.
            if aug:
                bx = aug.augbatch_train(bx)

            # Upload the minibatch to the GPU.
            self.model.sh_x.set_value(th.sandbox.cuda.CudaNdarray(bx))
            self.model.sh_y.set_value(by)

            # Do the forward *and* backward pass.
            cost, nll = self.fn_train(learning_rate)

            # Collect stats over the batches, so we can average.
            costs.append(cost)
            nlls.append(nll)

        # Average the stats over the batches.
        return sum(costs)/len(costs), sum(nlls)/len(nlls)


    def score_epoch(self, X, y, aug=None, fast=False):
        nlls = []
        errs = 0
        # Go through the training in minibatches.
        # As long as there's https://github.com/Theano/Theano/issues/2464
        # we need to pad the last batch.
        for bs, bx, by in batched_padded(self.model.batchsize, X, y):
            # For scoring, augmentation or not makes a big difference:
            if aug:
                # With augmentation, the model will be evaluated on potentially
                # many augmented versions of each batch and we need to average
                # the output class-probabilities of all those runs.
                # See "Return of the Devil in the Details" for details.
                ppreds = []
                for bx_aug in aug.augbatch_pred(bx, fast):
                    self.model.sh_x.set_value(th.sandbox.cuda.CudaNdarray(bx_aug))
                    # Due to the same bug as above, we need to cut off the
                    # padded part of the last batch.
                    ppreds.append(self.fn_pred_proba()[:bs])
                p_y_given_x = sum(ppreds)/len(ppreds)
                # TODO: Most of this actually belongs into the costlayer's
                #       class, since it's even very classification-specific!
                # Since we have the NLL on the CPU, we'll have to compute it here.
                nll = -np.mean(np.log(np.clip(p_y_given_x[np.arange(bs), by[:bs]], *self.nllclip)))
                err = np.sum(np.argmax(p_y_given_x[:bs], axis=1) != by[:bs])
            else:
                # While without augmentation, it's the straightforward thing.
                self.model.sh_x.set_value(th.sandbox.cuda.CudaNdarray(bx))
                self.model.sh_y.set_value(by)
                nll, err, pred = self.fn_eval()
            # Again, collect over batches...
            nlls.append(nll)
            errs += err
        # ...so we can return average nll over batches, and total errors.
        return sum(nlls)/len(nlls), errs


class MiniSGD(MinibatchOptimizer):

    def __init__(self, model, nllclip=(1e-15, 1-1e-15)):
        super(MiniSGD, self).__init__(model, nllclip)

        # For SGD, training is quite simple:
        # p_e+1 = p_e - lr * grad(p_e)
        g = T.grad(cost=model.cost, wrt=model.params)
        self.fn_train = th.function(
            inputs=[self.sh_learningrate],
            outputs=[model.cost, model.nll],
            updates=[(p, p - self.sh_learningrate * gp) for p, gp in zip(model.params, g)]
        )


class MiniMomentum(MinibatchOptimizer):

    def __init__(self, model, momentum, nllclip=(1e-15, 1-1e-15)):
        super(MiniMomentum, self).__init__(model, nllclip)

        # For momentum, we need a "mirror" of each parameter, which keeps track
        # of the "velocity" of that parameter during training.
        self.sh_v = [
            th.shared(np.zeros(p.get_value().shape), name='v_'+p.name)
            for p in model.params
        ]

        g = T.grad(cost=model.cost, wrt=model.params)

        # For Momentum SGD, the following training equation comes from:
        # "On the importance of initialization and momentum in deep learning"
        # v_e+1 = mom*v_e - lr * grad(p_e)
        # p_e+1 = p_e + v_e+1

        updates = []
        for sh_p, gp, sh_v in zip(model.params, g, self.sh_v):
            v = momentum * sh_v - self.sh_learningrate * gp
            updates.append((sh_v, v))
            updates.append((sh_p, sh_p + v))

        self.fn_train = th.function(
            inputs=[self.sh_learningrate],
            outputs=[model.cost, model.nll],
            updates=updates
        )
