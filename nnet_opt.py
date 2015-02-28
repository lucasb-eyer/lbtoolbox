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
    """
    Implements both the "Classical Momentum (CM)" and "Nesterov's
    Accelerated Gradient (NAG)" which are explained in further detail in

    "On the importance of initialization and momentum in deep learning"

    But the equation for NAG has been reshuffled by Nicolas Boulanger in

    https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617

    for easier implementation in Theano.

    TL;DR: Nesterov allows for larger momentum to be used, making it better.
           Very finicky parameter-selection.
    """

    def __init__(self, model, momentum, nesterov=False, nllclip=(1e-15, 1-1e-15)):
        super(MiniMomentum, self).__init__(model, nllclip)

        # For momentum, we need a "mirror" of each parameter, which keeps track
        # of the "velocity" of that parameter during training.
        self.sh_v = [
            th.shared(np.zeros_like(p.get_value()), broadcastable=p.broadcastable, name='v_'+p.name)
            for p in model.params
        ]

        g = T.grad(cost=model.cost, wrt=model.params)

        updates = []
        for sh_p, gp, sh_v in zip(model.params, g, self.sh_v):
            v = momentum * sh_v - self.sh_learningrate * gp
            updates.append((sh_v, v))

            if not nesterov:
                updates.append((sh_p, sh_p + v))
            else:
                updates.append((sh_p, sh_p + momentum * v - self.sh_learningrate * gp))

        self.fn_train = th.function(
            inputs=[self.sh_learningrate],
            outputs=[model.cost, model.nll],
            updates=updates
        )


class MiniAdaGrad(MinibatchOptimizer):
    """
    Implements Duchi's "Adaptive Subgradient" method, aka AdaGrad.
    Chris Dyer's "Notes on AdaGrad" are pretty awesome for practical purposes.

    TL;DR: AdaGrad doesn't need additional parameters and makes the
           optimization much less sensitive to the learning-rate!

    (Well, it needs `eps`, but that one hardly matters at all.)
    """

    def __init__(self, model, eps=1e-5, nllclip=(1e-15, 1-1e-15)):
        super(MiniAdaGrad, self).__init__(model, nllclip)

        # Adagrad needs to accumulate the square gradient of each parameter.
        # I wonder if this won't explode at some point? Probably should fully
        # read the original paper!
        # Edit: Matt Zeiler seems to agree cf. AdaDelta.
        self.sh_g2 = [
            th.shared(eps*np.ones_like(p.get_value()), broadcastable=p.broadcastable, name='g2_'+p.name)
            for p in model.params
        ]

        g = T.grad(cost=model.cost, wrt=model.params)

        updates = []
        for sh_p, gp, sh_g2 in zip(model.params, g, self.sh_g2):
            g2 = sh_g2 + gp*gp
            updates.append((sh_g2, g2))
            updates.append((sh_p, sh_p - self.sh_learningrate/T.sqrt(g2) * gp))
            # Instead of adding eps inside the square-root like most
            # implementations do, I just initialize `g2` to eps, that should
            # have the same effect, but cheaper.

        self.fn_train = th.function(
            inputs=[self.sh_learningrate],
            outputs=[model.cost, model.nll],
            updates=updates
        )


class MiniAdaDelta(MinibatchOptimizer):
    """
    Implements Matt Zeiler's "Adaptive Learningrate" method, aka AdaDelta.
    The paper itself is really neat, and both very convincing and practical.

    TL;DR: 1. AdaGrad quickly anneals, AdaDelta doesn't. (No proof.)
           2. AdaGrad *is* sensitive to learning-rate, AdaGrad not so much. (Table 1.)
           3. AdaGrad includes 2nd-order approximation. (3.2)
    """

    def __init__(self, model, rho=0.95, eps=1e-6, nllclip=(1e-15, 1-1e-15)):
        """
        `model`: The model for wich to compile functions.

        `rho`: The "momentum decay" of AdaDelta. The paper tests three values
               on MNIST: 0.9, 0.95 and 0.99, they don't change the score much.
               The paper also uses the same values for a speech task.

        `eps`: A regularization term only used to avoid singularities. The
               paper tests four values on MNIST: 1e-2, 1e-4, 1e-6, 1e-8;
               all of them work pretty well.

        `nllclip`: This should disappear soon. It's a hack.
        """
        super(MiniAdaDelta, self).__init__(model, nllclip)

        # Similarly to Adagrad, AdaDelta accumulates the square gradient of
        # each parameter, it just exponentially decays the old value,
        # effectively only summing over a recent window.
        self.sh_g2 = [
            th.shared(np.zeros_like(p.get_value()), broadcastable=p.broadcastable, name='g2_'+p.name)
            for p in model.params
        ]

        # Similarly to momentum, AdaDelta accumulates previous update values.
        # This also happens in a decaying fashion, so as to cover a window.
        self.sh_delta2 = [
            th.shared(np.zeros_like(p.get_value()), broadcastable=p.broadcastable, name='d2_'+p.name)
            for p in model.params
        ]

        g = T.grad(cost=model.cost, wrt=model.params)

        updates = []
        for sh_p, gp, sh_g2, sh_d2 in zip(model.params, g, self.sh_g2, self.sh_delta2):
            g2 = rho*sh_g2 + (1-rho)*gp*gp
            up = T.sqrt((sh_d2+eps) / (g2+eps)) * gp
            d2 = rho*sh_d2 + (1-rho)*up*up
            updates.append((sh_g2, g2))
            updates.append((sh_p, sh_p - up))
            updates.append((sh_d2, d2))

        # Notice how we never used the learning-rate!
        # We thus need to tell Theano that we're aware of the fact
        # that we're not using it.
        self.fn_train = th.function(
            inputs=[self.sh_learningrate],
            outputs=[model.cost, model.nll],
            updates=updates,
            on_unused_input='ignore'
        )
