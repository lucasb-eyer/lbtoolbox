import torch
import torch.nn as nn
import torch.nn.init


def maybe_cuda(what, use_cuda=True):
    if torch.cuda.is_available() and use_cuda:
        what = what.cuda()
    return what


def init(mod, init, bias=0):
    init(mod.weight)
    if hasattr(mod, 'bias'):
        torch.nn.init.constant(mod.bias, bias)
    return mod


def count_parameters(net, in_bytes=False):
    """
    Only works if `net` is an instance of `nn.Module`, i.e. has a `parameters()` method.
    See also https://discuss.pytorch.org/t/finding-the-total-number-of-trainable-parameters-in-a-graph/1751/2
    """
    getsize = (lambda p: p.data.nelement()) if not in_bytes else lambda p: p.data.nelement()*p.data.element_size()
    return sum(map(getsize, net.parameters()))


class View(nn.Module):
    def __init__(self, *dims, change_batch_dim=False):
        super(View, self).__init__()
        self._dims = dims
        self._batchdim = change_batch_dim

    def forward(self, x):
        if self._batchdim:
            return x.view(*self.dims)
        else:
            return x.view(x.size(0), *self._dims)
