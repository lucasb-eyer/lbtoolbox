from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.init


def maybe_cuda(what=None, use_cuda=0, **kw):
    """ Moves `what` to CUDA and returns it, if `use_cuda` and it's available.

    Actually, `use_cuda` is the GPU-index to be used, which means `0` uses the
    first GPU. To not use GPUs, set `use_cuda` to `False` instead.
    """
    if use_cuda is True: use_cuda = 0  # Backwards compatibility.
    if use_cuda is not False and torch.cuda.is_available():
        what = what.cuda(device=use_cuda, **kw)
    return what


def maybe_set_cuda(use_cuda=0):
    """ Similar to `maybe_cuda`, but sets the default GPU instead.

    This is useful because PyTorch *loves* to allocate stuff on GPU-0
    behind your back in some function calls that don't look like it.
    """
    if use_cuda is True: use_cuda = 0
    if use_cuda is not False and torch.cuda.is_available():
        torch.cuda.set_device(use_cuda)


def init(mod, weight=None, bias=0):
    """ Initializes `mod`'s weight with `weight` and returns it.
    Also sets the bias to the given constant value, if available.
    Finally, `weight` can be a number for convenience.

    Useful for the `Sequential` constructor and friends.
    """
    if weight is not None and getattr(mod, 'weight', None) is not None:
        if callable(weight):
            weight(mod.weight)
        else:
            torch.nn.init.constant_(mod.weight, weight)
    if getattr(mod, 'bias', None) is not None:
        torch.nn.init.constant_(mod.bias, bias)
    return mod


def count_parameters(net, in_bytes=False):
    """
    Only works if `net` is an instance of `nn.Module`, i.e. has a `parameters()` method.
    See also https://discuss.pytorch.org/t/finding-the-total-number-of-trainable-parameters-in-a-graph/1751/2
    """
    if in_bytes:
        getsize = lambda p: p.data.nelement()*p.data.element_size()
    else:
        getsize = lambda p: p.data.nelement()
    return sum(map(getsize, net.parameters()))


class View(nn.Module):
    """ Module that reshapes the incoming tensor. """

    def __init__(self, *dims, change_batch_dim=False):
        """ Resize the incoming tensor to `dims`.
        If `change_batch_dim` is True, `dims` include the first batch
        dimension, otherwise `dims` are taken to be everything else and the
        batch dimension is kept constant.
        """
        super(View, self).__init__()
        self._dims = dims
        self._batchdim = change_batch_dim

    def forward(self, x):
        if self._batchdim:
            return x.view(*self.dims)
        else:
            return x.view(x.size(0), *self._dims)


def flatten(x):
    """ Flattens all but the first dimension of `x`. """
    return x.view(x.size(0), -1)


###############################################################################
# Kaiming blocks


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return nn.Conv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


# Following https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua#L129
# Following https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua#L208
kaiming_init = partial(nn.init.kaiming_normal, a=0, mode='fan_out')
groupnorm32 = partial(nn.GroupNorm, 32)


# TODO: In recent works, the last BN gamma of a blockk is init'ed to 0 so as to
#       de-activate it at init, see for example the group-norm paper.


class PreActBlock(nn.Module):
    """
    Follows the implementation of "Identity Mappings in Deep Residual Networks" here:
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
    """
    def __init__(self, cin, cout=None, cmid=None, stride=1, downsample=None, norm=nn.BatchNorm2d, dropout=None, first=False):
        super(PreActBlock, self).__init__()
        cout = cout or cin
        cmid = cmid or cout

        self.bn1 = norm(cin)
        self.conv1 = conv3x3(cin, cmid, stride)
        self.bn2 = norm(cmid)
        self.conv2 = conv3x3(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        self.first = first
        self.stride = stride
        self.downsample = downsample
        if (stride != 1 or cin != cout) and downsample in (True, None):
            # Projection also with pre-activation according to paper.
            self.downsample = nn.Sequential(OrderedDict([
                ('bn', norm(cin)),
                ('relu', nn.ReLU(inplace=True)),
                ('conv', conv1x1(cin, cout, stride)),
            ]))
            def reset_downsample():
                init(self.downsample.conv, kaiming_init)
                init(self.downsample.bn, 1)
            self.downsample.reset_parameters = reset_downsample

        self.dropout = nn.Dropout(dropout) if dropout else None

        self.reset_parameters()

    def forward(self, x):
        # Conv'ed branch
        out = x
        if self.dropout is not None:
            out = self.dropout(out)

        # The first block has already applied pre-act before splitting, see Appendix.
        out = self.conv1(self.relu(self.bn1(out)) if not self.first else out)
        out = self.conv2(self.relu(self.bn2(out)))

        # Residual branch
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        return out + residual

    def reset_parameters(self):
        init(self.conv1, kaiming_init)
        init(self.conv2, kaiming_init)

        # Not the default =(
        init(self.bn1, 1)
        init(self.bn2, 1)

        if hasattr(self.downsample, 'reset_parameters') and callable(self.downsample.reset_parameters):
            self.downsample.reset_parameters()

        return self


PreActBlockGN32 = partial(PreActBlock, norm=groupnorm32)


class PreActBottleneck(nn.Module):
    """
    Follows the implementation of "Identity Mappings in Deep Residual Networks" here:
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """
    def __init__(self, cin, cout=None, cmid=None, stride=1, downsample=None, norm=nn.BatchNorm2d, dropout=None, first=False):
        super(PreActBottleneck, self).__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.bn1 = norm(cin)
        self.conv1 = conv1x1(cin, cmid)
        self.bn2 = norm(cmid)
        self.conv2 = conv3x3(cmid, cmid, stride)  # Original code has it on conv1!!
        self.bn3 = norm(cmid)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        self.first = first
        self.stride = stride
        self.downsample = downsample
        if (stride != 1 or cin != cout) and downsample in (True, None):
            # Projection also with pre-activation according to paper.
            self.downsample = nn.Sequential(OrderedDict([
                ('bn', norm(cin)),
                ('relu', nn.ReLU(inplace=True)),
                ('conv', conv1x1(cin, cout, stride)),
            ]))
            def reset_downsample():
                init(self.downsample.conv, kaiming_init)
                init(self.downsample.bn, 1)
            self.downsample.reset_parameters = reset_downsample

        self.dropout = nn.Dropout(dropout) if dropout else None

        self.reset_parameters()

    def forward(self, x):
        # Conv'ed branch
        out = x
        if self.dropout is not None:
            out = self.dropout(out)

        # The first block has already applied pre-act before splitting, see Appendix.
        out = self.conv1(self.relu(self.bn1(out)) if not self.first else out)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))

        # Residual branch
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)

        return out + residual

    def reset_parameters(self):
        init(self.conv1, kaiming_init)
        init(self.conv2, kaiming_init)
        init(self.conv3, kaiming_init)

        # Not the default =(
        init(self.bn1, 1)
        init(self.bn2, 1)
        init(self.bn3, 1)

        if hasattr(self.downsample, 'reset_parameters') and callable(self.downsample.reset_parameters):
            self.downsample.reset_parameters()

        return self


PreActBottleneckGN32 = partial(PreActBottleneck, norm=groupnorm32)


class NeXtBlockC(nn.Module):
    """
    Follows the implementation of "Aggregated Residual Transformations for Deep Neural Networks" here:
    https://github.com/facebookresearch/ResNeXt
    """
    def __init__(self, cin, cout=None, cmid=None, stride=1, downsample=None, norm=nn.BatchNorm2d):
        """
        Now, cmid is (C, D) which means C convolutions on D channels in the bottleneck.
        C == cardinality.
        """
        super(NeXtBlockC, self).__init__()
        cout = cout or cin
        C, D = cmid if isinstance(cmid, tuple) else ((cmid, cout//cmid//2) if cmid is not None else (4, cout//8))

        self.conv1 = conv1x1(cin, C*D)
        self.bn1 = norm(C*D)
        self.conv2 = conv3x3(C*D, C*D, groups=C, stride=stride)
        self.bn2 = norm(C*D)
        self.conv3 = conv1x1(C*D, cout)
        self.bn3 = norm(cout)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        self.downsample = downsample
        if (stride != 1 or cin != cout) and downsample in (True, None):
            self.downsample = nn.Sequential(OrderedDict([
                ('conv', conv1x1(cin, cout, stride)),
                ('bn', norm(cout))
            ]))
            def reset_downsample():
                init(self.downsample.conv, kaiming_init)
                init(self.downsample.bn, 1)
            self.downsample.reset_parameters = reset_downsample
            # TODO: They now optionally use strided, 0-padded identity (abusing avgpool) in the code.

        self.reset_parameters()

    def forward(self, x):
        # Conv'ed branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        # Residual branch
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)

    def reset_parameters(self):
        init(self.conv1, kaiming_init)
        init(self.conv2, kaiming_init)
        init(self.conv3, kaiming_init)

        # Not the default =(
        init(self.bn1, 1)
        init(self.bn2, 1)
        init(self.bn3, 1)

        if hasattr(self.downsample, 'reset_parameters') and callable(self.downsample.reset_parameters):
            self.downsample.reset_parameters()

        return self


NeXtBlockCGN32 = partial(NeXtBlockC, norm=groupnorm32)


class PreActNeXtBlockC(nn.Module):
    """
    My own "pre-activated" version of the ResNeXt block C.
    """
    def __init__(self, cin, cout=None, cmid=None, stride=1, downsample=None, norm=nn.BatchNorm2d):
        """
        Now, cmid is (C, D) which means C convolutions on D channels in the bottleneck.
        C == cardinality.
        """
        super(NeXtBlockC, self).__init__()
        cout = cout or cin
        C, D = cmid if isinstance(cmid, tuple) else ((cmid, cout//cmid//2) if cmid is not None else (4, cout//8))

        self.bn1 = norm(cin)
        self.conv1 = conv1x1(cin, C*D)
        self.bn2 = norm(C*D)
        self.conv2 = conv3x3(C*D, C*D, groups=C, stride=stride)
        self.bn3 = norm(C*D)
        self.conv3 = conv1x1(C*D, cout)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        self.downsample = downsample
        if (stride != 1 or cin != cout) and downsample in (True, None):
            self.downsample = nn.Sequential(OrderedDict([
                ('bn', norm(cin)),
                ('relu', nn.ReLU(inplace=True)),
                ('conv', conv1x1(cin, cout, stride)),
            ]))
            def reset_downsample():
                init(self.downsample.conv, kaiming_init)
                init(self.downsample.bn, 1)
            self.downsample.reset_parameters = reset_downsample
            # TODO: They now optionally use strided, 0-padded identity (abusing avgpool) in the code.

        self.reset_parameters()

    def forward(self, x):
        # Conv'ed branch
        out = x
        out = self.conv1(self.relu(self.bn1(out)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))

        # Residual branch
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        return out + residual

    def reset_parameters(self):
        init(self.conv1, kaiming_init)
        init(self.conv2, kaiming_init)
        init(self.conv3, kaiming_init)

        # Not the default =(
        init(self.bn1, 1)
        init(self.bn2, 1)
        init(self.bn3, 1)

        if hasattr(self.downsample, 'reset_parameters') and callable(self.downsample.reset_parameters):
            self.downsample.reset_parameters()

        return self


PreActNeXtBlockCGN32 = partial(PreActNeXtBlockC, norm=groupnorm32)
