from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init


def maybe_cuda(what, use_cuda=0, **kw):
    """ Moves `what` to CUDA and returns it, if `use_cuda` and it's available.

    Actually, `use_cuda` is the GPU-index to be used, which means `0` uses the
    first GPU. To not use GPUs, set `use_cuda` to `False` instead.
    """
    if use_cuda is True: use_cuda = 0  # Backwards compatibility.
    if use_cuda is not False and torch.cuda.is_available():
        what = what.cuda(device=use_cuda, **kw)
    return what


def init(mod, init_=None, bias=0):
    """ Initializes `mod` with `init` and returns it.
    Also sets the bias to the given constant value, if available.

    Useful for the `Sequential` constructor and friends.
    """
    if init_ is not None and getattr(mod, 'weight', None) is not None:
        init_(mod.weight)
    if getattr(mod, 'bias', None) is not None:
        torch.nn.init.constant(mod.bias, bias)
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


def conv3x3(cin, cout, stride=1, groups=1):
    return nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)


def conv1x1(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=False)


class PreActBlock(nn.Module):
    """
    Follows the implementation of "Identity Mappings in Deep Residual Networks" here:
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
    """
    def __init__(self, cin, cout=None, cmid=None, stride=1, downsample=None, dropout=None):
        super(PreActBlock, self).__init__()
        cout = cout or cin
        cmid = cmid or cout

        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = conv3x3(cin, cmid, stride)
        self.bn2 = nn.BatchNorm2d(cmid)
        self.conv2 = conv3x3(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        self.downsample = downsample
        if (stride != 1 or cin != cout) and downsample in (True, None):
            self.downsample = conv1x1(cin, cout, stride)

        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        # Conv'ed branch
        out = x
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv1(self.relu(self.bn1(out)))
        out = self.conv2(self.relu(self.bn2(out)))

        # Residual branch
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        return out + residual

    def reset_parameters(self):
        # Following https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua#L129
        nn.init.kaiming_normal(self.conv1.weight, a=0, mode='fan_out')
        nn.init.kaiming_normal(self.conv2.weight, a=0, mode='fan_out')

        # Not the default =(
        nn.init.constant(self.bn1.weight, 1)
        nn.init.constant(self.bn2.weight, 1)
        return self


class PreActBottleneck(nn.Module):
    """
    Follows the implementation of "Identity Mappings in Deep Residual Networks" here:
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """
    def __init__(self, cin, cout=None, cmid=None, stride=1, downsample=None, dropout=None):
        super(PreActBottleneck, self).__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = conv1x1(cin, cmid)
        self.bn2 = nn.BatchNorm2d(cmid)
        self.conv2 = conv3x3(cmid, cmid, stride)  # Original code has it on conv1!!
        self.bn3 = nn.BatchNorm2d(cmid)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        self.downsample = downsample
        if (stride != 1 or cin != cout) and downsample in (True, None):
            self.downsample = conv1x1(cin, cout, stride)

        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        # Conv'ed branch
        out = x
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv1(self.relu(self.bn1(out)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))

        # Residual branch
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)

        return out + residual

    def reset_parameters(self):
        # Following https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua#L129
        nn.init.kaiming_normal(self.conv1.weight, a=0, mode='fan_out')
        nn.init.kaiming_normal(self.conv2.weight, a=0, mode='fan_out')
        nn.init.kaiming_normal(self.conv3.weight, a=0, mode='fan_out')

        # Not the default =(
        nn.init.constant(self.bn1.weight, 1)
        nn.init.constant(self.bn2.weight, 1)
        nn.init.constant(self.bn3.weight, 1)
        return self


class NeXtBlockC(nn.Module):
    """
    Follows the implementation of "Aggregated Residual Transformations for Deep Neural Networks" here:
    https://github.com/facebookresearch/ResNeXt
    """
    def __init__(self, cin, cout=None, cmid=None, stride=1, downsample=None):
        """
        Now, cmid is (C, D) which means C convolutions on D channels in the bottleneck.
        C == cardinality.
        """
        super(NeXtBlockC, self).__init__()
        cout = cout or cin
        C, D = cmid if isinstance(cmid, tuple) else ((cmid, cout//cmid//2) if cmid is not None else (4, cout//8))

        self.conv1 = conv1x1(cin, C*D)
        self.bn1 = nn.BatchNorm2d(C*D)
        self.conv2 = conv3x3(C*D, C*D, groups=C, stride=stride)
        self.bn2 = nn.BatchNorm2d(C*D)
        self.conv3 = conv1x1(C*D, cout)
        self.bn3 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        self.downsample = downsample
        if (stride != 1 or cin != cout) and downsample in (True, None):
            self.downsample = nn.Sequential(OrderedDict([
                ('conv', conv1x1(cin, cout, stride)),
                ('bn', nn.BatchNorm2d(cout))
            ]))
            # TODO: They now optionally use strided, 0-padded identity (abusing avgpool) in the code.

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
        # Following https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua#L208
        nn.init.kaiming_normal(self.conv1.weight, a=0, mode='fan_out')
        nn.init.kaiming_normal(self.conv2.weight, a=0, mode='fan_out')
        nn.init.kaiming_normal(self.conv3.weight, a=0, mode='fan_out')

        # Not the default =(
        nn.init.constant(self.bn1.weight, 1)
        nn.init.constant(self.bn2.weight, 1)
        nn.init.constant(self.bn3.weight, 1)
        return self


class PreActNeXtBlockC(nn.Module):
    """
    My own "pre-activated" version of the ResNeXt block C.
    """
    def __init__(self, cin, cout=None, cmid=None, stride=1, downsample=None):
        """
        Now, cmid is (C, D) which means C convolutions on D channels in the bottleneck.
        C == cardinality.
        """
        super(NeXtBlockC, self).__init__()
        cout = cout or cin
        C, D = cmid if isinstance(cmid, tuple) else ((cmid, cout//cmid//2) if cmid is not None else (4, cout//8))

        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = conv1x1(cin, C*D)
        self.bn2 = nn.BatchNorm2d(C*D)
        self.conv2 = conv3x3(C*D, C*D, groups=C, stride=stride)
        self.bn3 = nn.BatchNorm2d(C*D)
        self.conv3 = conv1x1(C*D, cout)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        self.downsample = downsample
        if (stride != 1 or cin != cout) and downsample in (True, None):
            self.downsample = nn.Sequential(OrderedDict([
                ('bn', nn.BatchNorm2d(cin)),
                ('relu', nn.ReLU(inplace=True)),
                ('conv', conv1x1(cin, cout, stride)),
            ]))
            # TODO: They now optionally use strided, 0-padded identity (abusing avgpool) in the code.

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
        # Following https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua#L208
        nn.init.kaiming_normal(self.conv1.weight, a=0, mode='fan_out')
        nn.init.kaiming_normal(self.conv2.weight, a=0, mode='fan_out')
        nn.init.kaiming_normal(self.conv3.weight, a=0, mode='fan_out')

        # Not the default =(
        nn.init.constant(self.bn1.weight, 1)
        nn.init.constant(self.bn2.weight, 1)
        nn.init.constant(self.bn3.weight, 1)
        return self
