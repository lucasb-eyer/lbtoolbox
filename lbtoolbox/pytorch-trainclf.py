#!/usr/bin/env python3
# coding: utf-8
from argparse import ArgumentParser
from importlib import import_module
from itertools import count
from os.path import join as pjoin, isfile
from os import makedirs
from signal import SIGINT, SIGTERM
from time import time, sleep
import logging
import logging.config

import numpy as np
import lbtoolbox.io as lbio
from lbtoolbox.util import Uninterrupt, ramp, expdec, stairs
from lbtoolbox.chrono import Chrono

import torch
from torch.autograd import Variable
import torch.nn as nn
import lbtoolbox.pytorch as lbt

import torchvision as tv


try:
    profile
except NameError:
    def profile(f):
        return f


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Up to 3 trials because NFS
def save_retry(to_save, path, max_attempts=5):
    for _ in range(max_attempts):
        try:
            torch.save(to_save, path)
            return
        except OSError:
            sleep(1)
        except:
            raise


class Lighting:
    """  Alex Krizhevsky's "fancy PCA", literal translation from
    https://github.com/facebookresearch/ResNeXt/blob/master/datasets/transforms.lua#L184
    """

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


def mktrainval(dataset, datadir):
    if dataset == 'ImageNet':
        normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        pca_eigval = torch.FloatTensor([0.2175, 0.0188, 0.0045])
        pca_eigvec = torch.FloatTensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])

        # See https://github.com/facebookresearch/ResNeXt/blob/master/datasets/imagenet.lua#L80
        train_set = tv.datasets.ImageFolder(pjoin(datadir, 'train'), tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            Lighting(0.1, pca_eigval, pca_eigvec),
            normalize,
        ]))
        valid_set = tv.datasets.ImageFolder(pjoin(datadir, 'val'), tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            normalize,
        ]))
    elif dataset == 'CIFAR10':
        normalize = tv.transforms.Normalize(mean=tv.datasets.CIFAR10.mean, std=tv.datasets.CIFAR10.std)
        train_set = tv.datasets.CIFAR10(datadir, transform=tv.transforms.Compose([
            tv.transforms.Pad(4),
            tv.transforms.RandomCrop(32),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            normalize,
        ]), train=True, download=True)
        valid_set = tv.datasets.CIFAR10(datadir, transform=tv.transforms.Compose([
            tv.transforms.ToTensor(),
            normalize,
        ]), train=False, download=True)
    elif dataset == 'MNIST':
        normalize = tv.transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        train_set = tv.datasets.MNIST(datadir, train=True, download=True, transform=tv.transforms.Compose([
            tv.transforms.ToTensor(),
            normalize,
        ]))
        valid_set = tv.datasets.MNIST(datadir, train=False, download=True, transform=tv.transforms.Compose([
            tv.transforms.ToTensor(),
            normalize,
        ]))
    return train_set, valid_set


@profile
def main():
    # Lets cuDNN benchmark conv implementations and choose the fastest.
    # Only good if sizes stay the same within the main loop!
    torch.backends.cudnn.benchmark = True

    parser = ArgumentParser(description="Train an ImageNet network")
    parser.add_argument('--name',
                        help="Name of this run. Used for monitoring and checkpointing.")
    parser.add_argument('--model', required=True,
                        help="Name of file in models/ to load.")
    parser.add_argument('--modelargs', default="",
                        help="Arguments to pass to model's constructor.")
    parser.add_argument('--gpu', default=False, action='store', const=True, nargs='?', type=int,
                        help="Train on the GPU. Can pass a number to specify which one.")
    parser.add_argument('--data_parallel', action='store_true',
                        help="Train on all GPUs through data-parallelism.")
    parser.add_argument('--batch', type=int, default=256,
                        help="Batch size.")
    parser.add_argument('--batch-per-iter', type=int, default=1,
                        help="Number of batches to compute gradient on before updating weights.")
    parser.add_argument('--optim', default='SGD(p, lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)',
                        help="Expression for the optimizer, use `p` as placeholder for parameters.")
    parser.add_argument('--schedule', default='stairs(t, 0.1, 30, 0.01, 60, 0.001, 90, 0.0001, 120, None)',
                        help="Learning-rate schedule to use. See lbtoolbox.")
    parser.add_argument('--resume',
                        help="Checkpoint file to load for resuming.")
    parser.add_argument('--logdir', default='/fastwork/beyer/dumps/',
                        help="Where to log training info (small).")
    parser.add_argument('--logdir-extra', default=None,
                        help="Where to log full logits + targets (big).")
    parser.add_argument('--dumpdir', default='/fastwork/beyer/dumps/',
                        help="Where to dump model/optim parameters every epoch (small).")
    parser.add_argument('--dataset', default='ImageNet', choices=['ImageNet', 'CIFAR10', 'MNIST'],
                        help="Choose the dataset. Don't forget to set --datadir")
    parser.add_argument('--datadir', default='/home/beyer/.local/share/torchvision',
                        help="Path to the ImageNet data folder, preprocessed for torchvision.")
    parser.add_argument('--workers', type=int, default=8,
                        help="Number of background threads used to load data.")
    parser.add_argument('--eval_every', type=int, default=1,
                        help="Run prediction on validation set every so many epochs. Set 0 to disable.")
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help="Set verbosity level. Repeat to increase.")
    args = parser.parse_args()

    makedirs(pjoin(args.logdir, args.name), exist_ok=True)
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'stderr': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stderr',
            },
            'logfile': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': pjoin(args.logdir, args.name, 'train.log'),
                'mode': 'a',
            }
        },
        'loggers': {
            '': {
                'handlers': ['stderr', 'logfile'],
                'level': 'DEBUG',
                'propagate': True
            },
        }
    })
    # logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    logger = logging.getLogger(__name__)
    logger.flush = lambda: [h.flush() for h in logger.handlers]
    logger.info(args)

    lbt.maybe_set_cuda(args.gpu)

    modelnew = import_module('models.' + args.model).new
    model = eval("modelnew(" + args.modelargs + ")")

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    # Optionally resume from a checkpoint.
    # Load it to CPU first as we'll move the model to GPU later.
    # This way, we save a little bit of GPU memory when loading.
    if args.resume:
        logger.info("loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        e0 = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        logger.info("loaded checkpoint at epoch {}".format(e0))
    else:
        e0 = 0

    model = lbt.maybe_cuda(model, use_cuda=args.gpu)
    p = model.parameters()
    optim = eval('torch.optim.' + args.optim)

    if args.resume:
        optim.load_state_dict(checkpoint['optim'])

    train_set, valid_set = mktrainval(args.dataset, args.datadir)
    train_loader = torch.utils.data.DataLoader(train_set,
        batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_set,
        batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)


    makedirs(pjoin(args.dumpdir, args.name), exist_ok=True)
    if args.logdir:
        makedirs(pjoin(args.logdir, args.name, 'trainlog', 'costs'), exist_ok=True)
        if args.eval_every:
            makedirs(pjoin(args.logdir, args.name, 'vallog', 'costs'), exist_ok=True)
            makedirs(pjoin(args.logdir, args.name, 'vallog', 'accus'), exist_ok=True)
    if args.logdir_extra:
        makedirs(pjoin(args.logdir_extra, args.name, 'trainlog', 'logits'), exist_ok=True)
        makedirs(pjoin(args.logdir_extra, args.name, 'trainlog', 'targets'), exist_ok=True)
        if args.eval_every:
            makedirs(pjoin(args.logdir_extra, args.name, 'vallog', 'logits'), exist_ok=True)
            makedirs(pjoin(args.logdir_extra, args.name, 'vallog', 'targets'), exist_ok=True)

    # Store the training configuration in the logfile's attributes.
    logkw = vars(args)

    model.train()
    cri = lbt.maybe_cuda(nn.CrossEntropyLoss(), args.gpu)

    # Let's actually also save the initialization for better reproducibility!
    if e0 == 0:
        save_retry({
            'epoch': 0,
            'model': model.state_dict(),
            'optim': optim.state_dict(),
        }, pjoin(args.dumpdir, args.name, 'state-0.pth.tar'))

    chrono = Chrono()

    with Uninterrupt([SIGINT, SIGTERM]) as u:
        for e in count():
            e = e0 + e
            optim.zero_grad()

            log_logits = log_targets = log_costs = None

            end = time()
            for b, (X, y) in enumerate(train_loader):
                if u.interrupted:
                    break

                # Update learning-rate
                t = e + b/len(train_loader)
                lr = eval(args.schedule)

                if lr is None:
                    u.interrupted = True
                    break

                for param_group in optim.param_groups:
                    param_group['lr'] = lr

                X = Variable(lbt.maybe_cuda(X, args.gpu, async=True))
                y = Variable(lbt.maybe_cuda(y, args.gpu, async=True), requires_grad=False)

                # measure data loading time
                start = time()
                chrono._done('load', start - end)

                # compute output
                with chrono.measure('fprop'):
                    logits = model(X)
                    c = cri(logits, y)
                    c_num = float(c.data.cpu().numpy())  # Also ensures a sync point.

                # Log
                with chrono.measure('log'):
                    if args.logdir_extra:
                        logits_num = logits.data.cpu().numpy()
                        if log_logits is None:
                            log_logits = lbio.create_dat(
                                pjoin(args.logdir_extra, args.name, 'trainlog/logits/e{}'.format(e)),
                                dtype=np.float32, shape=(len(train_loader),) + logits_num.shape, **logkw)
                        log_logits[b,:len(logits_num),:] = logits_num

                        targets_num = y.data.cpu().numpy()
                        if log_targets is None:
                            log_targets = lbio.create_dat(
                                pjoin(args.logdir_extra, args.name, 'trainlog/targets/e{}'.format(e)),
                                dtype=np.int16, shape=(len(train_loader),) + targets_num.shape, **logkw)
                        log_targets[b,:len(targets_num)] = targets_num

                    if args.logdir:
                        if log_costs is None:
                            log_costs = lbio.create_dat(
                                pjoin(args.logdir, args.name, 'trainlog/costs/e{}'.format(e)),
                                dtype=np.float32, shape=len(train_loader), fillvalue=np.nan, **logkw)
                        log_costs[b] = c_num

                # Accumulate grads
                with chrono.measure('grads'):
                    (c / args.batch_per_iter).backward()

                # Update params
                if (b % args.batch_per_iter == 0) and b > 0:
                    with chrono.measure('update'):
                        optim.step()
                        optim.zero_grad()

                logger.info("[e {:.2f} | {}/{}]: {:.5f} (lr={})".format(e+b/len(train_loader), b, len(train_loader), c_num, lr))
                logger.flush()

                end = time()

            # Break before saving as the epoch is not done!
            if u.interrupted or isfile('/tmp/LUCAS_STOP_NOW_PLS'):
                break

            # Save the current state, try hard because NFS
            save_retry({
                'epoch': e+1,
                'model': model.state_dict(),
                'optim' : optim.state_dict(),
            }, pjoin(args.dumpdir, args.name, 'state-{}.pth.tar'.format(e+1)))


            if args.verbose:
                logger.info("Checkpoint saved")
                logger.flush()

            # TODO: Validate, but maybe in another script using this checkpoint!
            if args.eval_every and e % args.eval_every == 0:
                # switch to evaluate mode
                model.eval()

                vallog_logits = vallog_targets = vallog_costs = vallog_accus = None

                end = time()
                for b, (X, y) in enumerate(valid_loader):
                  with torch.no_grad():
                    X = Variable(lbt.maybe_cuda(X, args.gpu, async=True))
                    y = Variable(lbt.maybe_cuda(y, args.gpu, async=True))

                    # measure data loading time
                    start = time()
                    chrono._done('eval load', start - end)

                    # compute output
                    with chrono.measure('eval fprop'):
                        logits = model(X)
                        c = cri(logits, y)
                        c_num = float(c.data.cpu().numpy())  # Also ensures a sync point.

                    # measure accuracy and record loss
                    logits_num = logits.data.cpu()
                    targets_num = y.data.cpu()
                    accus_num = accuracy(logits_num, targets_num, topk=[1, 2, 5])
                    logits_num = logits_num.numpy()
                    targets_num = targets_num.numpy()
                    accus_num = np.array([a.numpy() for a in accus_num])

                    # Log
                    with chrono.measure('log'):
                        if args.logdir_extra:
                            if vallog_logits is None:
                                vallog_logits = lbio.create_dat(
                                    pjoin(args.logdir_extra, args.name, 'vallog/logits/e{}'.format(e)),
                                    dtype=np.float32, shape=(len(valid_loader),) + logits_num.shape, **logkw)
                            vallog_logits[b,:len(logits_num),:] = logits_num

                            if vallog_targets is None:
                                vallog_targets = lbio.create_dat(
                                    pjoin(args.logdir_extra, args.name, 'vallog/targets/e{}'.format(e)),
                                    dtype=np.int16, shape=(len(valid_loader),) + targets_num.shape, **logkw)
                            vallog_targets[b,:len(targets_num)] = targets_num

                        if args.logdir:
                            if vallog_costs is None:
                                vallog_costs = lbio.create_dat(
                                    pjoin(args.logdir, args.name, 'vallog/costs/e{}'.format(e)),
                                    dtype=np.float32, shape=len(valid_loader), fillvalue=np.nan, **logkw)
                            vallog_costs[b] = c_num

                            if vallog_accus is None:
                                vallog_accus = lbio.create_dat(
                                    pjoin(args.logdir, args.name, 'vallog/accus/e{}'.format(e)),
                                    dtype=np.float32, shape=(len(valid_loader),) + accus_num.shape, **logkw)
                            vallog_accus[b] = accus_num

                    # measure elapsed time
                    end = time()

                model.train()

                logger.info("Validation accus:")
                logger.info(str(np.mean(vallog_accus, axis=0)))
                logger.flush()

    logger.info(chrono)


if __name__ == "__main__":
    main()


# ImageNet Needs about 45min per epoch incl. validation on 2 GPUs. Reached 38.0% prec@1 and 64.7% prec@5 after 3 epochs.
