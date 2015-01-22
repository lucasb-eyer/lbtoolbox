#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt

# TODO: Get rid of scipy, it's a huge drain on RAM! (~500MB)
import scipy.signal as sps

from lbtoolbox import plotting as lbplt

def gauss2d(shape, mu=None, sigma=None, radians=None):
    mu = mu*np.ones_like(shape, dtype=np.double) if mu is not None else 0.5 * (np.array(shape) - 1)
    sigma = sigma*np.ones_like(shape, dtype=np.double) if sigma is not None else np.array(shape)/0.9

    x, y = np.meshgrid(*map(np.arange, shape))
    x, y = x - mu[0], y - mu[1]
    if radians:
        x, y = (np.cos(radians) * x - np.sin(radians) * y,
                np.sin(radians) * x + np.cos(radians) * y)

    return np.exp(-(x**2 + y**2)/np.sum(sigma**2)) / (2.0*np.pi*np.prod(sigma))

np.testing.assert_array_almost_equal(gauss2d((3,3)), gauss2d((3,3), radians=np.pi))

def DooG(shape, derivs, mu=None, sigma=None, radians=None):
    shape = np.array(shape)
    derivs = np.array(derivs) if derivs is not None else np.zeros_like(shape)

    # Add one pixel border per derivative so that we get the derivatives correctly everywhere.
    # (One derivative crops out one border pixel because of mode='valid')
    g = gauss2d(shape+2*derivs, np.array(mu)+derivs if mu else None, sigma, radians)

    Dx = np.array([[-0.5, 0, 0.5]])
    Dy = np.array([[-0.5], [0], [0.5]])
    # TODO: get working with cv2, so as to remove dependency on scipy.
    for _ in range(derivs[0]):
        #g = cv2.sepFilter2D(g, -1, Dx, 1)[:, 1:-1]
        g = sps.convolve2d(g, Dx, mode='valid')
    for _ in range(derivs[1]):
        #g = cv2.sepFilter2D(g, -1, 1, Dy)[1:-1, :]
        g = sps.convolve2d(g, Dy, mode='valid')

    # Normalization
    #g /= np.linalg.norm(g.flatten(), 1)
    g -= np.sum(g)/np.product(g.shape)

    # Just checking; should be 1.0
    #assert np.sum(np.abs(g)) - 1 < 1e-15
    assert np.abs(np.sum(g)) < 1e-15
    return g

def DooG_sym_bank(r):
    return (
        DooG(shape=(2*r+1, 2*r+1), derivs=(0, 2), sigma=0.5),
        DooG(shape=(2*r+1, 2*r+1), derivs=(0, 1), sigma=0.5),
        DooG(shape=(2*r+1, 2*r+1), derivs=(2, 0), sigma=0.5),
        DooG(shape=(2*r+1, 2*r+1), derivs=(1, 0), sigma=0.5),

        DooG(shape=(2*r+1, 2*r+1), derivs=(0, 2), sigma=1),
        DooG(shape=(2*r+1, 2*r+1), derivs=(0, 1), sigma=1),
        DooG(shape=(2*r+1, 2*r+1), derivs=(2, 0), sigma=1),
        DooG(shape=(2*r+1, 2*r+1), derivs=(1, 0), sigma=1),
    )

def show(bank):
    fig, axes = lbplt.subplotgrid_for(bank)

    extr = np.max(np.abs(bank))
    for filt, ax in zip(bank, axes.flat):
        im = ax.imshow(filt, interpolation='nearest', cmap=plt.cm.Spectral, vmin=-extr, vmax=extr)

    fig.subplots_adjust(right=0.76)
    fig.colorbar(im, cax=fig.add_axes([0.82, 0.1, 0.06, 0.8]))
