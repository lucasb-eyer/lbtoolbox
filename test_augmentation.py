#!/usr/bin/env python3

import unittest

import lbtoolbox.augmentation as lbaug

import numpy as np


class TestCropper(unittest.TestCase):

    def test_cropper(self):
        # Many random tests
        for _ in range(100):
            scrop = tuple(np.random.randint(1,100, size=2))
            c = lbaug.Cropper(scrop)

            # Computation of outshape.
            for i in range(100):
                s0 = np.random.randint(scrop[0]+1, scrop[0]+100)
                s1 = np.random.randint(scrop[1]+1, scrop[1]+100)
                self.assertEqual(c.outshape((s0, s1)), scrop)

            # Training crops
            for i in range(100):
                s0 = np.random.randint(scrop[0]+1, scrop[0]+100)
                s1 = np.random.randint(scrop[1]+1, scrop[1]+100)
                self.assertEqual(c.transform_train(np.random.rand(s0, s1)).shape, scrop)

            # Testing crops
            s0 = np.random.randint(scrop[0]+1, scrop[0]+100)
            s1 = np.random.randint(scrop[1]+1, scrop[1]+100)
            self.assertEqual(c.npreds(fast=False), 5)
            self.assertEqual(c.transform_pred(np.random.rand(s0, s1), 0, fast=False).shape, scrop)
            self.assertEqual(c.transform_pred(np.random.rand(s0, s1), 1, fast=False).shape, scrop)
            self.assertEqual(c.transform_pred(np.random.rand(s0, s1), 2, fast=False).shape, scrop)
            self.assertEqual(c.transform_pred(np.random.rand(s0, s1), 3, fast=False).shape, scrop)
            self.assertEqual(c.transform_pred(np.random.rand(s0, s1), 4, fast=False).shape, scrop)

            self.assertEqual(c.npreds(fast=True), 1)
            self.assertEqual(c.transform_pred(np.random.rand(s0, s1), 0, fast=True).shape, scrop)
