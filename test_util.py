#!/usr/bin/env python3

import unittest

import lbtoolbox.util as u

import numpy as np
import numpy.testing as npt


class TestUtilFunctions(unittest.TestCase):


    def test_tuplize(self):
        self.assertEqual(u.tuplize(1), (1,))
        self.assertEqual(u.tuplize((1,)), (1,))
        self.assertEqual(u.tuplize((1,2)), (1,2))

        self.assertEqual(u.tuplize([1]), (1,))
        self.assertEqual(u.tuplize([1], lists=False), ([1],))

        self.assertEqual(u.tuplize("a"), ("a",))
        self.assertEqual(u.tuplize("ab"), ("ab",))

        self.assertEqual(u.tuplize(None), None)
        self.assertEqual(u.tuplize(None, tuplize_none=True), tuple())


    def test_maybetuple(self):
        self.assertEqual(u.maybetuple((0,1)), (0,1))
        self.assertEqual(u.maybetuple((0,)), 0)
        self.assertEqual(u.maybetuple(range(2)), (0,1))
        self.assertEqual(u.maybetuple(range(1)), 0)
        with self.assertRaises(TypeError):
            u.maybetuple(3)


    def test_collect(self):
        self.assertEqual(u.collect([0, 1, 2]), (0,1,2))
        self.assertEqual(u.collect(range(3)), (0,1,2))
        self.assertEqual(u.collect([None, (0,1), 2, None]), (0,1,2))


    def test_batched_padded(self):
        # 1D array with fitting size
        l = list(u.batched_padded(3, np.arange(9)))
        self.assertEqual(len(l), 3)
        batchsizes, batches = zip(*l)
        self.assertEqual(batchsizes[0], 3)
        self.assertEqual(batchsizes[1], 3)
        self.assertEqual(batchsizes[2], 3)
        npt.assert_array_equal(batches[0], [0,1,2])
        npt.assert_array_equal(batches[1], [3,4,5])
        npt.assert_array_equal(batches[2], [6,7,8])

        # 1D array with need for padding
        l = list(u.batched_padded(3, np.arange(10)))
        self.assertEqual(len(l), 4)
        batchsizes, batches = zip(*l)
        self.assertEqual(batchsizes[0], 3)
        self.assertEqual(batchsizes[1], 3)
        self.assertEqual(batchsizes[2], 3)
        self.assertEqual(batchsizes[3], 1)
        npt.assert_array_equal(batches[0], [0,1,2])
        npt.assert_array_equal(batches[1], [3,4,5])
        npt.assert_array_equal(batches[2], [6,7,8])
        npt.assert_array_equal(batches[3], [9,0,0])

        # 2D array with fitting size
        l = list(u.batched_padded(3, np.array([[2*i, 2*i+1] for i in range(9)])))
        self.assertEqual(len(l), 3)
        batchsizes, batches = zip(*l)
        self.assertEqual(batchsizes[0], 3)
        self.assertEqual(batchsizes[1], 3)
        self.assertEqual(batchsizes[2], 3)
        npt.assert_array_equal(batches[0], [[0,1],[2,3],[4,5]])
        npt.assert_array_equal(batches[1], [[6,7],[8,9],[10,11]])
        npt.assert_array_equal(batches[2], [[12,13],[14,15],[16,17]])

        # 2D array with need for padding
        l = list(u.batched_padded(3, np.array([[2*i, 2*i+1] for i in range(10)])))
        self.assertEqual(len(l), 4)
        batchsizes, batches = zip(*l)
        self.assertEqual(batchsizes[0], 3)
        self.assertEqual(batchsizes[1], 3)
        self.assertEqual(batchsizes[2], 3)
        self.assertEqual(batchsizes[3], 1)
        npt.assert_array_equal(batches[0], [[0,1],[2,3],[4,5]])
        npt.assert_array_equal(batches[1], [[6,7],[8,9],[10,11]])
        npt.assert_array_equal(batches[2], [[12,13],[14,15],[16,17]])
        npt.assert_array_equal(batches[3], [[18,19],[0,0],[0,0]])


    def test_batched_padded_x(self):
        # 1D arrays with fitting size
        l = list(u.batched_padded_x(3, np.arange(9), np.arange(9)))
        self.assertEqual(len(l), 3)
        npt.assert_array_equal(l[0][0], [0,1,2])
        npt.assert_array_equal(l[0][1], [0,1,2])
        npt.assert_array_equal(l[1][0], [3,4,5])
        npt.assert_array_equal(l[1][1], [3,4,5])
        npt.assert_array_equal(l[2][0], [6,7,8])
        npt.assert_array_equal(l[2][1], [6,7,8])

        # 1D array with need for padding
        l = list(u.batched_padded_x(3, np.arange(10), np.arange(10)))
        self.assertEqual(len(l), 4)
        npt.assert_array_equal(l[0][0], [0,1,2])
        npt.assert_array_equal(l[0][1], [0,1,2])
        npt.assert_array_equal(l[1][0], [3,4,5])
        npt.assert_array_equal(l[1][1], [3,4,5])
        npt.assert_array_equal(l[2][0], [6,7,8])
        npt.assert_array_equal(l[2][1], [6,7,8])
        npt.assert_array_equal(l[3][0], [9,0,0])
        npt.assert_array_equal(l[3][1], [9])

        # 2D array with fitting size
        l = list(u.batched_padded_x(3, np.array([[2*i, 2*i+1] for i in range(9)]), np.arange(9)))
        self.assertEqual(len(l), 3)
        npt.assert_array_equal(l[0][0], [[0,1],[2,3],[4,5]])
        npt.assert_array_equal(l[0][1], [0, 1, 2])
        npt.assert_array_equal(l[1][0], [[6,7],[8,9],[10,11]])
        npt.assert_array_equal(l[1][1], [3, 4, 5])
        npt.assert_array_equal(l[2][0], [[12,13],[14,15],[16,17]])
        npt.assert_array_equal(l[2][1], [6, 7, 8])

        # 2D array with need for padding
        l = list(u.batched_padded_x(3, np.array([[2*i, 2*i+1] for i in range(10)]), np.arange(10)))
        self.assertEqual(len(l), 4)
        npt.assert_array_equal(l[0][0], [[0,1],[2,3],[4,5]])
        npt.assert_array_equal(l[0][1], [0, 1, 2])
        npt.assert_array_equal(l[1][0], [[6,7],[8,9],[10,11]])
        npt.assert_array_equal(l[1][1], [3, 4, 5])
        npt.assert_array_equal(l[2][0], [[12,13],[14,15],[16,17]])
        npt.assert_array_equal(l[2][1], [6, 7, 8])
        npt.assert_array_equal(l[3][0], [[18,19],[0,0],[0,0]])
        npt.assert_array_equal(l[3][1], [9])
