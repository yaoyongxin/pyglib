from __future__ import print_function

import unittest, random
import numpy as np
from scipy.linalg import expm,sqrtm
from pyglib.math.matrix_util import yield_derivative_f_matrix, \
        get_derivative_sroot_entropyn_at_h


def get_derivative_sroot_entropyn_at_h(a, h):
    '''
    evaluate p_{\squareroot(a(1-a))} / p_h.
    not correct.
    '''
    k0 = a.dot(np.eye(a.shape[0])-a)
    k0inv = np.linalg.inv(k0)
    k0srt = slinalg.sqrtm(k0)
    k1 = h - a.dot(h) - h.dot(a)
    res = 0.5*k0srt.dot(k0inv).dot(k1)
    return res


class KnowValues(unittest.TestCase):
    def test_derivative_exp_ix(self):
        # random Hermitian matrix of dimention 5.
        n = 5
        x = np.random.rand(n, n) + np.random.rand(n, n)*1.j
        x += np.conj(x.T)
        # Numerically calculate the derivative w.r.t. h.
        h = np.array([ \
            [0,              0, 1./np.sqrt(2.), 0, 0],
            [0,              0, 0,              0, 0],
            [1./np.sqrt(2.), 0, 0,              0, 0],
            [0,              0, 0,              0, 0],
            [0,              0, 0,              0, 0]])

        expix0 = expm(-1.j*x)
        delta = 1.e-7
        expix1 = expm(-1.j*(x + delta*h))
        partial0 = (expix1 - expix0)/delta
        f = lambda x: np.exp(-1.j*x)
        fp = lambda x: -1.j*np.exp(-1.j*x)
        partials = yield_derivative_f_matrix(x, [h], f, fp)
        for partial in partials:
            err = np.abs(np.max(partial - partial0))
            self.assertAlmostEqual(err, 0.)

    def test_derivative_srootn(self):
        # random Hermitian matrix of dimention 5.
        n = 5
        x = np.random.rand(n, n) + np.random.rand(n, n)*1.j
        x += np.conj(x.T)
        _,v = np.linalg.eigh(x)
        w = np.random.rand(5)
        # get a valid density matrix
        x = v.dot(np.diag(w)).dot(v.T.conj())

        # Numerically calculate the derivative w.r.t. h.
        h = np.array([ \
            [0,              0, 1./np.sqrt(2.), 0, 0],
            [0,              0, 0,              0, 0],
            [1./np.sqrt(2.), 0, 0,              0, 0],
            [0,              0, 0,              0, 0],
            [0,              0, 0,              0, 0]])
        sroot0 = sqrtm(x.dot(np.eye(x.shape[0])-x))
        delta = 1.e-7
        xp = x + delta*h
        sroot1 = sqrtm(xp.dot(np.eye(x.shape[0])-xp))
        partial0 = (sroot1 - sroot0)/delta
        partial1 = get_derivative_sroot_entropyn_at_h(x, h)

        f = lambda x: np.sqrt(x*(1-x))
        fp = lambda x: (0.5-x)/np.sqrt(x*(1-x))
        partials = yield_derivative_f_matrix(x, [h], f, fp)

        for partial2 in partials:
            print("loener")
            print(np.abs(np.max(partial2 - partial0)))

        print("check")
        print(np.max(np.abs(partial0)))
        print(np.max(np.abs(partial1)))

        err = np.abs(np.max(partial1 - partial0))

        print("partial2")
        print(err)

        self.assertAlmostEqual(err, 0.)

    def test_derivative_tr_exp_ix_a(self):
        # random Hermitian matrix of dimention 5.
        n = 5
        a = np.random.rand(n, n)
        x = np.random.rand(n, n) + np.random.rand(n, n)*1.j
        x += np.conj(x.T)
        # Numerically calculate the derivative w.r.t. h.
        h = np.array([ \
            [0,              0, 1./np.sqrt(2.), 0, 0],
            [0,              0, 0,              0, 0],
            [1./np.sqrt(2.), 0, 0,              0, 0],
            [0,              0, 0,              0, 0],
            [0,              0, 0,              0, 0]])

        expix0 = expm(-1.j*x)
        delta = 1.e-7
        expix1 = expm(-1.j*(x + delta*h))
        partial0 = (np.trace(np.dot(expix1, a)) -
                np.trace(np.dot(expix0, a)))/delta
        f = lambda x: np.exp(-1.j*x)
        fp = lambda x: -1.j*np.exp(-1.j*x)
        partials = yield_derivative_f_matrix(x, [h], f, fp)
        for partial in partials:
            # trace
            _partial = np.sum(partial.T*a)
            err = np.abs(_partial - partial0)
            self.assertAlmostEqual(err, 0.)


if __name__=="__main__":
    print("Tests for matrix_util.")
    unittest.main()
