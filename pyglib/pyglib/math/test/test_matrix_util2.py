from __future__ import print_function

import unittest, random
import numpy as np
from scipy.linalg import expm,sqrtm
from pyglib.math.matrix_util import yield_derivative_f_matrix, \
        get_derivative_sroot_entropyn_at_h


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
        n = 2
        x = np.random.rand(n, n) + np.random.rand(n, n)*1.j
        x += np.conj(x.T)
        _,v = np.linalg.eigh(x)
        w = np.random.rand(2)
        # get a valid density matrix
        x = v.dot(np.diag(w)).dot(v.T.conj())

        # Numerically calculate the derivative w.r.t. h.
        h = np.array([ \
            [0,              1./np.sqrt(2.)],
            [1./np.sqrt(2.), 0,            ]])
        delta = 1.e-7
        xp = x + delta*h
        sroot0 = sqrtm(np.eye(x.shape[0])-x)
        sroot1 = sqrtm(np.eye(x.shape[0])-xp)


        k0 = np.eye(x.shape[0])-x
        k0inv = np.linalg.inv(k0)

        sroot2 = sqrtm(np.eye(x.shape[0])-k0inv.dot(h)*delta).dot(sroot0)
        sroot3 = (np.eye(x.shape[0])-np.linalg.inv(np.eye(x.shape[0])-x) \
                .dot(h)*delta/2).dot(sroot0)
        sroot4 = sqrtm(k0.dot(k0inv).dot(np.eye(x.shape[0])-xp))

        diff = sroot1 - sroot2
        print(np.abs(np.max(sroot2 - sroot1)))
        print(np.abs(np.max(sroot3 - sroot1)))
        print(np.abs(np.max(sroot4 - sroot1)))
        quit()

        partial4 = (sroot3-sroot0)/delta


        partial0 = (sroot1 - sroot0)/delta


        print(np.abs(np.max(partial4 - partial0)))

        quit()

        partial1 = get_derivative_sroot_entropyn_at_h(x, h)

        f = lambda x: np.sqrt(1-x)
        fp = lambda x: -0.5/np.sqrt(1-x)
        partials = yield_derivative_f_matrix(x, [h], f, fp)

        for partial2 in partials:
            print("loener")

            print(partial2)
            print(partial0)

            print(np.abs(np.max(partial2 - partial0)))
            print(np.abs(np.max(partial2 - partial1)))

        print("check")
        print(np.max(np.abs(partial0)))
        print(np.max(np.abs(partial1)))

        err = np.abs(np.max(partial1 - partial0))

        print("partial2")
        print(err)

        self.assertAlmostEqual(err, 0., places=6)

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
