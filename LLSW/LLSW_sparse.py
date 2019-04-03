#Code for Rapid, Robust, and Reliable Blind Deconvolution via Nonconvex Optimization
# Minhui Huang

import numpy as np
from scipy.linalg import dft
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.linalg import svds


def Wir_flow():
    #initialization

    K=50
    N=50
    L = int(3 *(K + N))
    sp = 3


    tol = 1e-2
    t0 = 1
    alpha = 0.3
    beta = 0.5
    maxit = 100
    lam = 1

    dft_mat = dft(L)


    B = dft_mat[:,:K]/np.sqrt(L)
    C = np.random.randn(L,N) + 1j* np.random.randn(L,N)

    #ground truth

    h_0 = np.random.randn(K,1) + 1j*np.random.randn(K,1)
    h_0 = h_0/np.linalg.norm(h_0)
    shuf = np.arange(32)
    np.random.shuffle(shuf)
    x_0 = np.zeros([N, 1], dtype=complex)
    x_0[shuf[:3]] = np.random.randn(sp, 1) + 1j * np.random.randn(sp, 1)
    x_0 = x_0 / np.linalg.norm(x_0)

    def A_op(h, x):
        return (B.dot(h))*(C.dot(np.conj(x)))

    def A_op_T(y):
        return np.conj(B.T).dot(np.diag(y.flatten()).dot(np.conj(C)))

    def f(h,x):
        return np.linalg.norm(A_op(h,x) - y)**2

    d_0 = np.linalg.norm(h_0)*np.linalg.norm(x_0)
    u_h = np.sqrt(L*np.max(np.abs(B.dot(h_0)))**2)

    y = A_op(h_0,x_0)

    A_y = A_op_T(y)

    u, s, vh = svds(A_y,1)
    d = s[0]
    print(u.shape)
    print(vh.shape)

    h = np.sqrt(d) * u
    x = np.sqrt(d) * np.conj(vh.T)

    loss = []

    #Alternating minimization

    g_h = np.ones(h.shape[0], dtype=complex)
    g_h_ = np.zeros(h.shape[0], dtype=complex)


    while np.abs(np.linalg.norm(g_h) ** 2 - np.linalg.norm(g_h_) ** 2) > tol:


        g_h_ = g_h
        it = 0
        cost = 1
        cost_ = 0

        R = A_op(h, x) - y
        G = A_op_T(R)
        g_x =np.conj(G.T).dot(h)

        '''
        while it < maxit and np.abs(cost - cost_) > tol:
            t = 1
            while f(h, x - t * g_x) > f(h, x) + np.conj(g_x.T).dot(-t * g_x) + 1 / (2 * t) * (np.linalg.norm(t * g_x) ** 2):
                t = beta * t

            x = soft(x - t * g_x, t*lam)

            cost_ = cost
            cost = f(h,x) + lam* np.sum(np.abs(x))
            print(cost)
            loss.append(cost)
        '''

        R = A_op(h, x) - y
        G = A_op_T(R)
        g_h = G.dot(x)

        s = 0.1

        while f(h - s*g_h, x) > f(h, x) + np.conj(g_h.T).dot(-s * g_h) + 1 / (2 * s) * (np.linalg.norm(s * g_h) ** 2):
            s = beta * s

        h = h - s * g_h

        alpha = np.linalg.norm(x) / np.linalg.norm(h)
        h = alpha * h
        x = 1 / alpha * x

        cost = f(h, x) + lam * np.linalg.norm(x)
        loss.append(cost)
        print(cost)

    print(np.linalg.norm(h.dot(np.conj(x.T)) - h_0.dot(np.conj(x_0.T)))/np.linalg.norm(h_0.dot(np.conj(x_0.T))))
    print(np.hstack([x_0,x]))
    #return np.linalg.norm(h.dot(xt) - h_0.dot(np.conj(x_0.T)))/np.linalg.norm(h_0.dot(np.conj(x_0.T)))

    plt.figure()

    plt.semilogy(loss)
    plt.show()


def soft(x, lam):
        mask = (np.abs(x) > 0)
        x[mask] =np.maximum(np.abs(x[mask]) - lam, 0) * (x[mask]/np.abs(x[mask]))
        return x

if __name__ == "__main__":
    Wir_flow()