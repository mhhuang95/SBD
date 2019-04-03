#Code for Rapid, Robust, and Reliable Blind Deconvolution via Nonconvex Optimization
# Minhui Huang

import numpy as np
from scipy.linalg import dft
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.linalg import svds


def Wir_flow(coe):
    #initialization


    K=50
    N=50
    L = int(3 * (K + N))

    tol = 1e-12

    maxit = 10000
    lam = 0.0001
    sp = 3


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
    xt = np.conj(x.T)

    R = y - A_op(h,x)
    G = A_op_T(R)

    loss = []
    l = 0
    l_ = 1
    i = 0

    while np.linalg.norm(l - l_) > tol:
        l_ = l
        i+=1
        alpha = np.linalg.norm(xt) / np.linalg.norm(h)
        h = alpha * h
        xt = 1 / alpha * xt

        Gh = G.dot(np.conj(xt.T))
        AGhm = A_op(Gh, np.conj(xt.T))
        th = np.linalg.norm(Gh)**2/np.linalg.norm(AGhm)**2
        h += th*Gh

        R = R - th*AGhm
        G = A_op_T(R)

        Gmt = np.conj(h.T).dot(G)
        AhGmt = A_op(h, np.conj(Gmt.T))
        tmt = np.linalg.norm(Gmt) ** 2 / np.linalg.norm(AhGmt) ** 2
        xt = soft(xt + tmt * Gmt, tmt*lam)

        R = R - tmt * AhGmt
        G = A_op_T(R)

        loss.append(np.linalg.norm(R)/np.linalg.norm(y))
        l = loss[-1]
        print(i,loss[-1])



    print(np.linalg.norm(h.dot(xt) - h_0.dot(np.conj(x_0.T)))/np.linalg.norm(h_0.dot(np.conj(x_0.T))))

    print(np.hstack([x_0, xt.T]))
    plt.figure()

    plt.semilogy(loss)
    plt.xlabel('Iter')
    plt.ylabel('Error')
    plt.show()


def soft(x, lam):
        mask = (np.abs(x) > 0)
        x[mask] =np.maximum(np.abs(x[mask]) - lam, 0) * (x[mask]/np.abs(x[mask]))
        return x



if __name__ == "__main__":
    Wir_flow(3)

    '''
    coes = np.linspace(1,4,10)
    suss_rate = []
    for coe in coes:
        exp = 50
        res = 0
        for i in range(50):
            if Wir_flow(coe) < 0.01:
                res += 1
        suss_rate.append(res/exp)
    plt.figure()
    plt.plot(coes, suss_rate)
    plt.xlabel('L/(K+N)')
    plt.ylabel('Sucessful rate')
    plt.show()
    '''