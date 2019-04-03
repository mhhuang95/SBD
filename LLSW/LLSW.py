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
    L = int(coe* (K + N))


    tol = 1e-10
    t0 = 1
    alpha = 0.3
    beta = 0.5
    maxit = 1000

    dft_mat = dft(L)


    B = dft_mat[:,:K]/np.sqrt(L)
    C = np.random.randn(L,N) + 1j* np.random.randn(L,N)

    #ground truth

    h_0 = np.random.randn(K,1) + 1j*np.random.randn(K,1)
    h_0 = h_0/np.linalg.norm(h_0)
    x_0 = np.random.randn(N,1)+ 1j*np.random.randn(N,1)
    x_0 = x_0/np.linalg.norm(x_0)

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

    reltol = tol * np.linalg.norm(y)
    R = y - A_op(h,x)
    G = A_op_T(R)

    loss = []

    for iter in range(maxit):
        Gh = G.dot(np.conj(xt.T))
        Gxt = np.conj(h.T).dot(G)

        t = t0
        h0 = h
        xt0 = xt
        R0 = R

        while np.sum(R.real**2 + R.imag**2)>np.sum(R0.real**2 + R0.imag**2) - alpha*t*(np.sum(Gh.real**2 + Gh.imag**2) + np.sum(Gxt.real**2 + Gxt.imag**2)):

            h = h0 + t * Gh
            xt = xt0 + t * Gxt
            R = y - A_op(h, np.conj(xt.T))

            t = beta * t
            #print('Grad, iter:', iter, t)

        alpha = np.linalg.norm(xt) / np.linalg.norm(h)
        h = alpha * h
        xt = 1 / alpha * xt

        G = A_op_T(R)

        loss.append(np.linalg.norm(R)/np.linalg.norm(y))
        print(loss[-1])

        if np.linalg.norm(R) < reltol:
            break

    print(np.linalg.norm(h.dot(xt) - h_0.dot(np.conj(x_0.T)))/np.linalg.norm(h_0.dot(np.conj(x_0.T))))
    #print(np.hstack([x_0,xt.reshape([K,1])]))
    #return np.linalg.norm(h.dot(xt) - h_0.dot(np.conj(x_0.T)))/np.linalg.norm(h_0.dot(np.conj(x_0.T)))

    plt.figure()

    plt.semilogy(loss)
    plt.xlabel('Iter')
    plt.ylabel('Error')
    plt.show()





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