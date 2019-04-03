#Code for Rapid, Robust, and Reliable Blind Deconvolution via Nonconvex Optimization
# Minhui Huang

import numpy as np
from scipy.linalg import dft
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.linalg import svds

def soft(x, lam):
        mask = (np.abs(x) > 0)
        x[mask] =np.maximum(np.abs(x[mask]) - lam, 0) * (x[mask]/np.abs(x[mask]))
        return x



def main():
    #initialization


    K=200
    N=200
    sp = 3
    L = int(2 * (K + N))


    tol = 1e-10
    t0 = 1
    alpha = 0.3
    beta = 0.5
    maxit = 100000
    lam =  1

    dft_mat = dft(L)


    B = dft_mat[:,:K]/np.sqrt(L)
    C = np.random.randn(L,N) + 1j* np.random.randn(L,N)

    #ground truth

    h_0 = np.random.randn(K,1) + 1j*np.random.randn(K,1)
    h_0 = h_0/np.linalg.norm(h_0)
    shuf = np.arange(32)
    np.random.shuffle(shuf)
    x_0 = np.zeros([N,1], dtype=complex)
    x_0[shuf[:3]] = np.random.randn(sp,1)+ 1j*np.random.randn(sp,1)
    x_0 = x_0/np.linalg.norm(x_0)

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
    #print(u.shape)
    #print(vh.shape)

    h = np.sqrt(d) * u
    x = np.sqrt(d) * np.conj(vh.T)

    reltol = tol * np.linalg.norm(y)
    R = y - A_op(h,x)
    G = A_op_T(R)

    loss = []

    for iter in range(maxit):
        Gh = G.dot(x)
        Gx = np.conj(h.T).dot(G).reshape([N,1])


        h0 = h
        x0 = x

        t = 1
        while f(h0, x0 + t * Gx) > f(h0,x0) + np.conj(Gx.T).dot(t*Gx)+ 1/(2*t)*(np.linalg.norm(t*Gx)**2):

            t = beta * t
            print('Grad, iter:', iter, t)

        x = x0 + t * Gx
        #x = soft(x, t * lam)

        s = 1
        while f(h0 + s * Gh, x0) > f(h0,x0) + np.conj(Gh.T).dot(s*Gh)+ 1/(2*s)*(np.linalg.norm(s*Gh)**2):

            s = beta * s
            print('Grad, iter:', iter, s)

        h = h0 + s * Gh

        R = y - A_op(h, x)

        G = A_op_T(R)

        loss.append(np.linalg.norm(R)/np.linalg.norm(y))

        if np.linalg.norm(R) < reltol:
            break

        print(np.linalg.norm(h.dot(np.conj(x.T)) - h_0.dot(np.conj(x_0.T)))/np.linalg.norm(h_0.dot(np.conj(x_0.T))))
    print(np.hstack([x_0,x]))
    plt.figure()

    plt.semilogy(loss)
    plt.show()




if __name__ == "__main__":
    main()