#Code for Rapid, Robust, and Reliable Blind Deconvolution via Nonconvex Optimization
# Minhui Huang

import numpy as np
from scipy.linalg import dft
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.linalg import svds

def calG(B,h,rho,mu):
    L,K = B.shape[0], B.shape[1]
    vaG = 0
    devG = np.zeros([K ,1],dtype=complex)
    for l in range(L):
        bh = B[l].dot(h)
        vaG += (max(0, L*np.abs(bh)**2/mu -1))**2
        devG += max(0, 2 * (L * np.abs(bh) **2 / mu - 1)) * bh * np.conj(B[l].reshape([K,1]))
    vaG = rho*vaG
    devG = rho*devG
    return vaG, devG


def Wir_flow_reg(B,C,h_0,x_0):



    tol = 1e-10
    t0 = 1
    alpha = 0.3
    beta = 0.5
    rho = 5
    mu = 10
    maxit = 1000






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
        valG, devG = calG(B, h, rho, mu)

        Gh = G.dot(np.conj(xt.T)) - devG
        Gxt = np.conj(h.T).dot(G)

        t = t0
        h0 = h
        xt0 = xt
        R0 = R
        valG0 = valG

        while np.sum(R.real**2 + R.imag**2) + np.real(valG) > np.sum(R0.real**2 + R0.imag**2)+ np.real(valG0) - alpha*t*(np.sum(Gh.real**2 + Gh.imag**2) + np.sum(Gxt.real**2 + Gxt.imag**2)):

            h = h0 + t * Gh
            xt = xt0 + t * Gxt
            R = y - A_op(h, np.conj(xt.T))

            t = beta * t
            print('Grad, iter:', iter, t)

        alpha = np.linalg.norm(xt) / np.linalg.norm(h)
        h = alpha * h
        xt = 1 / alpha * xt

        G = A_op_T(R)

        loss.append(np.linalg.norm(R)/np.linalg.norm(y))

        if np.linalg.norm(R) < reltol:
            break

    print(np.linalg.norm(h.dot(xt) - h_0.dot(np.conj(x_0.T)))/np.linalg.norm(h_0.dot(np.conj(x_0.T))))
    #print(np.hstack([x_0,xt.reshape([K,1])]))
    return np.linalg.norm(h.dot(xt) - h_0.dot(np.conj(x_0.T)))/np.linalg.norm(h_0.dot(np.conj(x_0.T)))
    '''
    plt.figure()

    plt.semilogy(loss)
    plt.show()
    '''


def Wir_flow(B,C,h_0,x_0):
    #initialization

    tol = 1e-10
    t0 = 1
    alpha = 0.3
    beta = 0.5
    maxit = 1000

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
            print('Grad, iter:', iter, t)

        alpha = np.linalg.norm(xt) / np.linalg.norm(h)
        h = alpha * h
        xt = 1 / alpha * xt

        G = A_op_T(R)

        loss.append(np.linalg.norm(R)/np.linalg.norm(y))

        if np.linalg.norm(R) < reltol:
            break

    print(np.linalg.norm(h.dot(xt) - h_0.dot(np.conj(x_0.T)))/np.linalg.norm(h_0.dot(np.conj(x_0.T))))
    #print(np.hstack([x_0,xt.reshape([K,1])]))
    return np.linalg.norm(h.dot(xt) - h_0.dot(np.conj(x_0.T)))/np.linalg.norm(h_0.dot(np.conj(x_0.T)))
    '''
    plt.figure()

    plt.semilogy(loss)
    plt.show()
    '''




if __name__ == "__main__":

    coes = np.linspace(1, 4, 10)
    suss_rate = []
    reg_suss_rate= []
    for coe in coes:

        K = 50
        N = 50
        L = int(coe * (K + N))
        dft_mat = dft(L)
        B = dft_mat[:, :K] / np.sqrt(L)
        C = np.random.randn(L, N) + 1j * np.random.randn(L, N)

        # ground truth

        h_0 = np.random.randn(K, 1) + 1j * np.random.randn(K, 1)
        h_0 = h_0 / np.linalg.norm(h_0)
        x_0 = np.random.randn(N, 1) + 1j * np.random.randn(N, 1)
        x_0 = x_0 / np.linalg.norm(x_0)

        exp = 10
        res_reg = 0
        res= 0
        if Wir_flow_reg(B, C, h_0, x_0):
            res_reg +=1
        reg_suss_rate.append(res_reg / exp)

        if Wir_flow(B, C, h_0, x_0):
            res += 1
        suss_rate.append(res / exp)


    plt.figure()
    plt.plot(coes, suss_rate)
    plt.plot(coes, reg_suss_rate)
    plt.legend('grad','grad_reg')
    plt.xlabel('L/(K+N)')
    plt.ylabel('Sucessful rate')
    plt.show()
