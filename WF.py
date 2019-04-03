
import numpy as np

def compute_grad(x_k, A, b):
    grad_l2 = np.conj(A.T).dot(A.dot(x_k) - b)          #Wirtinger derivative of  ||Ax-b||_2^2
    return grad_l2

def f(x_k,A,b):

    return 0.5*np.linalg.norm(A.dot(x_k)-b)**2


def F(x_k,A,b,tau):
    return 0.5 * np.linalg.norm(A.dot(x_k) - b) ** 2 + tau*np.sum(np.abs(x_k))

def model(x,xk,A,b,GammaK):

    innerProd = np.conj(compute_grad(xk,A,b).T).dot(x - xk)
    xDiff = x - xk
    return f(xk,A,b) + innerProd + (1.0/(2.0*GammaK))*np.conj(xDiff.T).dot(xDiff)

def circulant_mat(a,m):
    res = np.zeros([m, m],dtype = complex)
    res[:,0] = a
    for i in range(1,m):
        res[:, i] = np.hstack([a[-i:], a[:m-i]])
    return res

def main():
    # initialization
    m = 128                                           #Number of measurements
    n = 128
    s = 3
    #A = np.random.randn(m, n) + 1j * np.random.randn(m, n)

    rams = np.random.randn(m)
    a0 = np.cos(rams) + 1j * np.sin(rams)
    A = circulant_mat(a0,m)

    xs = np.zeros([n, 1], dtype=complex)
    picks = np.random.permutation(np.arange(1, n))
    xs[picks[0:s], 0] = np.random.randn(s, 1).flatten() + 1j * np.random.randn(s, 1).flatten()

    #xs = np.random.randn(m, 1) + 1j * np.random.randn(m, 1)
    b = A.dot(xs)                                       #b = Ax
    print(np.linalg.norm(xs - np.linalg.solve(A,b)) / np.linalg.norm(xs))

    x_k = np.random.rand(n, 1) + 1j * np.random.rand(n, 1)
    x_k /= np.linalg.norm(x_k)

    epsilon = 1e-5
    beta = 0.7
    tau = 0.1
    g_x = 1

    for i in range(5000) :
        t = 0.1
        g_x = compute_grad(x_k, A, b)

        while f(x_k - t * g_x, A, b) > model(x_k - t * g_x, x_k, A, b, t):
            t = beta * t
        x_k = x_k - t * g_x
        mask = np.abs(x_k) > 0
        x_k[mask] = np.maximum(np.abs(x_k[mask]) - tau * t, 0) * (x_k[mask] / np.abs(x_k[mask]))
        #print(np.linalg.norm(xs -  x_k)/np.linalg.norm(xs))
        print(F(x_k,A,b,tau))

    print(np.hstack([xs,  x_k]))


if __name__ == "__main__":
    main()