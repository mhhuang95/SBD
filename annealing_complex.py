import numpy as np
import matplotlib.pyplot as plt


class annealing(object):
    def __init__(self, m, k):
        self.m = m
        self.k = k
        self.max_iter = 100
        self.epsilon = 1e-5
        self.a0 = np.zeros(k,dtype=complex)
        rams = np.random.randn(k)
        self.a0 = np.cos(rams) + 1j * np.sin(rams)
        self.a0 /= np.linalg.norm(self.a0)
        shuf = np.arange(32)
        np.random.shuffle(shuf)

        self.x0 = np.zeros(m,dtype=complex)
        self.x0[shuf[0:3]] = np.random.randn(3)+1j*np.random.randn(3)
        self.y = np.fft.ifft(np.fft.fft(self.x0)*np.fft.fft(self.a0,m))
        self.yhat = np.fft.fft(self.y)

        self.lams = [0.5, 0.1,0.01]
        self.lam = 0.5


    def init_a(self):
        start = np.random.randint(0, self.m)
        ainit = np.hstack([self.y, self.y])[start:(start + self.k)]
        ainit = np.hstack([np.ones(self.k - 1,dtype=complex), ainit, np.ones(self.k - 1,dtype=complex)])
        ainit /= np.abs(ainit)
        return ainit

    def soft(self, x, lam):
        for i in range(x.shape[0]):
            if x[i] != 0+0j:
                x[i] =np.maximum(np.abs(x[i]) - lam, 0) * (x[i]/np.abs(x[i]))
        return x

    def Exp(self, a, de):
        res = np.zeros(a.shape[0],dtype=complex)

        nde = np.abs(de)
        mask = (nde > 0)
        res[mask] = a[mask] * np.cos(nde[mask]) + de[mask] * np.sin(nde[mask])/nde[mask]
        res[1-mask] = a[1-mask]
        return res

    def proj2tan(self, a, g):
        return g - np.real(np.conj(a) * g) * a

    def calc_grad(self):

        max_it = 100
        tol = 1e-4
        cost = 1
        cost_ = 0

        ahat = np.fft.fft(self.a, self.m)
        a2hat = np.abs(ahat) ** 2
        ayhat = np.conj(ahat) * self.yhat
        s = 0.99 / np.max(a2hat)

        w = np.fft.fft(self.x)
        xhat_ = w
        t = 1
        it = 0

        while it < max_it and np.abs(cost - cost_) > tol:
            self.x = self.soft(np.real(np.fft.ifft(w - s * (a2hat * w - ayhat))), s * self.lam)

            t_ = (1 + np.sqrt(1 + 4 * t * t)) / 2
            xhat = np.fft.fft(self.x)
            w = xhat + (t - 1) / t_ * (xhat - xhat_)

            t = t_
            xhat_ = xhat
            it += 1

            cost_ = cost
            cost = 0.5 * np.linalg.norm(np.fft.ifft(ahat * xhat) - self.y) ** 2 + self.lam * np.sum(np.abs(self.x))
            self.costs.append(cost)

        xhat = np.fft.fft(self.x)
        g_a = np.fft.ifft(np.conj(xhat) * (xhat * np.fft.fft(self.a, self.m) - self.yhat))
        g_a = g_a[:self.a.shape[0]]

        return g_a, xhat

    def step(self):

        g_a, xhat = self.calc_grad()
        g_a = self.proj2tan(self.a, g_a)

        t = 0.99 / np.max(np.abs(xhat)) ** 2

        self.a = self.Exp(self.a, -t * g_a)
        self.a /= np.abs(self.a)

        ahat = np.fft.fft(self.a, self.m)
        obj = 0.5 * np.linalg.norm(np.fft.ifft(ahat * xhat) - self.y) ** 2 + self.lam * np.sum(np.abs(self.x))
        self.costs.append(obj)
        return g_a, t

    def solve(self):

        self.x = np.random.randn(self.m)+1j*np.random.randn(self.m)
        self.a = self.init_a()

        print("Kernel a: max_i|<s_i[a_0],a>| = ", maxdoshift(s.a0, s.a))
        print(np.linalg.norm(s.x0 - s.x))


        ahat = np.fft.fft(self.a, self.m)
        xhat = np.fft.fft(self.x)
        obj = 0.5 * np.linalg.norm(np.fft.ifft(ahat * xhat) - self.y) ** 2 + self.lam * np.sum(np.abs(self.x))

        self.costs = [obj]

        for lam in self.lams:
            i = 0
            g_a = np.ones(self.a.shape[0], dtype=complex)
            t = 0.1
            while i < self.max_iter and np.linalg.norm(g_a)**2 * t > self.epsilon:
            #while np.linalg.norm(g_a) ** 2 * t > self.epsilon:
                g_a, t= self.step()
                i += 1


def maxdoshift(a0, a):
    res = float("inf")
    for i in range(a.shape[0] - a0.shape[0]+1):
        res = min(res, np.sum(np.abs(a[i:i+a0.shape[0]] - a0)), np.sum(np.abs(a[i:i+a0.shape[0]] + a0)))
    return res


if __name__ == "__main__":

    m = 100
    k = 20
    s = annealing(m,k)
    s.solve()
    print("Kernel a: max_i|<s_i[a_0],a>| = ", maxdoshift(s.a0, s.a))
    print(np.linalg.norm(s.x0 - s.x))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(s.costs, 'k', label='obj')
    plt.show()