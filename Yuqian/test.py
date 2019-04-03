import numpy as np
import matplotlib.pyplot as plt


class annealing(object):
    def __init__(self, m, k):
        self.m = m
        self.k = k
        self.max_iter = 1000
        self.epsilon = 1e-4
        rams = np.random.randn(k)
        self.a0 = np.cos(rams) + 1j * np.sin(rams)
        shuf = np.arange(32)
        np.random.shuffle(shuf)
        self.x0 = np.zeros(m, dtype=complex)
        self.x0[shuf[0:3]] = np.random.randn(3) + 1j * np.random.randn(3)
        self.yhat = np.fft.fft(self.x0) * np.fft.fft(self.a0, m)
        self.y = np.fft.ifft(self.yhat)

        self.lams = [0.5]
        self.lam = 0.5
        self.x = np.random.randn(self.m) + 1j * np.random.randn(self.m)
        # self.a = self.init_a()
        rams = np.random.randn(k)
        self.a = np.cos(rams) + 1j * np.sin(rams)

    def init_a(self):

        start = np.random.randint(0, self.m)
        ainit = np.hstack([self.y, self.y])[start:(start + self.k)]
        # ainit = np.hstack([np.zeros(self.k - 1,dtype=complex), ainit, np.zeros(self.k - 1,dtype=complex)])
        ainit /= np.abs(ainit)
        return ainit

    def soft(self, x, lam):
        mask = (np.abs(x) > 0)
        x[mask] = np.maximum(np.abs(x[mask]) - lam, 0) * (x[mask] / np.abs(x[mask]))
        return x

    def Exp(self, a, de):
        res = np.zeros(a.shape[0], dtype=complex)

        nde = np.abs(de)
        mask = (nde > 0)
        res[mask] = a[mask] * np.cos(nde[mask]) + de[mask] * np.sin(nde[mask]) / nde[mask]
        mask = (nde == 0)
        res[mask] = a[mask]
        return res

    def proj2tan(self, a, g):
        return g - np.real(np.conj(a) * g) * a

    def f(self, x_k, a_k):
        return 0.5 * np.linalg.norm(np.fft.ifft(np.fft.fft(x_k) * np.fft.fft(a_k, self.m)) - self.y) ** 2

    def model_x(self, x, x_k, a, lam):
        return self.f(x_k, a) + np.conj(
            np.fft.ifft(np.conj(np.fft.fft(a, self.m)) * (np.fft.fft(x_k) * np.fft.fft(a, self.m) - self.yhat)).T).dot(
            x - x_k) + (1.0 / (2.0 * lam)) * np.conj((x - x_k).T).dot(x - x_k)

    def model_a(self, a, a_k, x, lam):
        return self.f(x, a_k) + np.conj(
            np.fft.ifft(np.conj(np.fft.fft(x)) * (np.fft.fft(x) * np.fft.fft(a_k, self.m) - self.yhat)).T)[
                                :self.a.shape[0]].dot(a - a_k) + (1.0 / (2.0 * lam)) * np.conj((a - a_k).T).dot(a - a_k)

    def circulant_mat(self, a):
        res = np.zeros([self.m, self.m], dtype=complex)
        res[:, 0] = a
        for i in range(1, self.m):
            res[:, i] = np.hstack([a[-i:], a[:self.m - i]])
        return res

    def calc_grad(self):

        tol = 1e-2
        cost = 1
        cost_ = 0
        ahat = np.fft.fft(self.a, self.m)

        g_x = np.ones(self.m, dtype=complex)
        g_x_ = np.zeros(self.m, dtype=complex)
        xhat = np.fft.fft(self.x)
        Ca = self.circulant_mat(self.a)
        self.x = self.soft(np.linalg.solve(Ca, self.y), 0.01)


        xhat = np.fft.fft(self.x)
        g_a = np.fft.ifft(np.conj(xhat) * (xhat * np.fft.fft(self.a, self.m) - self.yhat))[:self.a.shape[0]]
        print(np.linalg.norm(g_a) ** 2)

        return g_a, xhat

    def step(self):

        g_a, xhat = self.calc_grad()
        g_a = self.proj2tan(self.a, g_a)

        t = 0.1
        while self.f(self.x, self.a - t * g_a) > self.model_a(self.a - t * g_a, self.a, self.x, t):
            t = t * 0.7

        g_a = np.fft.ifft(np.conj(xhat) * (xhat * np.fft.fft(self.a, self.m) - self.yhat))[:self.a.shape[0]]
        self.a = self.a - t * g_a

        self.a /= np.abs(self.a)
        # print(np.linalg.norm(self.a - self.a0) / np.linalg.norm(self.a0))

        ahat = np.fft.fft(self.a, self.m)
        obj = 0.5 * np.linalg.norm(np.fft.ifft(ahat * xhat) - self.y) ** 2 + self.lam * np.sum(np.abs(self.x))
        self.costs.append(obj)
        return g_a

    def solve(self):

        ahat = np.fft.fft(self.a, self.m)
        xhat = np.fft.fft(self.x)
        obj = 0.5 * np.linalg.norm(np.fft.ifft(ahat * xhat) - self.y) ** 2 + self.lam * np.sum(np.abs(self.x))

        self.costs = [obj]

        for lam in self.lams:
            i = 0
            g_a = np.ones(self.a.shape[0], dtype=complex)
            g_a_ = np.zeros(self.a.shape[0], dtype=complex)

            # while i < self.max_iter and np.linalg.norm(g_a)**2 * t > self.epsilon:
            while np.abs(np.linalg.norm(g_a) ** 2 - np.linalg.norm(g_a_) ** 2) > self.epsilon:
                g_a_ = g_a
                g_a = self.step()
                i += 1


def maxdoshift(a0, a):
    res = float("inf")
    for i in range(a.shape[0] - a0.shape[0] + 1):
        res = min(res, np.sum(np.abs(a[i:i + a0.shape[0]] - a0)), np.sum(np.abs(a[i:i + a0.shape[0]] + a0)))
    return res


if __name__ == "__main__":
    m = 128
    k = 128

    s = annealing(m, k)
    xinit = s.x
    ainit = s.a
    s.solve()

    print(np.linalg.norm(s.circulant_mat(s.a).dot(s.x) - s.y) / np.linalg.norm(s.y))
    print(np.hstack([s.x.reshape([m, 1]), s.x0.reshape([m, 1])]))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(s.costs, 'k', label='obj')
    plt.show()