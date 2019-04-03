import numpy as np
import matplotlib.pyplot as plt


class ManPG(object):
    def __init__(self, m, k):
        self.m = m
        self.k = k
        self.max_iter = 1000
        self.epsilon = 1e-1
        self.a0 = np.zeros(k, dtype=complex)
        rams = np.random.randn(k)
        self.a0 = np.cos(rams) + 1j * np.sin(rams)
        shuf = np.arange(32)
        np.random.shuffle(shuf)
        self.x0 = np.zeros(m, dtype=complex)
        self.x0[shuf[0:3]] = np.random.randn(3) + 1j * np.random.randn(3)
        self.yhat = np.fft.fft(self.x0) * np.fft.fft(self.a0, m)
        self.y = np.fft.ifft(self.yhat)

        self.lams = [50]
        self.lam = 50

    def init_a(self):
        start = np.random.randint(0, self.m)
        ainit = np.hstack([self.y, self.y])[start:(start + self.k)]
        #ainit = np.hstack([np.ones(self.k - 1,dtype=complex), ainit, np.ones(self.k - 1,dtype=complex)])
        ainit /= np.abs(ainit)
        return ainit

    def soft(self, x, lam):
        mask = (np.abs(x) > 0)
        x[mask] =np.maximum(np.abs(x[mask]) - lam, 0) * (x[mask]/np.abs(x[mask]))
        return x

    def Exp(self, a, de):
        res = np.zeros(a.shape[0],dtype=complex)

        nde = np.abs(de)
        mask = (nde > 0)
        res[mask] = a[mask] * np.cos(nde[mask]) + de[mask] * np.sin(nde[mask])/nde[mask]
        mask = (nde == 0)
        res[mask] = a[mask]
        return res


    def proj2tan(self, a, g):
        return g - np.real(np.conj(a) * g) * a


    def F(self, a, lam):
        xhat = np.fft.fft(self.x)
        return 0.5 * np.linalg.norm(np.fft.ifft(np.fft.fft(a, self.m) * xhat) - self.y) ** 2 + lam * np.sum(
            np.abs(self.x))

    def rever(self, sig):
        i = 1
        j = sig.shape[0]-1
        while i < j:
            sig[i], sig[j] = sig[j], sig[i]
            i+=1
            j-=1
        return sig

    def f(self, x_k, a_k):
        return 0.5 * np.linalg.norm(np.fft.ifft(np.fft.fft(x_k)*np.fft.fft(a_k, self.m)) - self.y) ** 2

    def model_x(self, x, x_k, a, lam):
        return self.f(x_k, a) + np.conj(np.fft.ifft(np.conj(np.fft.fft(a, self.m))*(np.fft.fft(x_k)*np.fft.fft(a, self.m)-self.yhat)).T).dot(x-x_k) + (1.0 / (2.0 * lam)) * np.conj((x-x_k).T).dot(x-x_k)


    def step(self):

        ahat = np.fft.fft(self.a, self.m)

        xhat = np.fft.fft(self.x)
        g_x =np.fft.ifft( np.conj(ahat) * (ahat*xhat-self.yhat))

        g_a = np.fft.ifft(np.conj(xhat) * (ahat * xhat - self.yhat))

        g_a = self.proj2tan(self.a, g_a)

        t = 0.1

        while self.f(self.x - t * g_x, self.a) > self.model_x(self.x - t * g_x, self.x, self.a, t):
            t = t * 0.7

        self.x = self.soft(self.x - t* 2*g_x, t * self.lam)

        Lag_mul = t*(2*g_a.real*self.a.real + (-2)*g_a.imag*self.a.imag)
        da_r = Lag_mul * self.a.real - t*2*g_a.real
        da_i = Lag_mul * self.a.imag - t*(-2)*g_a.imag

        d_a = da_r +1j*da_i

        alpha = 1

        delta = 0.7
        gamma = 0.5
        while self.F(self.a+alpha*d_a, self.lam) > self.F(self.a,self.lam)- delta*alpha*np.linalg.norm(d_a)**2:
            alpha = gamma * alpha

        self.a = self.Exp(self.a, alpha * d_a)


        self.a /= np.abs(self.a)

        ahat = np.fft.fft(self.a, self.m)
        obj = 0.5 * np.linalg.norm(np.fft.ifft(ahat * xhat) - self.y) ** 2 + self.lam * np.sum(np.abs(self.x))
        #print(obj)

        return g_a, t, obj

    def solve(self):

        self.x = np.random.randn(self.m) + 1j * np.random.randn(self.m)
        self.a = self.init_a()

        ahat = np.fft.fft(self.a, self.m)
        xhat = np.fft.fft(self.x)
        obj = 0.5 * np.linalg.norm(np.fft.ifft(ahat * xhat) - self.y) ** 2 + self.lam * np.sum(np.abs(self.x))

        self.costs = [obj]

        for lam in self.lams:
            self.lam = lam
            i = 0
            g_a = np.ones(self.a.shape[0], dtype=complex)
            g_a_ = np.zeros(self.a.shape[0], dtype=complex)
            t = 0.1
            while np.abs(np.linalg.norm(g_a) ** 2 - np.linalg.norm(g_a_)**2)  > self.epsilon:
                g_a_ = g_a
                g_a, t, obj = self.step()
                i += 1
                self.costs.append(obj)
                #print(obj)
            #print(i)

    def circulant_mat(self, a):
        res = np.zeros([self.m, self.m],dtype = complex)
        res[:,0] = a
        for i in range(1,self.m):
            res[:, i] = np.hstack([a[-i:], a[:self.m-i]])
        return res

def maxdoshift(a0, a):
    k = a0.shape[0]
    p = a.shape[0]

    res = 0
    for i in range(p - k+1):
        res = max(res, np.abs(np.sum(np.hstack([np.zeros(i), a0, np.zeros(p - k - i)]) * a)))
    return res


if __name__ == "__main__":
    '''
    m = 128
    k = 128
    s = ManPG(m, k)
    s.solve()
    print(np.hstack([s.x0.reshape([m,1]), s.x.reshape([m,1])]))
    print(s.lam * np.sum(np.abs(s.x0)))

    print("Kernel a error = ", maxdoshift(s.a0, s.a))
    print(np.linalg.norm(s.x0 - s.x))

    print(np.linalg.norm(s.circulant_mat(s.a).dot(s.x) - s.y)/np.linalg.norm(s.y))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(s.costs, 'k', label='obj')
    '''

    dec = 5
    m = 128
    k = 128
    for i in range(20):
        s = ManPG(m, k)
        s.solve()
        print(i+1,'&', np.around(s.lam * np.sum(np.abs(s.x0)),dec),'&', np.around(s.costs[-1],dec))
