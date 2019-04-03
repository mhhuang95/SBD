import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

from sklearn.datasets import load_digits

class ManPG(object):
    def __init__(self, m, k,a0,x0,a,x):
        self.m = m
        self.k = k
        self.theta = 1 / k
        self.max_iter = 1000
        self.epsilon = 1e-5

        self.a0 = a0
        self.x0 = x0
        self.y = np.real(np.fft.ifft(np.fft.fft(x0)*np.fft.fft(a0,m)))
        self.yhat = np.fft.fft(self.y)
        self.a = a
        self.x = x

        self.lams = [0.5, 0.1]
        self.lam = 0.5

    def init_a(self):
        start = np.random.randint(0, self.m)
        ainit = np.hstack([self.y,self.y])[start:(start + self.k)]
        ainit /= np.linalg.norm(ainit)
        ainit = np.hstack([np.zeros(self.k-1), ainit, np.zeros(self.k-1)])
        return ainit

    def conv(self, x,a):
        xhat = np.fft.fft(x)
        ahat = np.fft.fft(a,self.m)
        return np.fft.ifft(xhat*ahat)

    def soft(self,x,lam):
        return np.sign(x)*np.maximum(np.abs(x)-lam,0)

    def Exp(self,a,de):
        nde = np.linalg.norm(de)
        if nde > 0:
            return np.cos(nde) * a + np.sin(nde)*de/nde
        else:
            return a

    def proj2tan(self,a,g):
        return g-a.T.dot(g)*a

    def calc_grad(self):

        ahat = np.fft.fft(self.a, self.m)
        a2hat = np.abs(ahat)**2
        ayhat = np.conj(ahat) * self.yhat
        s = 0.99/np.max(a2hat)

        xhat = np.fft.fft(self.x)

        self.x = self.soft(np.real(np.fft.ifft(xhat-s*(a2hat*xhat-ayhat))),s*self.lam)

        g_a = np.real(np.fft.ifft(np.conj(xhat) * (xhat * ahat - self.yhat)))
        g_a = g_a[:self.a.shape[0]]

        return g_a, xhat

    def F(self,a,lam):
        xhat = np.fft.fft(self.x)
        return 0.5*np.linalg.norm(np.fft.ifft(np.fft.fft(a,self.m) * xhat) - self.y)**2 + lam*np.sum(np.abs(self.x))

    def step(self):

        g_a, xhat = self.calc_grad()
        g_a = self.proj2tan(self.a, g_a)

        t = 0.99/np.max(np.abs(xhat)**2)
        d_a = t*(self.a.T.dot(g_a))*self.a - t * g_a


        alpha = 1
        '''
        delta = 0.7
        gamma = 0.5
        while self.F(self.a+alpha*d_a, self.lam) > self.F(self.a,self.lam)-delta*alpha*np.linalg.norm(d_a)**2:
            alpha = gamma * alpha
        '''
        self.a =self.Exp(self.a,alpha*d_a)
        self.a /= np.linalg.norm(self.a)

        ahat = np.fft.fft(self.a,self.m)
        obj = 0.5 * np.linalg.norm(np.fft.ifft(ahat * xhat) - self.y) ** 2 + self.lam * np.sum(np.abs(self.x))

        return g_a,t, obj

    def solve(self):


        self.a /= np.linalg.norm(self.a)
        ahat = np.fft.fft(self.a, self.m)
        xhat = np.fft.fft(self.x)
        obj = 0.5 * np.linalg.norm(np.fft.ifft(ahat * xhat) - self.y) ** 2 + self.lam * np.sum(np.abs(self.x))

        self.costs = [obj]

        for lam in self.lams:
            self.lam = lam
            i = 0
            g_a = np.ones(self.a.shape[0])
            t = 0.1
            while np.linalg.norm(g_a)**2 * t > self.epsilon:
                g_a, t,obj = self.step()
                i+=1
                self.costs.append(obj)
            print(i)


def maxdoshift(a0,a):
    k = a0.shape[0]
    p = a.shape[0]

    res = 0
    for i in range(p-k):
        res = max(res, np.abs(np.sum(np.hstack([np.zeros(i), a0, np.zeros(p-k-i)])*a)))
    return res


def init_a(m,k,y):
    start = np.random.randint(0, m)
    ainit = np.hstack([y,y])[start:(start + k)]
    ainit = ainit/np.linalg.norm(ainit)
    ainit = np.hstack([np.zeros(k-1), ainit, np.zeros(k-1)])
    return ainit


if __name__ == "__main__":


    res = []
    m = 255*255
    k = 20
    img = np.array(mpimg.imread('x.jpg'))
    img = np.array(img / np.max(img))
    i1 = np.gradient(img)[0]
    i1 = i1 / np.max(i1)
    print(np.sum(i1 < 0.1) / (255 * 255))
    img = img*(img > 0.5)
    print(np.sum(img == 0)/(255*255))
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(1)

    x0 = img
    ker = np.array(mpimg.imread('f.jpg'))
    a0 = (ker/np.linalg.norm(ker))

    #for i in range(1):

    y = np.real(np.fft.ifft(np.fft.fft(x0)*np.fft.fft(a0,m)))

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(y)

    '''
    y = y.flatten()
    a = init_a(m,k,y)
    x = np.random.randn(m)
    s = ManPG(m,k,a0, x0,a,x)
    start_time = time.time()
    s.solve()
    print("Kernel a: max_i|<s_i[a_0],a>| = ", maxdoshift(s.a0, s.a))


    recover = s.x
    recover = recover.reshape([255,255])
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(recover)
    '''
    plt.show()






    '''

    m = 100000
    k = 200
    a0 = np.random.randn(k)
    a0 /= np.linalg.norm(a0)
    theta = 1 / k
    x0 = (np.random.rand(m) <= theta) * np.random.randn(m)
    y = np.real(np.fft.ifft(np.fft.fft(x0) * np.fft.fft(a0, m)))
    a = init_a(m, k, y)
    x = np.random.randn(m)

    s = ManPG(m,k, a0,x0,a,x)
    start_time = time.time()
    s.solve()
    print(s.a0)
    print("Kernel a: max_i|<s_i[a_0],a>| = ", maxdoshift(s.a0, s.a))


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(s.a0,'k',label='a0')
    ax.plot(s.a,'k--',label='a')

    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(s.costs, 'k', label='obj')
    
    plt.show()
    '''