import numpy as np
import matplotlib.pyplot as plt


class anealing(object):
    def __init__(self, m, k):
        self.m = m
        self.k = k
        self.theta = 2 / k
        self.max_iter = 1000
        self.epsilon = 1e-3

        self.a0 = np.random.randn(k)
        self.a0 /= np.linalg.norm(self.a0)
        self.x0 = (np.random.rand(m) <= self.theta) * np.random.randn(m)
        self.y = np.real(self.conv(self.x0,self.a0))
        self.yhat = np.fft.fft(self.y)

        self.lams = [0.5, 0.1, 0.01]
        self.lam = 0.5

        '''
        self.max_lam = 0.5
        self.min_lam = 0.00001
        self.t = 0
        self.lam = 0.5
        self.num_steps = 20000
        self.dlam = (self.max_lam - self.min_lam)/self.num_steps
        '''

    def init_a(self):
        start = 55
        ainit = np.hstack([self.y,self.y])[start:(start + self.k)]
        ainit = np.hstack([np.zeros(self.k - 1), ainit, np.zeros(self.k - 1)])
        ainit /= np.linalg.norm(ainit)
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

    def evaluate(self,a,lam):
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
            self.x = self.soft(np.real(np.fft.ifft(w-s*(a2hat*w-ayhat))),s*self.lam)

            t_ = (1 + np.sqrt(1 + 4 * t * t)) / 2
            xhat = np.fft.fft(self.x)
            w = xhat + (t - 1) / t_ * (xhat - xhat_)

            t = t_
            xhat_ = xhat
            it+=1

            cost_ = cost
            cost = 0.5*np.linalg.norm(np.real(np.fft.ifft(ahat*xhat)) - self.y)**2+self.lam * np.sum(np.abs(self.x))
            self.costs.append(cost)
        return cost

    def calc_grad(self):

        cost = self.evaluate(self.a, self.lam)

        xhat = np.fft.fft(self.x)
        g_a = np.real(np.fft.ifft(np.conj(xhat) * (xhat * np.fft.fft(self.a, self.m) - self.yhat)))
        g_a = g_a[:self.a.shape[0]]

        return g_a, cost, xhat

    def step(self):

        g_a, cost,xhat = self.calc_grad()
        g_a = self.proj2tan(self.a,g_a)

        t = 0.99/np.max(np.abs(xhat))**2

        self.a =self.Exp(self.a,-t*g_a)
        self.a /= np.linalg.norm(self.a)

        ahat = np.fft.fft(self.a, self.m)
        obj = 0.5 * np.linalg.norm(np.real(np.fft.ifft(ahat * xhat)) - self.y) ** 2 + self.lam * np.sum(np.abs(self.x))
        return g_a, cost,t, obj

    def solve(self):

        self.x = np.random.randn(self.m)
        self.a = self.init_a()

        ahat = np.fft.fft(self.a, self.m)
        xhat = np.fft.fft(self.x)
        obj = 0.5 * np.linalg.norm(np.real(np.fft.ifft(ahat * xhat)) - self.y) ** 2 + self.lam * np.sum(np.abs(self.x))

        self.costs = [obj]

        self.a = self.calc_grad()[0]
        #print(self.a)
        self.a = -1*self.a
        self.a /= np.linalg.norm(self.a)
        g_a = np.ones(self.a.shape[0])
        t = 0.1


        for lam in self.lams:
            i = 0
            g_a = np.ones(self.a.shape[0])
            t = 0.1
            # while i < self.max_iter and np.linalg.norm(g_a)**2 * t > self.epsilon:
            while np.linalg.norm(g_a) ** 2 * t > self.epsilon:
                # while i < self.max_iter:
                g_a, cost, t, obj = self.step()
                i += 1
        #print("--- %s seconds ---" % (time.time() - start))

def maxdoshift(a0,a):
    return np.max(np.abs(np.correlate(a0, a, "full")))


if __name__ == "__main__":
    '''
    res = []

    for i in range(20):
        s = anealing(10000, 20)
        start_time = time.time()
        s.solve()
        print("Experiment",i+1,"running time: --- %s seconds ---" % (time.time() - start_time))
        res.append(maxdoshift(s.a0, s.a))
        print("Kernel a: max_i|<s_i[a_0],a>| = ", maxdoshift(s.a0, s.a))

    print("average max_i|<s_i[a_0],a>|:", sum(res) / len(res))

    '''


    s = anealing(10000,20)
    s.solve()
    print("Kernel a: max_i|<s_i[a_0],a>| = ",maxdoshift(s.a0,s.a))
    print(np.linalg.norm(s.x0 - s.x))
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(s.a0,'r-',label='a0')
    ax.plot(s.a,'b-',label='a')


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(s.costs, 'k', label='obj')
    plt.show()