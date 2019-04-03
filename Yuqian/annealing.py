import numpy as np
import matplotlib.pyplot as plt



class annealing(object):
    def __init__(self, m,k,theta):

        self.m = m
        self.k = k
        self.theta = theta
        self.epsilon = 1e-3
        self.max_iter = 1000
        self.a0 = np.random.randn(k)
        self.a0 /= np.linalg.norm(self.a0)
        self.x0 = (np.random.rand(m) <= self.theta) * np.random.randn(m)
        self.yhat = np.fft.fft(self.x0)*np.fft.fft(self.a0,m)
        print(np.linalg.norm(np.fft.ifft(np.fft.fft(self.x0) * np.fft.fft(self.a0,m) - self.yhat)))
        self.y = np.real(np.fft.ifft(self.yhat))
        self.lams = [0.5*np.sqrt(self.k*self.theta), 0.5, 0.1, 0.01]
        self.lam = 0.5
        self.x = np.random.randn(m)
        self.a = self.ainit()
        self.costs = []
        #self.a = -1 * self.calc_grad()[0]
        #self.a /= np.linalg.norm(self.a)


    def ainit(self):
        start = np.random.randint(0, self.m)
        ainit = np.hstack([self.y, self.y])[start:(start + self.k)]
        ainit = np.hstack([np.zeros(self.k - 1), ainit, np.zeros(self.k - 1)])
        ainit /= np.linalg.norm(ainit)
        return ainit

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

    def circulant_mat(self, a):
        res = np.zeros([self.m, self.m])
        res[:, 0] = a
        for i in range(1, self.m):
            res[:, i] = np.hstack([a[-i:], a[:self.m - i]])
        return res

    def calc_grad(self):
        max_it = 100
        tol = 1e-4
        it=0
        cost = 1
        cost_ = 0

        ahat = np.fft.fft(self.a, self.m)
        a2hat = np.abs(ahat)**2
        ayhat = np.conj(ahat) * self.yhat
        s = 0.99/np.max(a2hat)

        w = np.fft.fft(self.x)
        xhat_ = w
        t=1
        '''
        Ca = self.circulant_mat(np.hstack([self.a, np.zeros(self.m-self.a.shape[0])]))
        self.x=np.linalg.solve(Ca, self.y)
        '''


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

        ahat = np.fft.fft(self.a, self.m)
        xhat = np.fft.fft(self.x)
        obj = 0.5 * np.linalg.norm(np.real(np.fft.ifft(ahat * xhat)) - self.y) ** 2 + self.lam * np.sum(np.abs(self.x))
        self.costs.append(obj)

        for lam in self.lams:
            self.lam = lam
            i = 0
            g_a = np.ones(self.a.shape[0])
            t = 0.1
            while i < self.max_iter and np.linalg.norm(g_a)**2 * t > self.epsilon:
                g_a, cost,t,obj = self.step()
                i+=1



def maxdoshift(a0, a):
    return np.max(np.abs(np.correlate(a0, a, "full"))), np.argmax(np.abs(np.correlate(a0, a, "full")))


if __name__ == "__main__":

    '''
    m = 500000
    ks = np.linspace(1000, 10000, 10)
    thetas = np.logspace(-3, -1, 10)
    res = np.zeros([10, 10])
    
    F = open("a_anne.txt", "w")

    for i in range(1):
        for l, k in enumerate(ks):
            for j, theta in enumerate(thetas):
                s = annealing(m, int(k), theta)
                s.solve()
                res[j][l] += 1 - maxdoshift(s.a0, s.a)
                F.write(str(1 - maxdoshift(s.a0, s.a)))
            F.write('\n')

    F.close()
    '''
    m = 10000
    k = 20
    theta = 0.1
    s = annealing(m, int(k), theta)
    s.solve()
    err, idx = maxdoshift(s.a0,s.a)
    print(err)
    print(np.linalg.norm(np.fft.ifft(np.fft.fft(s.x) * np.fft.fft(s.a, m) - s.yhat)))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(s.a0, 'r-', label='a0')
    ax.plot(s.a, 'b-', label='a')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(s.x0, 'r-', label='x0')
    ax.plot(s.x, 'b-', label='x')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(s.costs)
    plt.show()
