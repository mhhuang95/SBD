import numpy as np
import matplotlib.pyplot as plt
import time


def maxdoshift(a0, a):
    return np.max(np.abs(np.correlate(a0, a, "full")))
class ManPG(object):
    def __init__(self, m, k,x0,a0,x,a):
        self.m = m
        self.k = k
        self.theta = 1 / k
        self.max_iter = 1000
        self.epsilon = 1e-7

        self.a0 = a0
        self.x0 = x0
        self.y = np.real(np.fft.ifft(np.fft.fft(x0)*np.fft.fft(a0,m)))
        self.yhat = np.fft.fft(self.y)
        self.a = a
        self.x = x

        self.lams = [0.1]
        self.lam = 0.1

    def init_a(self):
        start = np.random.randint(0, self.m)
        ainit = np.hstack([self.y,self.y])[start:(start + self.k)]
        ainit /= np.linalg.norm(ainit)
        ainit = np.hstack([np.zeros(self.k-1), ainit, np.zeros(self.k-1)])
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
        return 0.5*np.linalg.norm(np.real(np.fft.ifft(np.fft.fft(a,self.m) * xhat)) - self.y)**2 + lam*np.sum(np.abs(self.x))

    def step(self):

        g_a, xhat = self.calc_grad()
        g_a = self.proj2tan(self.a, g_a)

        t = 0.99/np.max(np.abs(xhat)**2)
        d_a = t*(self.a.T.dot(g_a))/(self.a.T.dot(self.a)) - t * g_a

        alpha = 1
        delta = 0.7
        gamma = 0.5
        while self.F(self.a+alpha*d_a, self.lam) > self.F(self.a,self.lam)-delta*alpha*np.linalg.norm(d_a)**2:
            alpha = gamma * alpha

        self.a =self.Exp(self.a,alpha*d_a)
        self.a /= np.linalg.norm(self.a)

        ahat = np.fft.fft(self.a,self.m)
        obj = 0.5 * np.linalg.norm(np.real(np.fft.ifft(ahat * xhat)) - self.y) ** 2 + self.lam * np.sum(np.abs(self.x))

        return g_a,t, obj

    def solve(self):


        #start = time.time()

        self.a /= np.linalg.norm(self.a)
        ahat = np.fft.fft(self.a, self.m)
        xhat = np.fft.fft(self.x)
        obj = 0.5 * np.linalg.norm(np.real(np.fft.ifft(ahat * xhat)) - self.y) ** 2 + self.lam * np.sum(np.abs(self.x))

        self.costs = [obj]

        for lam in self.lams:
            self.lam = lam
            i = 0
            g_a = np.ones(self.a.shape[0])
            t = 0.1
            while i < self.max_iter and np.linalg.norm(g_a)**2 * t > self.epsilon:
                g_a, t,obj = self.step()
                i+=1
                self.costs.append(obj)
        #print("--- %s seconds ---" % (time.time() - start))
        '''
        while 1 - maxdoshift(self.a0, self.a) > 0.0001:
            g_a, t, obj = self.step()
            self.costs.append(obj)
        '''

class anealing(object):
    def __init__(self, x0,a0,x,a):
        self.m = x0.shape[0]
        self.k = a0.shape[0]
        self.theta = 1 / self.k
        self.max_iter = 1000
        self.epsilon = 1e-3
        self.x0 = x0
        self.a0 = a0
        self.y =  np.real(np.fft.ifft(np.fft.fft(x0)*np.fft.fft(a0,m)))
        self.yhat = np.fft.fft(self.y)
        self.lams = [0.1]
        self.lam = 0.1
        self.x = x
        self.a = a


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

        #start = time.time()
        ahat = np.fft.fft(self.a, self.m)
        xhat = np.fft.fft(self.x)
        obj = 0.5 * np.linalg.norm(np.real(np.fft.ifft(ahat * xhat)) - self.y) ** 2 + self.lam * np.sum(np.abs(self.x))
        self.costs = [obj]

        self.a = self.calc_grad()[0]
        #print(self.a)
        self.a = -1*self.a
        self.a /= np.linalg.norm(self.a)

        for lam in self.lams:
            self.lam = lam
            i = 0
            g_a = np.ones(self.a.shape[0])
            t = 0.1
            while i < self.max_iter and np.linalg.norm(g_a)**2 * t > self.epsilon:
            #while i < self.max_iter:
                g_a, cost,t,obj = self.step()
                i+=1
                #self.costs.append(obj)
        #print("--- %s seconds ---" % (time.time() - start))
        '''
        while 1 - maxdoshift(self.a0, self.a) > 0.0001:
            g_a,cost, t, obj = self.step()
        '''

def init_a(m,k,y):
    start = np.random.randint(0, m)
    ainit = np.hstack([y,y])[start:(start + k)]
    ainit /= np.linalg.norm(ainit)
    ainit = np.hstack([np.zeros(k-1), ainit, np.zeros(k-1)])
    return ainit


if __name__ == "__main__":

    '''
    k=200
    m=100000
    accu_ane = []
    accu_man = []
    time_ane = []
    time_man = []
    iter_ane = []
    iter_man = []

    for i in range(20):
        a0 = np.random.randn(k)
        a0 /= np.linalg.norm(a0)
        theta = 1 / k
        x0 = (np.random.rand(m) <= theta) * np.random.randn(m)
        y = np.real(np.fft.ifft(np.fft.fft(x0) * np.fft.fft(a0, m)))
        x = np.random.randn(m)
        a = init_a(m,k,y)

        s1 = anealing(x0,a0,x,a)
        start1 = time.time()
        s1.solve()
        t_anealing = time.time() - start1
        time_ane.append(t_anealing)
        accu_ane.append(1-maxdoshift(s1.a0, s1.a))

        s2 = ManPG(m,k,x0,a0,x,a)
        start2 = time.time()
        s2.solve()
        t_manpg = time.time() - start2
        time_man.append(t_manpg)
        accu_man.append(1-maxdoshift(s2.a0, s2.a))
        print(i + 1, '& %.6f'%t_manpg,'& %.6f'%t_anealing, '& %f'%(1- maxdoshift(s2.a0, s2.a)),'& %f'%(1-maxdoshift(s1.a0, s1.a)),'&',len(s2.costs),'&',len(s1.costs))
        #print(i+1,'&',len(s2.costs),'&',len(s1.costs))
        iter_ane.append(len(s1.costs))
        iter_man.append(len(s2.costs))


    print("Average",'& %.6f'%(sum(time_man) / len(time_man)),'& %.6f'%(sum(time_ane) / len(time_ane)),'& %f'%(sum(accu_man) / len(accu_man)),'& %f'%(sum(accu_ane) / len(accu_ane)),'&',sum(iter_man) / len(iter_man),'&', sum(iter_ane) / len(iter_ane) )
    #print("Average",'&',sum(iter_man) / len(iter_man),'&', sum(iter_ane) / len(iter_ane))

    '''
    m = 10000
    k = 20
    a0 = np.random.randn(k)
    a0 /= np.linalg.norm(a0)
    theta = 1 / k
    x0 = (np.random.rand(m) <= theta) * np.random.randn(m)
    y = np.real(np.fft.ifft(np.fft.fft(x0) * np.fft.fft(a0, m)))
    x = np.random.randn(m)
    a = init_a(m, k, y)
    s1 = anealing(x0, a0, x, a)
    s1.solve()
    print("ane Kernel a: max_i|<s_i[a_0],a>| = ", maxdoshift(s1.a0, s1.a))
    print(np.linalg.norm(s1.x0 - s1.x)/np.linalg.norm(s1.x0))
    per = 0
    for l in s1.x:
        if abs(l) < 1e-3:
            per += 1
    print(per / m)
    print(s1.costs[-1])
    s2 = ManPG(m,k,x0, a0, x, a)

    s2.solve()
    print("man Kernel a: max_i|<s_i[a_0],a>| = ", maxdoshift(s2.a0, s2.a))
    print(np.linalg.norm(s2.x0 - s2.x)/np.linalg.norm(s2.x0))
    per = 0
    for l in s2.x:
        if abs(l) < 1e-3:
            per += 1
    print(per/m)
    print(s2.costs[-1])

    print(s1.costs[len(s2.costs)-1])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(s1.costs, 'b-', label='Anealing')
    ax.semilogy(s2.costs, 'r-', label='ManPG')
    plt.title("Objective function for ManPG and Anealing, m=1000000,k=2000")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function")
    plt.legend()
    plt.show()

