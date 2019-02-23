import numpy as np
import matplotlib.pyplot as plt

class anealing(object):
    def __init__(self, m, k):
        self.m = m
        self.k = k
        self.theta = 1 / k
        self.max_iter = 1000
        self.epsilon = 1e-3
        self.beta = 0.7
        self.a0 = np.random.randn(k,1)
        self.a0 /= np.linalg.norm(self.a0)
        self.x0 = np.zeros([m,1])
        shuff = np.arange(m)
        np.random.shuffle(shuff)
        self.x0[shuff[:int(self.m * self.theta)]] = np.random.randn(int(self.m * self.theta), 1)
        self.y = self.conv(self.x0,self.a0)
        self.lams = [0.5*1/np.sqrt(self.k*self.theta), 0.5]
        self.lam = 0.1

    def init_a(self):
        start = np.random.randint(0, self.m)
        ainit = np.vstack([self.y,self.y])[start:(start + self.k), :]
        ainit /= np.linalg.norm(ainit)
        ainit = np.vstack([np.zeros([self.k-1,1]), ainit, np.zeros([self.k-1,1])])
        return ainit

    def con_grad(self, x):
        m = x.shape[0]
        grad = np.zeros([m, m])
        for i in range(m):
            for j in range(m):
                grad[i][j] = x[(i - j) % m,0]
        return grad

    def conv(self, x,a):
        a = np.vstack([a, np.zeros([self.m-self.k,1])])
        C_x = self.con_grad(x)
        return C_x.dot(a)

    def soft(self,x,lam):
        return np.sign(x)*np.maximum(x-lam,0)

    def Exp(self,a,de):
        nde = np.linalg.norm(de)
        if nde > 0:
            return np.cos(nde) * a + np.sin(nde)*de/nde
        else:
            return a

    def proj2tan(self,a,g):
        return g-a.T.dot(g)*a

    def calc_grad(self):
        max_it = 10000
        tol = 1e-4
        it=0
        cost = 1
        cost_ = 0

        s = 0.01
        a_ = np.vstack([self.a, np.zeros([self.m - self.p,1])])
        C_a = self.con_grad(a_)
        g_x = C_a.T.dot(C_a.dot(self.x)-self.y)
        while 0.5*np.linalg.norm(C_a.dot(self.x-s*g_x)-self.y)**2 > 0.5*np.linalg.norm(C_a.dot(self.x)-self.y)**2 -0.5* s*g_x.T.dot(g_x):
            s = self.beta * s

        t = 1
        x_ = self.x
        while it < max_it and cost - cost_ > tol:
            self.x = self.soft(self.x, self.lam*s)

            t_ = (1 + np.sqrt(1 + 4 * t * t)) / 2
            self.x = self.x + (t - 1) / t_ * (self.x - x_)

            t = t_
            x_ = self.x
            it+=1

            cost_ = cost
            cost = 0.5*np.linalg.norm(C_a.dot(self.x)-self.y)+self.lam * np.sum(np.abs(self.x))

        C_x = self.con_grad(self.x)
        g_a = C_x.T.dot(C_x.dot(a_)-self.y)
        g_a = g_a[:self.p,:]

        return g_a,cost

    def step(self):

        g_a, cost = self.calc_grad()
        g_a = self.proj2tan(self.a,g_a)
        g_a_ = np.vstack([g_a, np.zeros([self.m - self.p, 1])])

        a_ = np.vstack([self.a, np.zeros([self.m - self.p, 1])])
        t = 0.01
        C_x = self.con_grad(self.x)
        while 0.5 * np.linalg.norm((C_x.dot(a_ - t * g_a_)) - self.y) ** 2 > 0.5 * np.linalg.norm(
                        C_x.dot(a_) - self.y) ** 2 - 0.5 * t * g_a.T.dot(g_a):
            t = self.beta * t

        self.a =self.Exp(self.a,-t*g_a)
        self.a /= np.linalg.norm(self.a)
        return g_a, cost,t

    def solve(self):

        self.a = self.init_a()
        self.x = np.random.randn(self.m, 1)
        self.p = self.a.shape[0]
        self.a,_ = self.calc_grad()
        self.a = -self.a
        self.a /= np.linalg.norm(self.a)

        costs = []
        for lam in self.lams:
            self.lam = lam
            i = 0
            g_a = np.ones([self.p, 1])
            t = 0.1
            while i < self.max_iter and np.linalg.norm(g_a)**2 * t > self.epsilon:
                g_a, cost,t = self.step()
                i+=1
                #print("Iteration:", i, "norm g_a:", np.linalg.norm(g_a))
                #costs.append(cost)


def maxdoshift(a0,a):
    k = a0.shape[0]
    p = a.shape[0]

    res = 0
    for i in range(p-k):
        res = max(res, np.abs(np.sum(np.vstack([np.zeros([i,1]), a0, np.zeros([p-k-i,1])])*a)))
    return res

if __name__ == "__main__":
    res = []
    for i in range(20):
        s = anealing(1000,20)
        s.solve()
        res.append(maxdoshift(s.a0, s.a))

    print("average max_i|<s_i[a_0],a>|:", sum(res)/len(res))
    #print("Kernel a: max_i|<s_i[a_0],a>| = ",maxdoshift(s.a0,s.a))
    #print(s.a)
    #print(s.a0)

    '''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(s.a0,'k',label='a0')
    ax.plot(s.a,'k--',label='a')
    plt.show()
    '''