import numpy as np
import matplotlib.pyplot as plt


class annealing(object):
    def __init__(self, m, k,sparsity):
        self.m = m
        self.k = k
        self.max_iter = 1000
        self.epsilon = 1e-4
        self.costs = []
        rams = np.random.randn(k)
        self.a0 = np.cos(rams) + 1j * np.sin(rams)
        shuf = np.arange(32)
        np.random.shuffle(shuf)
        self.x0 = np.zeros(m,dtype=complex)
        self.x0[shuf[0:sparsity]] = 3*np.random.randn(sparsity)+1j*3*np.random.randn(sparsity)
        self.yhat = np.fft.fft(self.x0)*np.fft.fft(self.a0,m)
        self.y = np.fft.ifft(self.yhat)


        self.lam = 25
        self.x = 3*np.random.randn(self.m) + 1j*3 * np.random.randn(self.m)
        # self.a = self.init_a()
        '''
        rams = np.random.randn(k)
        self.a = self.a0 + 0.01*(np.random.randn(self.m) + 1j * np.random.randn(self.m))
        self.a /= np.abs(self.a)
        '''
        self.a = self.init_a()


    def init_a(self):
        
        start = np.random.randint(0, self.m)
        ainit = np.hstack([self.y, self.y])[start:(start + self.k)]
        #ainit = np.hstack([np.zeros(self.k - 1,dtype=complex), ainit, np.zeros(self.k - 1,dtype=complex)])
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

    def f(self, x_k, a_k):
        return 0.5 * np.linalg.norm(np.fft.ifft(np.fft.fft(x_k)*np.fft.fft(a_k, self.m)) - self.y) ** 2

    def model_x(self, x, x_k, a, lam):
        return self.f(x_k, a) + np.conj(np.fft.ifft(np.conj(np.fft.fft(a, self.m))*(np.fft.fft(x_k)*np.fft.fft(a, self.m)-self.yhat)).T).dot(x-x_k) + (1.0 / (2.0 * lam)) * np.conj((x-x_k).T).dot(x-x_k)

    def model_a(self, a, a_k,x, lam):
        return self.f(x, a_k) + np.conj(np.fft.ifft(np.conj(np.fft.fft(x))*(np.fft.fft(x)*np.fft.fft(a_k, self.m)-self.yhat)).T)[:self.a.shape[0]].dot(a-a_k) + (1.0 / (2.0 * lam)) * np.conj((a-a_k).T).dot(a-a_k)

    def circulant_mat(self, a):
        a = np.hstack([a, np.zeros(self.m-self.k,dtype = complex )])
        res = np.zeros([self.m, self.m],dtype = complex)
        res[:,0] = a
        for i in range(1,self.m):
            res[:, i] = np.hstack([a[-i:], a[:self.m-i]])
        return res

    def calc_grad(self):

        tol = 1e-2
        cost = 1
        cost_ = 0
        it = 0
        max_it = 1000
        ahat = np.fft.fft(self.a, self.m)



        g_x = np.ones(self.m, dtype=complex)
        #g_x_ = np.zeros(self.m, dtype=complex)
        xhat = np.fft.fft(self.x)
        #Ca = self.circulant_mat(self.a)
        #tmp = np.linalg.solve(Ca, self.y)

        while it < max_it and np.abs(cost - cost_) > tol:

            g_x_ = g_x
            s = 0.1

            g_x = np.fft.ifft(np.conj(ahat)*(xhat*ahat-self.yhat))
            while self.f(self.x - s * g_x,self.a) > self.model_x(self.x - s * g_x, self.x, self.a, s):
                s = s * 0.7

            self.x = self.soft(self.x - s* g_x, s*self.lam)
            xhat = np.fft.fft(self.x)

            cost_ = cost
            cost = 0.5 * np.linalg.norm(np.fft.ifft(ahat * xhat) - self.y) ** 2 + self.lam * np.sum(np.abs(self.x))
            self.costs.append(cost)
            print(cost)

            it+=1
            #print('t',np.linalg.norm(tmp - self.x) / np.linalg.norm(tmp), cost)

        xhat = np.fft.fft(self.x)
        g_a = np.fft.ifft(np.conj(xhat) * (xhat * np.fft.fft(self.a, self.m) - self.yhat))[:self.a.shape[0]]
        #print(np.linalg.norm(g_a) ** 2)

        return g_a, xhat

    def step(self):

        g_a, xhat = self.calc_grad()
        g_a = self.proj2tan(self.a, g_a)


        t = 0.1
        while self.f(self.x, self.a - t * g_a) > self.model_a(self.a - t * g_a, self.a, self.x, t):
            t = t * 0.7

        g_a = np.fft.ifft(np.conj(xhat) * (xhat * np.fft.fft(self.a, self.m) - self.yhat))[:self.a.shape[0]]
        self.a = self.a -t * g_a

        self.a /= np.abs(self.a)
        #print(np.linalg.norm(self.a - self.a0) / np.linalg.norm(self.a0))

        ahat = np.fft.fft(self.a, self.m)
        obj = 0.5 * np.linalg.norm(np.fft.ifft(ahat * xhat) - self.y) ** 2 + self.lam * np.sum(np.abs(self.x))
        self.costs.append(obj)
        print(obj)
        return g_a

    def solve(self):

        ahat = np.fft.fft(self.a, self.m)
        xhat = np.fft.fft(self.x)
        obj = 0.5 * np.linalg.norm(np.fft.ifft(ahat * xhat) - self.y) ** 2 + self.lam * np.sum(np.abs(self.x))

        self.costs.append(obj)

        i = 0
        g_a = np.ones(self.a.shape[0], dtype=complex)
        g_a_ = np.zeros(self.a.shape[0], dtype=complex)


        while np.abs(np.linalg.norm(g_a) ** 2 - np.linalg.norm(g_a_)**2)  > self.epsilon:
        #for i in range(1):
            g_a_ = g_a
            g_a= self.step()
            i += 1



def maxdoshift(a0, a):
    res = float("inf")
    for i in range(a.shape[0] - a0.shape[0]+1):
        res = min(res, np.sum(np.abs(a[i:i+a0.shape[0]] - a0)), np.sum(np.abs(a[i:i+a0.shape[0]] + a0)))
    return res

def is_shift(l1,l2):
    if len(l1)!= len(l2):
        return False
    ls = len(l1)
    for i in range(ls):
        tmp = l1[i:] + [128 + x for x in l1[:i]]
        tmp_set = []
        for i, x in enumerate(tmp):
            if x - l2[i] not in tmp_set:
                tmp_set.append(x-l2[i])
        if len(tmp_set) == 1:
            return True
    return False

if __name__ == "__main__":

    m = 128
    k = 128
    sparsity = 3
    dec=5
    t = 0

    '''

    err_y = []
    reerr_y = []
    err_a = []
    err_x = []


    while t < 20:

        s = annealing(m,k,sparsity)
        s.solve()

        reerr_y.append(np.linalg.norm(s.circulant_mat(s.a).dot(s.x) -  s.y)/np.linalg.norm(s.y))
        err_y.append(np.linalg.norm(s.circulant_mat(s.a).dot(s.x) - s.y))
        #print(np.hstack([s.x.reshape([m,1]), s.x0.reshape([m,1])]))
        #print(s.x[np.abs(s.x) > 0.0001] )
        res = []
        for i, j in enumerate(np.abs(s.x) > 0.0001):
            if j == True:
                res.append(i)
        #print(res)
        #print(s.x0[np.abs(s.x0) > 0.0001])
        res0 = []
        for i, j in enumerate(np.abs(s.x0) > 0.0001):
            if j == True:
                res0.append(i)
        #print(res0)

        if len(res) == sparsity and res[0] - res0[0] == res[1] - res0[1] == res[2] - res0[2]:
            t+=1
            shift = res[0] - res0[0]
            if shift != 0:
                a_sh = np.hstack([s.a[-shift:], s.a[:(k - shift)%k]])
                x_sh = np.hstack([s.x[shift:], s.x[:shift]])
            else:
                a_sh = s.a
                x_sh = s.x

            if a_sh.shape[0] != x_sh.shape[0]:
                print(shift)
                print(a_sh.shape[0] , x_sh.shape[0])

            err_x.append(np.linalg.norm(s.x0 - x_sh)/np.linalg.norm(s.x0))


            err_a.append(np.linalg.norm(s.a0 - a_sh)/np.linalg.norm(s.a0))

            print(t,'& ',np.around(s.lam * np.sum(np.abs(s.x0)), dec),'&', np.around(s.costs[-1], dec),'&', res0, '&', res, '&',res[0] - res0[0],'&',
                  np.around(np.linalg.norm(s.circulant_mat(s.a).dot(s.x) - s.y) / np.linalg.norm(s.y),dec), '&',
                  np.around(np.linalg.norm(s.circulant_mat(s.a).dot(s.x) - s.y) ,dec), '&',
                  np.around(np.linalg.norm(s.x0 - x_sh) / np.linalg.norm(s.x0),dec),'&',np.around(np.linalg.norm(s.a0 - a_sh)/np.linalg.norm(s.a0),dec))
    #print('%.5f'%(sum(reerr_y)/len(reerr_y)))
    print('Ave','&','-','&','-','&','-','&','-','&%.5f'% (sum(reerr_y)/len(reerr_y)),'&%.5f'%(sum(err_y)/len(err_y)),'&%.5f'%(sum(err_a)/len(err_a)),'&%.5f'%(sum(err_x)/len(err_x)))

    '''

    s = annealing(m, k,sparsity)
    s.solve()
    print(s.lam * np.sum(np.abs(s.x)))

    print(np.linalg.norm(s.circulant_mat(s.a).dot(s.x) -  s.y)/np.linalg.norm(s.y))
    print(np.linalg.norm(s.circulant_mat(s.a).dot(s.x) - s.y))
    #print(np.hstack([s.x.reshape([m,1]), s.x0.reshape([m,1])]))
    print(s.x[np.abs(s.x) > 0.0001] )
    res = []
    for i, j in enumerate(np.abs(s.x) > 0.0001):
        if j == True:
            res.append(i)
    print(res)
    print(s.x0[np.abs(s.x0) > 0.0001])
    res0 = []
    for i, j in enumerate(np.abs(s.x0) > 0.0001):
        if j == True:
            res0.append(i)
    print(res0)
    if is_shift(res, res0):
        t += 1
        shift = res[0] - res0[0]
        if shift != 0:
            a_sh = np.hstack([s.a[-shift:], s.a[:(k - shift) % k]])
            x_sh = np.hstack([s.x[shift:], s.x[:shift]])
        else:
            a_sh = s.a
            x_sh = s.x


        print(np.linalg.norm(s.x0 - x_sh))
        print(np.linalg.norm(s.x0 - x_sh)/np.linalg.norm(s.x0))

        print(np.linalg.norm(s.a0 - a_sh))
        print(np.linalg.norm(s.a0 - a_sh)/np.linalg.norm(s.a0))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(s.costs, 'k', label='obj')
    plt.show()
