import numpy as np
import matplotlib.pyplot as plt


class relu:
    def f(x):
        return np.maximum(x, 0)

    def df(y):
        return y > 0


class sigmoid:
    def f(x):
        return 1 / (1 + np.exp(-x))

    def df(y):
        return y * (1 - y)


class tanh:
    def f(x):
        return np.tanh(x)

    def df(y):
        return 1 - y ** 2


class __:
    def f(x):
        return x

    def df(y):
        return 1


def bnd(*args):
    return 1 / np.sqrt(np.prod(args))


def genwbs(*args):
    return np.random.uniform(0, bnd(*args), args)


class NN:
    def __init__(self, dims, fs):
        self.w = np.array([genwbs(n, m) for m, n in zip(dims[:-1], dims[1:])], dtype=object)
        self.b = np.array([genwbs(n) for n in dims[1:]], dtype=object)
        self.l = np.array([np.zeros_like(n) for n in dims], dtype=object)
        self.fs = fs

    def forward(self, x):
        self.l[0] = x
        for i, (w, b, f) in enumerate(zip(self.w, self.b, self.fs)):
            self.l[i + 1] = f.f((np.dot(w, self.l[i]) + b).astype(float))
        return self.l[-1]

    def __call__(self, x):
        return self.forward(x)

    def backward(self, e):
        dw, db = [], []
        for i, (w, f) in enumerate(zip(self.w[::-1], self.fs[::-1])):
            e *= f.df(self.l[-1 - i])
            db.append(e)
            dw.append(np.outer(e, self.l[-2 - i]))
            e = np.dot(w.T, e)
        return np.array(dw[::-1], dtype=object), np.array(db[::-1], dtype=object), e


class RNN(NN):
    def __init__(self, n, xn, yn, xdim, hdim, ydim, dims, fs, hf, yf):
        super().__init__([xdim + hdim] + dims + [ydim + hdim], fs + [__])
        self.n = n
        self.xn = xn
        self.yn = yn
        self.xdim = xdim
        self.hdim = hdim
        self.ydim = ydim
        self.hf = hf
        self.yf = yf
        self.ls = []
        self.h = genwbs(hdim)

    def rnnforward(self, xs):
        h = self.h
        xs = np.concatenate((xs, np.zeros((self.n - self.xn, self.xdim))))
        for x in xs:
            yh = self.forward(np.concatenate((x, self.hf.f(h))))
            y, h = self.yf.f(yh[:self.ydim]), yh[-self.hdim:]
            self.l[-1][:self.ydim], self.l[-1][-self.hdim:] = y, h
            self.ls.append(self.l)
        return np.array([l[-1][:self.ydim] for l in self.ls[-self.yn:]])

    def __call__(self, xs):
        return self.rnnforward(xs)

    def rnnbackward(self, yes):
        dws, dbs = [], []
        yes = np.concatenate((np.zeros((self.xn, self.ydim)), yes))
        he = np.zeros_like(self.h)
        for ye, l in zip(yes[::-1], self.ls[::-1]):
            self.l = l
            ye *= self.yf.df(l[-1][:self.ydim])
            e = np.concatenate((ye, he))
            dw, db, e = self.backward(e)
            he = e[-self.hdim:] * self.hf.df(l[0][-self.hdim:])
            dws.append(dw)
            dbs.append(db)
        return sum(dws), sum(dbs)
