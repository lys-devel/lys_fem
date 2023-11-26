import numpy as np
from numpy import *
from . import mfem


def generateCoefficient(coefs, dim):
    """
    Generate MFEM coefficient from coefs.
    coefs is a dictionary that contains values for domains/boundaries.
    For example, coef = {1: "x", 2:"y"} and dim=3 means f(x,y,z)=x in domain 1, f(x,y,z)=y in domain 2.
    """
    shape = np.array(list(coefs.values()))[0].shape
    if len(shape) == 0:
        return ScalarCoef(coefs, dim)
    elif len(shape) == 1:
        return VectorCoef(coefs, dim, shape[0])
    elif len(shape) == 2:
        return MatrixCoef(coefs, dim, shape[0])


class ScalarCoef(mfem.PyCoefficient):
    def __init__(self, coefs, dim):
        super().__init__()
        self._funcs = _generateFuncs(coefs, dim)
        self._default = 0.0

    def Eval(self, T, ip):
        self._attr = T.Attribute
        return super().Eval(T, ip)

    def EvalValue(self, x):
        return float(_eval_attr(x, self._funcs, self._attr, self._default))


class VectorCoef(mfem.VectorPyCoefficient):
    def __init__(self, coefs, dim, size):
        super().__init__(size)
        self._coefs = coefs
        self._dim = dim
        self._funcs = _generateFuncs(coefs, dim)
        self._default = np.array([0] * size, dtype=float)

    def __getitem__(self, index):
        coefs = {key: value[index] for key, value in self._coefs.items()}
        return ScalarCoef(coefs, self._dim)

    def Eval(self, K, T, ip):
        self._attr = T.Attribute
        return super().Eval(K, T, ip)

    def EvalValue(self, x):
        return _eval_attr(x, self._funcs, self._attr, self._default)


class MatrixCoef(mfem.MatrixPyCoefficient):
    def __init__(self, coefs, dim, size):
        super().__init__(size)
        self._funcs = _generateFuncs(coefs, dim)
        self._default = np.zeros((size, size), dtype=float)

    def Eval(self, K, T, ip):
        self._attr = T.Attribute
        return super().Eval(K, T, ip)

    def EvalValue(self, x):
        return _eval_attr(x, self._funcs, self._attr, self._default)


def _generateFuncs(coefs, dim):
    return {domain: np.vectorize(lambda x: _generateFunc(x, dim))(func) for domain, func in coefs.items()}


def _generateFunc(funcStr, dim):
    vars = "x"
    if dim > 1:
        vars += ",y"
    if dim > 2:
        vars += ",z"
    return eval("lambda " + vars + ": " + funcStr, globals())


def _eval_attr(x, funcs, attr, default):
    if attr not in funcs:
        return default
    else:
        return np.vectorize(lambda f: f(*x), otypes=[float])(funcs[attr])
