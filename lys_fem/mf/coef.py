import sympy as sp
import numpy as np
from . import mfem


def generateCoefficient(coefs, dim):
    """
    Generate MFEM coefficient from sympy expression dictionary.
    coefs should be a dictionary such as {1: x, 2: y, "default": 0}.
    The keys indicate domain/boundary attributes. "default" key is used as default value. 
    """
    if isinstance(coefs, mfem.GridFunction):
        return ScalarCoef(coefs, dim)
    if isinstance(coefs, dict):
        return generateCoefficient(_parseDictToSympy(coefs), dim)
    shape = np.shape(coefs)
    if len(shape) == 0:
        return ScalarCoef(coefs, dim)
    if len(shape) == 1:
        return VectorCoef(coefs, dim)
    elif len(shape) == 2:
        return MatrixCoef(coefs, dim)
    
def _parseDictToSympy(coefs):
    d = sp.Symbol("domain")
    shape = np.shape(coefs[list(coefs.keys())[0]])
    if len(shape) == 0:
        tuples = [(value, sp.Eq(d, sp.Integer(key))) for key, value in coefs.items() if key != "default"]
        if "default" in coefs:
            tuples = tuples + [(coefs["default"], True)]
        else:
            tuples = tuples + [(tuples[0][0], True)]
        return sp.Piecewise(*tuples)
    else:
        return [_parseDictToSympy({key: value[i] for key, value in coefs.items()}) for i in range(shape[0])]


class ScalarCoef(mfem.PyCoefficient):
    def __init__(self, coefs, dim):
        super().__init__()
        self._dim = dim
        self._coefs = coefs
        if isinstance(coefs, mfem.GridFunction):
            self._funcs = mfem.GridFunctionCoefficient(coefs)
        else:
            self._funcs = sp.lambdify(_generateArgs(dim), coefs)

    def Eval(self, T, ip):
        if isinstance(self._coefs, mfem.GridFunction):
            return self._funcs.Eval(T,ip)
        else:
            self._attr = T.Attribute
            return super().Eval(T, ip)

    def EvalValue(self, x):
        return float(self._funcs(self._attr, *x))


class VectorCoef(mfem.VectorPyCoefficient):
    def __init__(self, coefs, dim):
        super().__init__(dim)
        self._coefs = np.array([generateCoefficient(c,dim) for c in coefs])
        self._dim = dim

    def __getitem__(self, index):
        return self._coefs[index]

    @staticmethod
    def fromScalars(coefs):
        return VectorCoef([c._coefs for c in coefs], coefs[0]._dim)

    def Eval(self, K, T, ip):
        for c in self._coefs:
            c._attr =T.Attribute
        K.Set(1, np.array([c.Eval(T,ip) for c in self._coefs]))


class MatrixCoef(mfem.MatrixPyCoefficient):
    def __init__(self, coefs, dim):
        super().__init__(dim)
        self._coefs = np.array([[generateCoefficient(c,dim) for c in cs] for cs in coefs])
        self._dim = dim

    def __getitem__(self, index):
        return self._coefs[index]

    @staticmethod
    def fromScalars(coefs):
        return MatrixCoef([[c._coefs for c in cs] for cs in coefs], coefs[0][0]._dim)

    def Eval(self, K, T, ip):
        for cs in self._coefs:
            for c in cs:
                c._attr = T.Attribute
        K.Set(1, mfem.DenseMatrix(np.array([[c.Eval(T,ip) for c in cs] for cs in self._coefs])))


def _generateArgs(dim):
    d,x,y,z = sp.symbols("domain,x,y,z")
    if dim == 1:
        return d,x
    elif dim == 2:
        return d,x,y
    elif dim == 3:
        return d,x,y,z

