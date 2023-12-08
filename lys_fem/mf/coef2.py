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
        self._mfem = True
        if isinstance(coefs, (int, float, sp.Integer, sp.Float)):
            print("const")
            self._funcs = mfem.ConstantCoefficient(float(coefs))
            self.Eval = self._funcs.Eval
        elif isinstance(coefs, (mfem.SumCoefficient, mfem.ProductCoefficient, mfem.PowerCoefficient)):
            self._funcs = coefs
            self.Eval = self._funcs.Eval
        elif isinstance(coefs, mfem.GridFunction):
            self._funcs = mfem.GridFunctionCoefficient(coefs)
            self.Eval = self._funcs.Eval
        else:
            self._funcs = sp.lambdify(_generateArgs(dim), coefs)
            self.Eval = self._Eval
            self._mfem = False

    def _Eval(self, T, ip):
        self._attr = T.Attribute
        return super().Eval(T, ip)

    def EvalValue(self, x):
        return float(self._funcs(self._attr, *x))

    def __neg__(self):
        return self * (-1)

    def __pow__(self, value):
        v1 = tomf(self)
        obj = mfem.PowerCoefficient(v1, float(value))
        obj._v1 = v1
        return ScalarCoef(obj, self._dim)

    def __add__(self, value):
        return genCoef(mfem.SumCoefficient, self._dim, self, value)

    def __sub__(self, value):
        return genCoef(mfem.SumCoefficient, self._dim, self, value, 1.0, -1.0)

    def __mul__(self, value):
        return genCoef(mfem.ProductCoefficient, self._dim, self, value)

    def __rmul__(self, value):
        return genCoef(mfem.ProductCoefficient, self._dim, value, self)

    def __truediv__(self, value):
        return self*value**(-1)
    
    def __rtruediv__(self, value):
        return value * self**(-1)

def genCoef(cls, dim, value1, value2, *args):
    v1, v2 = tomf(value1), tomf(value2)
    obj = cls(v1, v2, *args)
    obj._v1 = v1
    obj._v2 = v2
    return ScalarCoef(obj, dim)

def tomf(value):
    if isinstance(value, ScalarCoef):
        if value._mfem:
            return value._funcs
        else:
            return value
    else:      
        return mfem.ConstantCoefficient(value)


class VectorCoef(mfem.VectorPyCoefficient):
    def __init__(self, coefs, dim):
        super().__init__(dim)
        self._coefs = np.array([generateCoefficient(c,dim) for c in coefs])
        self._dim = dim
        self._func = mfem.VectorArrayCoefficient(len(self._coefs))
        for i, c in enumerate(self._coefs):
            self._func.Set(i,c)
        self.Eval = self._func.Eval

    def __getitem__(self, index):
        return self._coefs[index]

    @staticmethod
    def fromScalars(coefs):
        return VectorCoef([c._coefs for c in coefs], coefs[0]._dim)


class MatrixCoef(mfem.MatrixPyCoefficient):
    def __init__(self, coefs, dim):
        super().__init__(dim)
        self._coefs = np.array([[generateCoefficient(c,dim) for c in cs] for cs in coefs])
        self._dim = dim
        self._func = mfem.MatrixArrayCoefficient(len(self._coefs))
        for i, cs in enumerate(self._coefs):
            for j, c in enumerate(cs):
                self._func.Set(i,j,c)
        self.Eval = self._func.Eval

    def __getitem__(self, index):
        return self._coefs[index]

    @staticmethod
    def fromScalars(coefs):
        return MatrixCoef([[c._coefs for c in cs] for cs in coefs], coefs[0][0]._dim)


def _generateArgs(dim):
    d,x,y,z = sp.symbols("domain,x,y,z")
    if dim == 1:
        return d,x
    elif dim == 2:
        return d,x,y
    elif dim == 3:
        return d,x,y,z

