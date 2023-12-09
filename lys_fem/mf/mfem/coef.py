import sympy as sp
import numpy as np

from . import mfem_orig

def generateCoefficient(coefs, dim=None):
    """
    Generate MFEM coefficient from sympy expression dictionary.
    coefs should be a dictionary such as {1: x, 2: y, "default": 0}.
    The keys indicate domain/boundary attributes. "default" key is used as default value. 
    """
    from . import GridFunction
    if isinstance(coefs, dict):
        return generateCoefficient(_parseDictToSympy(coefs), dim)
    elif isinstance(coefs, (int, float, sp.Integer, sp.Float)):
        return ConstantCoefficient(float(coefs))
    elif isinstance(coefs, GridFunction):
        return GridFunctionCoefficient(coefs)

    shape = np.shape(coefs)
    if len(shape) == 0:
        return SympyCoefficient(coefs)
    if len(shape) == 1:
        return VectorArrayCoefficient(coefs)
    elif len(shape) == 2:
        return MatrixArrayCoefficient(coefs)
    
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



class _ScalarCoefBase:
    def __neg__(self):
        return -1.0 * self

    def __pow__(self, value):
        return PowerCoefficient(self, float(value))
    
    def __add__(self, value):
        return SumCoefficient(value, self)

    def __sub__(self, value):
        return SumCoefficient(self, value, 1.0, -1.0)

    def __mul__(self, value):
        return ProductCoefficient(self, value)

    def __rmul__(self, value):
        return ProductCoefficient(value, self)

    def __truediv__(self, value):
        return self*value**(-1)
    
    def __rtruediv__(self, value):
        return value * self**(-1)


class GridFunctionCoefficient(mfem_orig.GridFunctionCoefficient, _ScalarCoefBase):
    def __init__(self, gf):
        super().__init__(gf)
        self._obj = self
        self._gf = gf


class ConstantCoefficient(mfem_orig.ConstantCoefficient, _ScalarCoefBase):
    def __init__(self, value):
        super().__init__(float(value))
        self._obj = self
        self._value = value


class SumCoefficient(mfem_orig.SumCoefficient, _ScalarCoefBase):
    def __init__(self, v1, v2, *args):
        super().__init__(v1, v2, *args)
        self._obj = self
        self._v1 = v1
        self._v2 = v2


class ProductCoefficient(mfem_orig.ProductCoefficient, _ScalarCoefBase):
    def __init__(self, v1, v2):
        super().__init__(v1, v2)
        self._obj = self
        self._v1 = v1
        self._v2 = v2


class PowerCoefficient(mfem_orig.PowerCoefficient, _ScalarCoefBase):
    def __init__(self, v1, v2):
        super().__init__(v1, v2)
        self._obj = self
        self._v1 = v1
        self._v2 = v2


class SympyCoefficient(mfem_orig.PyCoefficient, _ScalarCoefBase):
    def __init__(self, coefs):
        super().__init__()
        self._obj = self
        self._coefs = coefs
        self._funcs = sp.lambdify(sp.symbols("domain,x,y,z"), coefs)
        self._func = None

    def Eval(self, T, ip):
        self._attr = T.Attribute
        return super().Eval(T, ip)

    def EvalValue(self, x):
        if self._func is None:
            if len(x) == 1:
                self._func = lambda d, x: self._funcs(d,x,0,0)
            elif len(x) == 2:
                self._func = lambda d,x,y: self._funcs(d,x,y,0)
            else:
                self._func = self._funcs
        return float(self._func(self._attr, *x))


class VectorArrayCoefficient(mfem_orig.VectorArrayCoefficient):
    def __init__(self, coefs):
        super().__init__(len(coefs))
        self._objs = self
        self._obj = coefs
        self._coefs = [c if isinstance(c, _ScalarCoefBase) else generateCoefficient(c) for c in coefs]
        for i, c in enumerate(self._coefs):
            self.Set(i,c,own=False)

    def __getitem__(self, index):
        return self._coefs[index]


class MatrixArrayCoefficient(mfem_orig.MatrixArrayCoefficient):
    def __init__(self, coefs):
        super().__init__(len(coefs))
        self._objs = self
        self._obj = coefs
        self._coefs = [[c if isinstance(c, _ScalarCoefBase) else generateCoefficient(c) for c in cs] for cs in coefs]
        for i, cs in enumerate(self._coefs):
            for j, c in enumerate(cs):
                self.Set(i,j,c,own=False)

    def __getitem__(self, index):
        return self._coefs[index[0]][index[1]]

