import sympy as sp
from . import mfem


def generateCoefficient(coefs, mesh):
    """
    Generate MFEM coefficient from sympy expression dictionary.
    coefs should be a dictionary such as {1: x, 2: y, "default": 0}.
    The keys indicate domain/boundary attributes. "default" key is used as default value. 
    """
    if isinstance(coefs, dict):
        return generateCoefficient(_parseDictToSympy(coefs), mesh)
    dim = mesh.SpaceDimension()
    shape = _checkShape(coefs)
    if len(shape) == 0:
        return ScalarCoef(coefs, dim)
    if len(shape) == 1:
        return VectorCoef(coefs, dim)
    elif len(shape) == 2:
        return MatrixCoef(coefs, dim)

def _checkShape(value):
    if isinstance(value, sp.Piecewise):
        return _checkShape(value.subs("domain", -1))
    if isinstance(value, (sp.Float, sp.Integer, float, int)):
        return tuple()
    else:
        shape = sp.shape(value)
        if len(shape) == 2:
            if shape[1] == 1:
                shape=(shape[0],)
        return shape
    
def _parseDictToSympy(coefs):
    def convert(val):
        if isinstance(val, (int, float, sp.Basic)):
            return val
        else:
            return sp.Matrix(val)
    d = sp.Symbol("domain")
    tuples = [(convert(value), sp.Eq(d, sp.Integer(key))) for key, value in coefs.items() if key != "default"]
    if "default" in coefs:
        tuples = tuples + [(convert(coefs["default"]), True)]
    else:
        tuples = tuples + [(tuples[0][0], True)]
    return sp.Piecewise(*tuples)
    

class ScalarCoef(mfem.PyCoefficient):
    def __init__(self, coefs, dim):
        super().__init__()
        self._coefs = coefs
        self._dim = dim
        self._funcs = sp.lambdify(_generateArgs(dim), coefs)

    def Eval(self, T, ip):
        self._attr = T.Attribute
        return super().Eval(T, ip)

    def EvalValue(self, x):
        return float(self._funcs(self._attr, *x))


class VectorCoef(mfem.VectorPyCoefficient):
    def __init__(self, coefs, dim):
        super().__init__(dim)
        self._coefs = coefs
        self._dim = dim
        self._funcs = sp.lambdify(_generateArgs(dim), coefs)

    def __getitem__(self, index):
        if isinstance(self._coefs, sp.Piecewise):
            coefs = sp.Piecewise(*[(key[index], value) for key, value in self._coefs.args])
        else:
            coefs = self._coefs[index]
        return ScalarCoef(coefs, self._dim)

    @staticmethod
    def fromScalars(coefs):
        return VectorCoef(sp.Matrix([c._coefs for c in coefs]), coefs[0]._dim)

    def Eval(self, K, T, ip):
        self._attr = T.Attribute
        return super().Eval(K, T, ip)

    def EvalValue(self, x):
        return self._funcs(self._attr, *x).astype(float)


class MatrixCoef(mfem.MatrixPyCoefficient):
    def __init__(self, coefs, dim):
        super().__init__(dim)
        self._coefs = coefs
        self._dim = dim
        self._funcs = sp.lambdify(_generateArgs(dim), coefs)

    def __getitem__(self, index):
        if isinstance(self._coefs, sp.Piecewise):
            coefs = sp.Piecewise(*[(key[index], value) for key, value in self._coefs.args])
        else:
            coefs = self._coefs[index]
        if isinstance(index, int):
            return VectorCoef(coefs, self._dim)
        else:
            return ScalarCoef(coefs, self._dim)

    @staticmethod
    def fromScalars(coefs):
        return MatrixCoef(sp.Matrix([[c._coefs for c in cs] for cs in coefs]), coefs[0][0]._dim)

    def Eval(self, K, T, ip):
        self._attr = T.Attribute
        return super().Eval(K, T, ip)

    def EvalValue(self, x):
        res = self._funcs(self._attr, *x)
        return res.astype(float)


def _generateArgs(dim):
    d,x,y,z = sp.symbols("domain,x,y,z")
    if dim == 1:
        return d,x
    elif dim == 2:
        return d,x,y
    elif dim == 3:
        return d,x,y,z

