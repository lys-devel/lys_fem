import numpy as np
from lys_fem import mf
if mf.parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem


def generateCoefficient(coefs, geom, type=None):
    if len(geom.getPhysicalGroups(3)) != 0:
        dim = 3
    elif len(geom.getPhysicalGroups(2)) != 0:
        dim = 2
    else:
        dim = 1
    shape = np.array(list(coefs.values()))[0].shape
    if type is not None:
        pass
    elif len(shape) == 0:
        return ScalarCoef(coefs, dim, geom)
    elif len(shape) == 1:
        return VectorCoef(coefs, dim, geom, shape[0])
    elif len(shape) == 2:
        return MatrixCoef(coefs, dim, geom, shape[0])


class ScalarCoef(mfem.PyCoefficient):
    def __init__(self, coefs, dim, geom):
        super().__init__()
        self._geom = geom
        self._dim = dim
        self._coefs = coefs
        self._funcs = _generateFuncs(coefs, dim)

    def EvalValue(self, x):
        return _eval(x, self._funcs, self._geom, self._dim, 0)


class VectorCoef(mfem.VectorPyCoefficient):
    def __init__(self, coefs, dim, geom, size):
        super().__init__(size)
        self._geom = geom
        self._dim = dim
        self._coefs = coefs
        self._funcs = _generateFuncs(coefs, dim)
        self._default = np.array([0] * size, dtype=float)

    def EvalValue(self, x):
        return _eval(x, self._funcs, self._geom, self._dim, self._default)


class MatrixCoef(mfem.MatrixPyCoefficient):
    def __init__(self, coefs, dim, geom, size):
        super().__init__(size)
        self._geom = geom
        self._dim = dim
        self._coefs = coefs
        self._funcs = _generateFuncs(coefs, dim)
        self._default = np.zeros((size, size), dtype=float)

    def EvalValue(self, x):
        return _eval(x, self._funcs, self._geom, self._dim, self._default)


def _generateFuncs(coefs, dim):
    return {domain: np.vectorize(lambda x: _generateFunc(x, dim))(func) for domain, func in coefs.items()}


def _generateFunc(funcStr, dim):
    vars = "x"
    if dim > 1:
        vars += ",y"
    if dim > 2:
        vars += ",z"
    return eval("lambda " + vars + ": " + funcStr)


def _eval(x, funcs, geom, dim, default):
    d = _findDomain(x, geom, dim)
    if d not in funcs:
        return default
    else:
        return np.vectorize(lambda f: f(*x), otypes=[float])(funcs[d])


def _findDomain(x, geom, dim):
    while len(x) < 3:
        x = np.append(x, 0)
    for d, tag in geom.getEntities(dim):
        if geom.isInside(d, tag, x):
            return tag
