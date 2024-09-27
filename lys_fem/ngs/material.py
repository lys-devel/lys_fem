import numpy as np
import sympy as sp
import ngsolve

from lys_fem.fem import FEMCoefficient
from . import util


def generateMaterial(fem, mesh):
    scale = fem.geometries.scale
    sols = {"x": ngsolve.x*scale, "y": ngsolve.y*scale, "z": ngsolve.z*scale}
    sols.update({str(key): _generateCoefficient(value) for key, value in fem.parameters.getSolved().items()})
    sols.update({key: coef.solution.obj.coef(coef.expression, coef.index) for key, coef in fem.solutionFields.items()})

    mats = fem.materials.materialDict(mesh.dim)
    mats.update(fem.geometries.geometryParameters())
    return NGSParams(mats, mesh, sols)


class NGSParams(dict):
    def __init__(self, dic, mesh, sols):
        super().__init__()
        self._mesh = mesh
        self.__parse(dic, mesh, sols)

    def __parse(self, dic, mesh, coefs):
        # Replace all items by dic
        while True:
            n = len(coefs)
            for key, value in dic.items():
                value = self.__subs(value.value, coefs)
                coefs[key] = _generateCoefficient(value, mesh)
            if len(coefs) == n:
                break
        # Convert all coefs into NGSFunction
        for key, value in coefs.items():
            self[key] = util.NGSFunction(value, name=key)

    def __getitem__(self, expr):
        if isinstance(expr, (list, tuple, np.ndarray)):
            return util.NGSFunctionVector([self[ex] for ex in expr])
        if isinstance(expr, str):
            expr = sp.parsing.sympy_parser.parse_expr(expr)
        if isinstance(expr, sp.Basic):
            return sp.lambdify(sp.symbols(list(self.keys())), expr)(**self)
        else:
            return expr

    def eval(self, expr):
        coefs = {key: item.eval() for key, item in self.items()}
        return _generateCoefficient(self.__subs(expr.value, coefs), self._mesh, geom=expr.geometryType)
    
    def __subs(self, value, dic):
        if isinstance(value, dict):
            return {key: self.__subs(val, dic) for key, val in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [self.__subs(v, dic) for v in value]
        elif isinstance(value, (int, float, sp.Integer, sp.Float)):
            return value
        elif isinstance(value, sp.Basic):
            return sp.lambdify(sp.symbols(list(dic.keys())), value, modules=[ngsolve])(**dic)
        else:
            return value


def _generateCoefficient(coef, mesh=None, geom="Domain"):
    if isinstance(coef, dict):
        return _generateCoefficient(FEMCoefficient(coef, geom), mesh)
    if isinstance(coef, FEMCoefficient):
        geom = coef.geometryType.lower()
        if geom == "const":
            return _generateCoefficient(coef.value)
        coefs = {geom+str(key): _generateCoefficient(value) for key, value in coef.value.items() if key != "default"}
        if coef.default is not None:
            default = _generateCoefficient(coef.default)
        else:
            default = None
        if geom=="domain":
            return mesh.MaterialCF(coefs, default=default)
        else:
            return mesh.BoundaryCF(coefs, default=default)
    elif isinstance(coef, (list, tuple, np.ndarray)):
        return ngsolve.CoefficientFunction(tuple([_generateCoefficient(c) for c in coef]), dims=np.shape(coef))
    elif isinstance(coef, (int, float, sp.Integer, sp.Float)):
        return ngsolve.CoefficientFunction(coef)
    elif isinstance(coef, ngsolve.CoefficientFunction):
        return coef
