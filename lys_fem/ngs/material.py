import numpy as np
import sympy as sp
import ngsolve

from lys_fem.fem import FEMCoefficient
from . import util


def generateMaterial(fem, mesh):
    scale = fem.geometries.scale
    sols = {"x": util.NGSFunction(ngsolve.x*scale, name="x"), "y": util.NGSFunction(ngsolve.y*scale, name="y"), "z": util.NGSFunction(ngsolve.z*scale, name="z"), "t": util.t}
    sols.update({str(key): util.NGSFunction(value) for key, value in fem.parameters.getSolved().items()})
    sols.update({key: util.NGSFunction(coef.solution.obj.coef(coef.expression, coef.index)) for key, coef in fem.solutionFields.items()})

    mats = fem.materials.materialDict(mesh.dim)
    mats.update(fem.geometries.geometryParameters())
    return NGSParams(mats, mesh, sols)


class NGSParams(dict):
    def __init__(self, dic, mesh, sols):
        super().__init__(sols)
        self._mesh = mesh
        for key, value in dic.items():
            self[key] = _generateCoefficient(value, mesh, name=key, dic=self)

    def __getitem__(self, expr):
        if isinstance(expr, str):
            expr = sp.parsing.sympy_parser.parse_expr(expr)
        return _generateCoefficient(expr, self._mesh, dic=self)
    

def _generateCoefficient(coef, mesh=None, name="Undefined", dic={}):
    if isinstance(coef, FEMCoefficient):
        geom = coef.geometryType.lower()
        if geom == "const":
            return _generateCoefficient(coef.value, name=name, dic=dic)
        coefs = {geom+str(key): _generateCoefficient(value, dic=dic) for key, value in coef.value.items() if key != "default"}
        if coef.default is not None:
            default = _generateCoefficient(coef.default, dic=dic)
        else:
            default = None
        return util.NGSFunction(coefs, mesh, geomType=geom, default=default, name=name)
    elif isinstance(coef, (list, tuple, np.ndarray)):
        return util.NGSFunction([_generateCoefficient(c, dic=dic) for c in coef], name=name)
    elif isinstance(coef, (int, float, sp.Integer, sp.Float, ngsolve.CoefficientFunction)):
        return util.NGSFunction(coef, name=name)
    elif isinstance(coef, sp.Basic):
        return sp.lambdify(sp.symbols(list(dic.keys())), coef, modules=[util])(**dic)
    else:
        print("error")
