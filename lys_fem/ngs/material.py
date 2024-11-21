import numpy as np
import sympy as sp
import ngsolve

from lys_fem.fem import FEMCoefficient
from . import util


def generateMaterial(fem, mesh):
    sols = {"x": util.NGSFunction(ngsolve.x, name="x"), "y": util.NGSFunction(ngsolve.y, name="y"), "z": util.NGSFunction(ngsolve.z, name="z"), "t": util.t}
    sols.update({str(key): util.NGSFunction(value) for key, value in fem.parameters.getSolved().items()})
    sols.update({key: util.SolutionFieldFunction(coef.get(), tdep=coef.index is None) for key, coef in fem.solutionFields.items()})

    mats = fem.materials.materialDict(mesh.dim)
    mats.update(fem.geometries.geometryParameters())
    return NGSParams(fem, mats, mesh, sols)


class NGSParams(dict):
    def __init__(self, fem, dic, mesh, sols):
        super().__init__(sols)
        self._fem = fem
        self._mesh = mesh
        self._const = NGSConstants()
        for key, value in dic.items():
            self[key] = _generateCoefficient(value, mesh, name=key, dic=self)

    def __getitem__(self, expr):
        d = dict(self._const)
        d.update(self)
        return _generateCoefficient(expr, self._mesh, dic=d)

    def updateSolutionFields(self, step):
        for key, f in self.items():
            if isinstance(f, util.SolutionFieldFunction) and f.isTimeDependent:
                self._fem.solutionFields[key].update(step)

    @property
    def const(self):
        return self._const


class NGSConstants(dict):
    def __init__(self):
        self["c"] = util.NGSFunction(2.99792458e8, name="c")
        self["e"] = util.NGSFunction(-1.602176634e-19, name="e")
        self["pi"] = util.NGSFunction(np.pi, name="pi")
        self["k_B"] = util.NGSFunction(1.3806488e-23, name="k_B")
        self["g_e"] = util.NGSFunction(1.760859770e11, name="g_e")
        self["mu_0"] = util.NGSFunction(1.25663706e-6, name="mu_0")
        self["mu_B"] = util.NGSFunction(9.2740100657e-24 , name="mu_B")
        self["eps_0"] = util.NGSFunction(8.8541878128e-12, name="eps_0")
        self["dti"] = util.Parameter("dti", -1)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            super().__getattr__(key)


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
    elif isinstance(coef, str):
        return util.eval(coef, dic)
    else:
        print("error")
