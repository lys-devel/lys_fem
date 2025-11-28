import numpy as np
import sympy as sp
import ngsolve

from lys_fem.fem import FEMCoefficient
from . import util


def generateMaterial(fem, solutions=True):
    sols = dict(NGSConstants())
    sols.update(fem.parameters.eval())
    if solutions:
        sols.update(fem.solutionFields.eval())
    sols.update(fem.randomFields)
    res = NGSParams(fem, sols)
    return res


class NGSParams(dict):
    def __init__(self, fem, sols):
        super().__init__(sols)
        self._fem = fem
        for key, value in fem.materials.materialDict(fem.dimension).items():
            self[key] = _generateCoefficient(value, name=key, dic=self, J="R" if key != "R" else None)
        for key, value in fem.geometries.geometryParameters().items():
            self[key] = _generateCoefficient(value, name=key, dic=self)

    def __getitem__(self, expr):
        return _generateCoefficient(expr, dic=dict(self), name=str(expr))
    
    @property
    def jacobi(self):
        res = {}
        if "J" in self:
            res["J"] = self["J"].T
        if "R" in self:
            res["R"] = self["R"]
        return res

    def updateSolutionFields(self, step):
        for key, f in self.items():
            if isinstance(f, util.SolutionFieldFunction) and f.isTimeDependent:
                self._fem.solutionFields[key].update(step)
            if isinstance(f, util.RandomFieldFunction) and f.isTimeDependent:
                f.update()

    @property
    def const(self):
        return NGSConstants()


class NGSConstants(dict):
    def __init__(self):
        self["x"] = util.NGSFunction(ngsolve.x, name="x")
        self["y"] = util.NGSFunction(ngsolve.y, name="y")
        self["z"] = util.NGSFunction(ngsolve.z, name="z")
        self["t"] = util.t
        self["dti"] = util.dti

        self["c"] = util.NGSFunction(2.99792458e8, name="c")
        self["e"] = util.NGSFunction(-1.602176634e-19, name="e")
        self["pi"] = util.NGSFunction(np.pi, name="pi")
        self["k_B"] = util.NGSFunction(1.3806488e-23, name="k_B")
        self["g_e"] = util.NGSFunction(1.760859770e11, name="g_e")
        self["mu_0"] = util.NGSFunction(1.25663706e-6, name="mu_0")
        self["mu_B"] = util.NGSFunction(9.2740100657e-24 , name="mu_B")
        self["eps_0"] = util.NGSFunction(8.8541878128e-12, name="eps_0")

        self["Ve"] = util.VolumeField(name = "Ve")

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            super().__getattr__(key)


def _generateCoefficient(coef, name="Undefined", dic={}, J=None):
    if isinstance(coef, FEMCoefficient):
        geom = coef.geometryType.lower()
        if geom == "const":
            return _generateCoefficient(coef.value, name=name, dic=dic,J=J)
        coefs = {geom+str(key): _generateCoefficient(value, dic=dic) for key, value in coef.value.items() if key != "default"}
        if coef.default is not None:
            default = _generateCoefficient(coef.default, dic=dic)
        else:
            default = None
        return util.DomainWiseFunction(coefs, geomType=geom, default=default, name=name, J=J)
    elif isinstance(coef, (list, tuple, np.ndarray)):
        return util.NGSFunction([_generateCoefficient(c, dic=dic) for c in coef], name=name)
    elif isinstance(coef, (int, float, sp.Integer, sp.Float, ngsolve.CoefficientFunction)):
        return util.NGSFunction(coef, name=name)
    elif isinstance(coef, str):
        return util.eval(coef, dic)
    else:
        print("error")
