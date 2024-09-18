import numpy as np
import sympy as sp
import ngsolve
from . import util


def generateMaterial(fem, mesh):
    mats = fem.materials.materialDict(mesh.dim)
    mats.update(fem.geometries.geometryParameters())
    #mats.update(fem.parameters)
    return NGSParams(mats, mesh, fem.geometries.scale)


class NGSParams(dict):
    def __init__(self, dic, mesh, scale):
        super().__init__()
        self._mesh = mesh
        self._coefs = {"x": ngsolve.x*scale, "y": ngsolve.y*scale, "z": ngsolve.z*scale}
        self.__parse(dic, mesh)

    def __parse(self, dic, mesh):
        # Replace all items by dic
        while True:
            n = len(self._coefs)
            for key, value in dic.items():
                value = self.__subs(value.value, self._coefs)
                self._coefs[key] = util.generateCoefficient(value, mesh)
            if len(self._coefs) == n:
                break
        # Convert all coefs into NGSFunction
        for key, value in self._coefs.items():
            self[key] = util.NGSFunction(value, name=key)

    def eval(self, expr):
        return util.generateCoefficient(self.__subs(expr.value, self._coefs), self._mesh, geom=expr.geometryType)
    
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
