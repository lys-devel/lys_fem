import sympy as sp
import numpy as np

import ngsolve
from ngsolve import x,y,z, CoefficientFunction
from ngsolve.fem import Einsum
from ..models.common import DirichletBoundary

def prod(args):
    res = args[0]
    for arg in args[1:]:
        res = res * arg
    return res


def CrossProduct(u, v):
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    eijk = generateCoefficient(eijk)
    return Einsum('ijk,i,j->k', eijk, u, v)


def generateGeometry(region):
    return  "|".join([region.geometryType.lower() + str(r) for r in region])


def generateDirichletCondition(model):
    conditions = model.boundaryConditions.get(DirichletBoundary)
    bdr_dir = {i: [] for i in range(model.variableDimension())}
    for b in conditions:
        for axis, check in enumerate(b.values):
            if check:
                bdr_dir[axis].extend(b.geometries.getSelection())
    return list(bdr_dir.values())


def generateGeometryCoefficient(mesh, conditions):
    coefs = {}
    for c in conditions:
        for d in c.geometries:
            coefs[d] = c.values
        type = c.geometries.geometryType.lower()
    return generateCoefficient(coefs, mesh, type)


def generateCoefficient(coef, mesh=None, geom="domain", **kwargs):
    if isinstance(coef, dict):
        coefs = {geom+str(key): generateCoefficient(value) for key, value in coef.items()}
        if geom=="domain":
            return mesh.MaterialCF(coefs, **kwargs)
        else:
            return mesh.BoundaryCF(coefs, **kwargs)
    elif isinstance(coef, (list, tuple, np.ndarray)):
        return CoefficientFunction(tuple([generateCoefficient(c) for c in coef]), dims=np.shape(coef))
    elif isinstance(coef, (int, float, sp.Integer, sp.Float)):
        return CoefficientFunction(coef)
    elif isinstance(coef, CoefficientFunction):
        return coef
    else:
        return sp.lambdify(sp.symbols("x,y,z"), coef, modules=[{"abs": _absolute}, ngsolve])(x,y,z)


def _absolute(x):
    return x.Norm()