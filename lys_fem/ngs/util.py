import sympy as sp
import numpy as np

import ngsolve
from ngsolve import x,y,z, CoefficientFunction
from ngsolve.fem import Einsum

from lys_fem.fem import FEMCoefficient
from ..models.common import DirichletBoundary

def prod(args):
    res = args[0]
    for arg in args[1:]:
        res = res * arg
    return res


def cross(u,v):
    return [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]


def dot(u,v):
    res = u[0] * v[0]
    if len(u) > 1:
        res += u[1]*v[1]
    if len(u) > 2:
        res += u[2]*v[2]
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


def generateCoefficient(coef, mesh=None, geom="domain", xscale=1, **kwargs):
    if isinstance(coef, FEMCoefficient):
        geom = coef.geometryType.lower()
        coefs = {geom+str(key): generateCoefficient(value, xscale=coef.xscale) for key, value in coef.items()}
        if geom=="domain":
            return mesh.MaterialCF(coefs, **kwargs)/coef.scale
        else:
            return mesh.BoundaryCF(coefs, **kwargs)/coef.scale
    elif isinstance(coef, (list, tuple, np.ndarray)):
        return CoefficientFunction(tuple([generateCoefficient(c, xscale=xscale) for c in coef]), dims=np.shape(coef))
    elif isinstance(coef, (int, float, sp.Integer, sp.Float)):
        return CoefficientFunction(coef)
    elif isinstance(coef, CoefficientFunction):
        return coef
    else:
        return sp.lambdify(sp.symbols("x,y,z"), coef, modules=[{"abs": _absolute}, ngsolve])(x*xscale,y*xscale,z*xscale)


def _absolute(x):
    return x.Norm()