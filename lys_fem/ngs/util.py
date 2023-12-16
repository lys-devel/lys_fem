import sympy as sp
import numpy as np
from ngsolve import x,y,z, CoefficientFunction

def prod(args):
    res = args[0]
    for arg in args[1:]:
        res = res * arg
    return res

def generateDomainCoefficient(mesh, conditions):
    coefs = {}
    for c in conditions:
        for d in c.domains:
            coefs[d] = c.values
    return generateCoefficient(coefs, mesh)


def generateBoundaryCoefficient(mesh, conditions):
    coefs = {}
    for c in conditions:
        for d in c.boundaries:
            coefs[d] = c.values
    return generateCoefficient(coefs, mesh, geom="Boundary")


def generateCoefficient(coef, mesh=None, geom="Domain"):
    if isinstance(coef, dict):
        coefs = {str(key): generateCoefficient(value) for key, value in coef.items()}
        if geom=="Domain":
            return mesh.MaterialCF(coefs)
        else:
            return mesh.BoundaryCF(coefs)
    elif isinstance(coef, (list, tuple)):
        return CoefficientFunction(tuple([generateCoefficient(c) for c in coef]), dims=np.shape(coef))
    elif isinstance(coef, (int, float, sp.Integer, sp.Float)):
        return CoefficientFunction(coef)
    elif isinstance(coef, CoefficientFunction):
        return coef
    else:
        return sp.lambdify(sp.symbols("x,y,z"), coef)(x,y,z)