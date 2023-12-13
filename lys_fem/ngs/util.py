import sympy as sp
from ngsolve import x,y,z, CoefficientFunction

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
    return generateCoefficient(coefs, mesh, type="Boundary")


def generateCoefficient(coef, mesh=None, type="Domain"):
    if isinstance(coef, dict):
        coefs = {str(key): generateCoefficient(value) for key, value in coef.items()}
        if type=="Domain":
            return mesh.MaterialCF(coefs)
        else:
            return mesh.BoundaryCF(coefs)
    if isinstance(coef, (int, float)):
        return CoefficientFunction(coef)
    else:
        return sp.lambdify(sp.symbols("x,y,z"), coef)(x,y,z)