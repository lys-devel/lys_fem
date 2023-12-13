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
    return generateCoefficient(coefs, mesh)


def generateCoefficient(coef, mesh=None):
    if isinstance(coef, dict):
        coefs = {str(key): generateCoefficient(value) for key, value in coef.items()}
        print(coefs)
        return mesh.MaterialCF(coefs)
    if isinstance(coef, (int, float)):
        return coef
    else:
        return sp.lambdify(sp.symbols("x,y,z"), coef)(x,y,z)