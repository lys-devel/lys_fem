import sympy as sp
from .coef import generateCoefficient


def generateDomainCoefficient(mesh, conditions, default):
    coefs = {}
    for c in conditions:
        for d in mesh.attributes:
            if c.domains.check(d):
                coefs[d] = c.values

    # convert dict to sympy piesewise expression
    d = sp.Symbol("domain")
    tuples = [(value, sp.Eq(d, sp.Integer(key))) for key, value in coefs.items()] + [(default, True)]
    coefs = sp.Piecewise(*tuples)

    return generateCoefficient(coefs, mesh.Dimension())


def generateSurfaceCoefficient(mesh, conditions, default):
    coefs = {}
    for b in conditions:
        for d in mesh.bdr_attributes:
            if b.boundaries.check(d):
                coefs[d] = b.values

    # convert dict to sympy piesewise expression
    d = sp.Symbol("domain")
    tuples = [(value, sp.Eq(d, sp.Integer(key))) for key, value in coefs.items()] + [(default, True)]
    coefs = sp.Piecewise(*tuples)

    return generateCoefficient(coefs, mesh.Dimension())