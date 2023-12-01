from .coef import generateCoefficient


def generateDomainCoefficient(mesh, conditions, default):
    coefs = {"default": default}
    for c in conditions:
        for d in mesh.attributes:
            if c.domains.check(d):
                coefs[d] = c.values
    return generateCoefficient(coefs, mesh.SpaceDimension())


def generateSurfaceCoefficient(mesh, conditions, default):
    coefs = {"default": default}
    for b in conditions:
        for d in mesh.bdr_attributes:
            if b.boundaries.check(d):
                coefs[d] = b.values
    return generateCoefficient(coefs, mesh.SpaceDimension())