from .mfem import generateCoefficient



def generateDomainCoefficient(mesh, conditions, default):
    coefs = {"default": default}
    for c in conditions:
        for d in c.geometries:
            coefs[d] = c.values
    return generateCoefficient(coefs)


def generateSurfaceCoefficient(mesh, conditions, default):
    coefs = {"default": default}
    for c in conditions:
        for d in c.geometries:
            coefs[d] = c.values
    return generateCoefficient(coefs)

