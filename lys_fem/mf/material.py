from .coef import generateCoefficient


def generateMaterial(fem, mesh):
    mats = fem.materials
    # find all parameter names and their groups
    groups = {}
    for m in mats:
        for p in m:
            if p.name in groups:
                groups[p.name] |= set(p.getParameters(mesh.SpaceDimension()).keys())
            else:
                groups[p.name] = set(p.getParameters(mesh.SpaceDimension()).keys())

    # create coefficient for respective parameter
    res = {}
    for group, params in groups.items():
        res[group] = {pname: __generateCoefForParameter(pname, mats, group, mesh) for pname in params}
    return res


def __generateCoefForParameter(pname, mats, group, mesh):
    coefs = {"default": mats.defaultParameter(group, mesh.SpaceDimension())[pname]}
    for m in mats:
        p = m[group]
        if p is None:
            continue
        for d in mesh.attributes:
            if m.domains.check(d):
                coefs[d] = p.getParameters(mesh.SpaceDimension())[pname]
    return generateCoefficient(coefs, mesh.SpaceDimension())
