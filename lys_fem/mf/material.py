from .coef import generateCoefficient


def generateMaterial(fem, geom):
    mats = fem.materials
    # find all parameter names and their groups
    groups = {}
    for m in mats:
        for p in m:
            if p.name in groups:
                groups[p.name] |= set(p.getParameters().keys())
            else:
                groups[p.name] = set(p.getParameters().keys())

    # create coefficient for respective parameter
    res = {}
    for group, params in groups.items():
        tmp = {pname: __generateCoefForParameter(pname, mats, group, fem, geom) for pname in params}
        res[group] = tmp
    return res


def __generateCoefForParameter(pname, mats, group, fem, geom):
    attrs = [tag for dim, tag in geom.getEntities(fem.dimension)]
    coefs = {}
    for m in mats:
        p = m[group]
        if p is None:
            continue
        for d in attrs:
            if m.domains.check(d):
                coefs[d] = p.getParameters()[pname]
    return generateCoefficient(coefs, fem.dimension)
