

from .model import models as modelList
from .coef import generateCoefficient


def generateModel(fem, geom, mesh, mat):
    _modelDict = {m.name: m for m in modelList}
    models = []
    for m in fem.models:
        model = _modelDict[m.name](m, mesh, mat)
        attrs = [tag for dim, tag in geom.getEntities(fem.dimension)]
        coefs = {}
        for init in m.initialConditions:
            for d in attrs:
                if init.domains.check(d):
                    coefs[d] = init.values
        c = generateCoefficient(coefs, geom)
        model.setInitialValue(c)
        models.append(model)
    return models
