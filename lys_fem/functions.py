def addMaterialParameter(group, param):
    from .fem.material import materialParameters
    if group not in materialParameters:
        materialParameters[group] = []
    materialParameters[group].append(param)


def addModel(name, model):
    from .fem.model import models
    models[name] = model


def addGeometry(group, geom):
    from .fem.geometry import geometryCommands
    if group not in geometryCommands:
        geometryCommands[group] = []
    geometryCommands[group].append(geom)
