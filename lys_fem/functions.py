def addMaterialParameter(group, param):
    from .fem.material import materialParameters
    if group not in materialParameters:
        materialParameters[group] = []
    materialParameters[group].append(param)


def addModel(group, model):
    from .fem.model import models
    if group not in models:
        models[group] = []
    models[group].append(model)


def addGeometry(group, geom):
    from .fem.geometry import geometryCommands
    if group not in geometryCommands:
        geometryCommands[group] = []
    geometryCommands[group].append(geom)


def addSolver(group, solver):
    from .fem.solver import solvers
    if group not in solvers:
        solvers[group] = []
    solvers[group].append(solver)
