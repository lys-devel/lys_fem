from ..fem import DirichletBoundary

modelList = {}


def addMFEMModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, mesh, mat):
    return [modelList[m.name](m, mesh, mat) for m in fem.models]


class MFEMModel:
    def __init__(self, model):
        self._model = model

    @property
    def dirichletCondition(self):
        conditions = [b for b in self._model.boundaryConditions if isinstance(b, DirichletBoundary)]
        bdr_dir = {i: [] for i in range(self._model.variableDimension())}
        for b in conditions:
            for axis, check in enumerate(b.components):
                if check:
                    bdr_dir[axis].extend(b.boundaries.getSelection())
        return bdr_dir

    @property
    def variableName(self):
        return self._model.variableName

    @property
    def preconditioner(self):
        return None
    
    @property
    def timeUnit(self):
        return 1

class MFEMLinearModel(MFEMModel):
    pass


class MFEMNonlinearModel(MFEMModel):
    pass
