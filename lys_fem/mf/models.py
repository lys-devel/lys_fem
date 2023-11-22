from . import mfem
from .coef import generateCoefficient
from ..fem import DirichletBoundary, NeumannBoundary, Source

modelList = {}


def addMFEMModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, mesh, mat):
    return [modelList[m.name](m, mesh, mat) for m in fem.models]


class MFEMModel:
    def __init__(self, model):
        self._model = model

    def generateDomainFunction(self, space, conditions):
        c = self.generateDomainCoefficient(space, conditions)
        gf = mfem.GridFunction(space)
        gf.ProjectCoefficient(c)
        return gf

    def generateDomainCoefficient(self, space, conditions):
        coefs = {}
        for c in conditions:
            for d in space.GetMesh().attributes:
                if c.domains.check(d):
                    coefs[d] = c.values
        return generateCoefficient(coefs, space.GetMesh().Dimension())
    
    def generateSurfaceCoefficient(self, space, conditions):
        bdr_stress = {}
        for b in conditions:
            for d in space.GetMesh().bdr_attributes:
                if b.boundaries.check(d):
                    bdr_stress[d] = b.values
        return generateCoefficient(bdr_stress, space.GetMesh().Dimension())

    def essential_tdof_list(self, space):
        res = []
        for axis, b in self._dirichletCondition().items():
            if len(b) == 0:
                continue
            ess_bdr = mfem.intArray(space.GetMesh().bdr_attributes.Max())
            ess_bdr.Assign(0)
            for i in b:
                ess_bdr[i - 1] = 1
            ess_tdof_list = mfem.intArray()
            space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list, axis)
            res.extend([i for i in ess_tdof_list])
        return mfem.intArray(res)

    def _dirichletCondition(self):
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, x):
        pass

    @property
    def M(self):
        return self._M

    @property
    def grad_Mx(self):
        return self.M

    @property
    def K(self):
        return self._K

    @property
    def grad_Kx(self):
        return self.K

    @property
    def b(self):
        return self._B

    @property
    def grad_b(self):
        return None

    @property
    def x0(self):
        return self._X0

    @property
    def xt0(self):
        return self._Xt0


class MFEMNonlinearModel(MFEMModel):
    pass
