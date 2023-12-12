from ngsolve import H1, grad, dx
from lys_fem.ngs import NGSModel, util


class NGSHeatConductionModel(NGSModel):
    def __init__(self, model, mesh, mat=None):
        super().__init__(model)
        self._mesh = mesh
        self._model = model
        self._fes = H1(mesh, order=1, dirichlet=[1,3])
        self._init = util.generateDomainCoefficient(mesh, self._model.initialConditions)

    @property
    def spaces(self):
        return [self._fes]

    @property
    def weakform(self):
        u,v =self._fes.TnT()
        gu, gv = grad(u), grad(v)
        #Cv, k = sp.symbols("Cv,k")

        wf = (gu*gv) * dx
        return wf

    @property
    def initialValues(self):
        return [self._init]