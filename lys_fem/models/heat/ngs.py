from ngsolve import grad, dx, ds

from lys_fem.ngs import NGSModel, dti, util
from ..common import NeumannBoundary


class NGSHeatConductionModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh)
        self._model = model
        self._mat = mat
        self.__generateVariables(mesh, model)

    def __generateVariables(self, mesh, model):
        dirichlet = util.generateDirichletCondition(model)
        init = util.generateGeometryCoefficient(mesh, model.initialConditions)
        for eq in model.equations:
            self.addVariable(eq.variableName, eq.variableDimension, dirichlet, init, eq.geometries)

    @property
    def bilinearform(self):
        Cv, k = self._mat["C_v"], self._mat["k"]
        wf = 0
        for sp, eq in zip(self.spaces, self._model.equations):
            u,v =sp.TnT()
            gu, gv = grad(u), grad(v)
            wf += (Cv*u*v * dti + k * (gu*gv)) * dx(definedon=util.generateGeometry(eq.geometries))

        return wf
    
    @property
    def linearform(self):
        Cv = self._mat["C_v"]
        wf = 0
        for sp, u0, eq in zip(self.spaces, self.sol, self._model.equations):
            u,v =sp.TnT()
            wf += Cv * u0 * v * dti * dx(definedon=util.generateGeometry(eq.geometries))
        
        if self._model.boundaryConditions.have(NeumannBoundary):
            f = util.generateGeometryCoefficient(self.mesh, self._model.boundaryConditions.get(NeumannBoundary))
            wf += f * v * ds
        return wf
