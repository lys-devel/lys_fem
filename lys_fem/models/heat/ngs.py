from ngsolve import grad, dx, ds

from lys_fem.fem import NeumannBoundary
from lys_fem.ngs import NGSModel, dti, util


class NGSHeatConductionModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh)
        self._model = model
        self._mat = mat

    @property
    def bilinearform(self):
        Cv, k = self._mat["C_v"], self._mat["k"]
        u,v =self.TnT()
        gu, gv = grad(u), grad(v)

        wf = Cv*u*v * dti * dx + k * (gu*gv) * dx

        return wf
    
    @property
    def linearform(self):
        u,v =self.TnT()
        u0 = self.sol[0]
        Cv = self._mat["C_v"]
        wf = Cv * u0 * v * dti * dx
        
        if self._model.boundaryConditions.have(NeumannBoundary):
            f = util.generateBoundaryCoefficient(self.mesh, self._model.boundaryConditions.get(NeumannBoundary))
            wf += f * v * ds
        return wf
