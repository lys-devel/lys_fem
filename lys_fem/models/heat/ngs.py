from ngsolve import grad, dx, ds

from lys_fem.ngs import NGSModel, dti, util
from ..common import NeumannBoundary


class NGSHeatConductionModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True)
        self._model = model
        self._mat = mat

    def bilinearform(self, tnt, sols):
        Cv, k = self._mat["C_v"], self._mat["k"]
        wf = 0
        for eq in self._model.equations:
            u,v = tnt[eq.variableName]
            gu, gv = grad(u), grad(v)
            wf += (Cv*u*v * dti + k * (gu*gv)) * dx(definedon=util.generateGeometry(eq.geometries))

        return wf
    
    def linearform(self, tnt, sols):
        Cv = self._mat["C_v"]
        wf = 0
        for eq in self._model.equations:
            u,v = tnt[eq.variableName]
            u0 = sols[eq.variableName]
            wf += Cv * u0 * v * dti * dx(definedon=util.generateGeometry(eq.geometries))
        
            c = self._model.boundaryConditions.coef(NeumannBoundary)
            if c is not None:
                f = util.generateCoefficient(c, self.mesh)
                wf += f * v * ds
        return wf
