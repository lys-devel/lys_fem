from lys_fem.fem import NeumannBoundary
from lys_fem.mf import mfem, MFEMLinearModel, util


class MFEMHeatConductionModel(MFEMLinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model)
        self._mat = mat
        self._fec = mfem.H1_FECollection(1, mesh.Dimension())
        self._space = mfem.FiniteElementSpace(mesh, self._fec, 1, mfem.Ordering.byVDIM)
        self._initialize(model)

    def _initialize(self, model):
        ess_tdof = self.essential_tdof_list(self._space)
        c = self.generateDomainCoefficient(self._space, model.initialConditions)
        self._X0 = util.initialValue(self._space, c)
        self._M = util.bilinearForm(self._space, ess_tdof, domainInteg=mfem.MassIntegrator(self._mat["Heat Conduction"]["C_v"]))
        self._K = util.bilinearForm(self._space, ess_tdof, domainInteg=mfem.DiffusionIntegrator(self._mat["Heat Conduction"]["k"]))
        self._B = util.linearForm(self._space, ess_tdof, self._K, self.x0, boundaryInteg=self._boundary_linear(model))

    def _boundary_linear(self, model):
        res = []
        # neumann boundary
        neumann = [b for b in model.boundaryConditions if isinstance(b, NeumannBoundary)]
        if len(neumann) != 0:
            c = self.generateSurfaceCoefficient(self._space, neumann)
            res.append(mfem.VectorBoundaryLFIntegrator(c))
        return res

    def RecoverFEMSolution(self, X):
        self._X0 = X
        return self.solution
    
    @property
    def solution(self):
        self.gf = mfem.GridFunction(self._space)
        self.gf.SetFromTrueDofs(self._X0)
        return self.gf
        