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
        self.x0 = util.initialValue(self._space, util.generateDomainCoefficient(self._space, model.initialConditions))
        self.M = util.bilinearForm(self._space, ess_tdof, domainInteg=mfem.MassIntegrator(self._mat["Heat Conduction"]["C_v"]))
        self.K = util.bilinearForm(self._space, ess_tdof, domainInteg=mfem.DiffusionIntegrator(self._mat["Heat Conduction"]["k"]))
        self.b = util.linearForm(self._space, ess_tdof, self.K, self.x0, boundaryInteg=self._boundary_linear(model))

    def _boundary_linear(self, model):
        res = []
        # neumann boundary
        neumann = [b for b in model.boundaryConditions if isinstance(b, NeumannBoundary)]
        if len(neumann) != 0:
            c = util.generateSurfaceCoefficient(self._space, neumann)
            res.append(mfem.VectorBoundaryLFIntegrator(c))
        return res
    
    @property
    def solution(self):
        gf = mfem.GridFunction(self._space)
        gf.SetFromTrueDofs(self.x0)
        return gf
        