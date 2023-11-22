from lys_fem.fem import NeumannBoundary
from lys_fem.mf import mfem, MFEMLinearModel


class MFEMHeatConductionModel(MFEMLinearModel):
    def __init__(self, model, mesh, mat):
        self._mat = mat
        super().__init__(mfem.H1_FECollection(1, mesh.Dimension()), model, mesh)
        self._initialize(model)

    def _initialize(self, model):
        ess_tdof = self.essential_tdof_list()
        c = self.generateDomainCoefficient(self.space, model.initialConditions)
        self._X0 = self.initialValue(self.space, c)
        self._M = self.bilinearForm(self.space, ess_tdof, mfem.MassIntegrator(self._mat["Heat Conduction"]["C_v"]))
        self._K = self.bilinearForm(self.space, ess_tdof, mfem.DiffusionIntegrator(self._mat["Heat Conduction"]["k"]))
        self._B = self.linearForm(self.space, ess_tdof, self._K, self.x0, [], self._boundary_linear(model))

    def _boundary_linear(self, model):
        res = []
        # neumann boundary
        neumann = [b for b in model.boundaryConditions if isinstance(b, NeumannBoundary)]
        if len(neumann) != 0:
            c = self.generateSurfaceCoefficient(self.space, neumann)
            res.append(mfem.VectorBoundaryLFIntegrator(c))
        return res

    def RecoverFEMSolution(self, X):
        self._X0 = X
        self.gf = mfem.GridFunction(self.space)
        self.gf.SetFromTrueDofs(X)
        return self.gf
