from lys_fem.mf import mfem, MFEMModel


class MFEMPoissonModel(MFEMModel):
    def __init__(self, model, mesh, mat):
        super().__init__(mfem.H1_FECollection(2, mesh.Dimension()), model, mesh)
        self._Jac = None

    def assemble_k(self):
        a = mfem.BilinearForm(self.space)
        a.AddDomainIntegrator(mfem.DiffusionIntegrator())
        a.Assemble()
        return a
    
    def M(self):
        return None

    @property
    def K(self):
        if not self._initK:
            self._initK = True
            self._K = mfem.SparseMatrix()
            self.ess_tdof_list = self.essential_tdof_list()
            self._k = self.assemble_k()
            self._k.FormSystemMatrix(self.ess_tdof_list, self._K)
        return self._K

    @property
    def b(self):
        if not self._initB:
            self._initB = True
            self.ess_tdof_list = self.essential_tdof_list()
            self._B = mfem.Vector()

            self._b = self.assemble_b()
            tmp = mfem.GridFunction(self.space, self._b)
            tmp.GetTrueDofs(self._B)
            K = self.K
            self._k.EliminateVDofsInRHS(self.ess_tdof_list, self.x0, self._B)
        return self._B
