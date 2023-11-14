from lys_fem.mf import mfem, MFEMLinearModel, MFEMNonlinearModel


class MFEMLinearTestModel(MFEMLinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(mfem.H1_FECollection(1, mesh.Dimension()), model, mesh)

    def assemble_a(self):
        a = mfem.BilinearForm(self.space)
        a.AddDomainIntegrator(mfem.DiffusionIntegrator())
        a.Assemble()
        return a


class MFEMNonlinearTestModel(MFEMNonlinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(mfem.H1_FECollection(1, mesh.Dimension()), model, mesh)

    def assemble_a(self):
        self.a = _NonlinearOperator(self)
        return self.a


class _NonlinearOperator(mfem.PyOperator):
    def __init__(self, model):
        super().__init__(model.space.GetTrueVSize())
        self._model = model

    def assemble(self, x, ess_tdof_list):
        self.x = mfem.GridFunction(self._model.space)
        self.x.SetFromTrueDofs(x)
        self.x_c = mfem.GridFunctionCoefficient(self.x)

        self.a = mfem.BilinearForm(self._model.space)
        self.a.AddDomainIntegrator(mfem.DiffusionIntegrator(self.x_c))
        self.a.Assemble()
        self.A = mfem.SparseMatrix()
        self.a.FormSystemMatrix(ess_tdof_list, self.A)

        self.x2 = mfem.GridFunction(self._model.space)
        self.x2.SetFromTrueDofs(x)
        self.x2.Neg()
        self.dx = mfem.GridFunction(self._model.space)
        self.x2.GetDerivative(1, 0, self.dx)
        self.dx_c = mfem.GridFunctionCoefficient(self.dx)

        self.da = mfem.BilinearForm(self._model.space)
        self.da.AddDomainIntegrator(mfem.DiffusionIntegrator(self.x_c))
        self.da.AddDomainIntegrator(mfem.MixedScalarWeakDerivativeIntegrator(self.dx_c))
        self.da.Assemble()
        self.DA = mfem.SparseMatrix()
        self.da.FormSystemMatrix(ess_tdof_list, self.DA)

    def EliminateVDofsInRHS(self, ess_tdof_list, x, b):
        self.tmp = mfem.GridFunction(self._model.space, b)
        self._B = mfem.Vector()
        self.tmp.GetTrueDofs(self._B)
        self.a.EliminateVDofsInRHS(ess_tdof_list, x, self._B)
        return self._B
