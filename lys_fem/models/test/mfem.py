from lys_fem.mf import mfem, MFEMLinearModel, MFEMNonlinearModel


class MFEMLinearTestModel(MFEMLinearModel):
    def __init__(self, fec, model, mesh, mat):
        super().__init__(fec, model, mesh)

    def assemble_a(self):
        a = mfem.BilinearForm(self.space)
        a.AddDomainIntegrator(mfem.DiffusionIntegrator())
        a.Assemble()
        return a


class MFEMNonlinearTestModel(MFEMNonlinearModel):
    def __init__(self, fec, model, mesh, mat):
        super().__init__(fec, model, mesh)

    def assemble_a(self):
        self.a = _NonlinearOperator(self)
        return self.a


class _NonlinearOperator(mfem.PyOperator):
    def __init__(self, model):
        super().__init__(model.space.GetTrueVSize())
        self._model = model
        self._init = False

    def _assemble(self, x):
        self.x = mfem.GridFunction(self._model.space)
        self.x.SetFromTrueDofs(x)
        self.x_c = mfem.GridFunctionCoefficient(self.x)

        self.a = mfem.BilinearForm(self._model.space)
        self.a.AddDomainIntegrator(mfem.DiffusionIntegrator(self.x_c))
        self.a.Assemble()

        self.x2 = mfem.GridFunction(self._model.space)
        self.x2.SetFromTrueDofs(mfem.Vector([-xx for xx in x]))
        self.dx = mfem.GridFunction(self._model.space)
        self.x2.GetDerivative(1, 0, self.dx)
        self.dx_c = mfem.GridFunctionCoefficient(self.dx)

        self.da = mfem.BilinearForm(self._model.space)
        self.da.AddDomainIntegrator(mfem.DiffusionIntegrator(self.x_c))
        self.da.AddDomainIntegrator(mfem.MixedScalarWeakDerivativeIntegrator(self.dx_c))
        self.da.Assemble()

    def initializeRHS(self, ess_tdof_list, x, b):
        self._assemble(x)
        A = mfem.SparseMatrix()
        self.B = mfem.Vector()
        self.X = mfem.Vector()
        self.a.FormLinearSystem(ess_tdof_list, x, b, A, self.X, self.B)
        return self.B, self.X

    def Mult(self, x, y):
        if self._init is False:
            self._init = True
        else:
            self._assemble(x)
        self.A = mfem.SparseMatrix()
        self.a.FormSystemMatrix(self._model.essential_tdof_list(), self.A)
        self.A.Mult(x, y)

    def GetGradient(self, x):
        self.dA = mfem.SparseMatrix()
        self.da.FormSystemMatrix(self._model.essential_tdof_list(), self.dA)
        return self.dA
