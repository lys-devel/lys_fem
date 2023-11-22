from lys_fem.mf import mfem, util, MFEMLinearModel, MFEMNonlinearModel


class MFEMLinearTestModel(MFEMLinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(mfem.H1_FECollection(1, mesh.Dimension()), model, mesh)
        ess_tdof = self.essential_tdof_list()
        c = self.generateDomainCoefficient(self.space, model.initialConditions)
        self._X0 = util.initialValue(self.space, c)
        self._K = util.bilinearForm(self.space, ess_tdof, domainInteg=[mfem.DiffusionIntegrator()])
        self._B = util.linearForm(self.space, ess_tdof, self._K, self.x0)

    def RecoverFEMSolution(self, X):
        self._X0 = X
        self.gf = mfem.GridFunction(self.space)
        self.gf.SetFromTrueDofs(X)
        return self.gf


class MFEMNonlinearTestModel(MFEMNonlinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(mfem.H1_FECollection(1, mesh.Dimension()), model, mesh)
        self.ess_tdof_list = self.essential_tdof_list()
        c = self.generateDomainCoefficient(self.space, model.initialConditions)
        self._X0 = util.initialValue(self.space, c)

    def update(self, u):
        uc, duc = self._createCoef(u)
        self._K = util.bilinearForm(self.space, self.ess_tdof_list, domainInteg=mfem.DiffusionIntegrator(uc))
        self._dK = util.bilinearForm(self.space, self.ess_tdof_list, domainInteg=mfem.MixedScalarWeakDerivativeIntegrator(duc))
        self._B = util.linearForm(self.space, self.ess_tdof_list, self._K, self.x0)

    def RecoverFEMSolution(self, X):
        self._X0 = X
        x = mfem.GridFunction(self.space)
        x.SetFromTrueDofs(X)
        return x

    def _createCoef(self, u):
        self.u_gf = mfem.GridFunction(self.space)
        self.u_gf.SetFromTrueDofs(u)
        self.u_c = mfem.GridFunctionCoefficient(self.u_gf)

        self._du_gf = mfem.GridFunction(self.space)
        self.u_gf.GetDerivative(1, 0, self._du_gf)
        self._du_c = mfem.GridFunctionCoefficient(self._du_gf)

        return self.u_c, self._du_c

    @property
    def K(self):
        return self._K

    @property
    def grad_Kx(self):
        return self._K - self._dK

    @property
    def b(self):
        return self._B

    def grad_b(self):
        return None

    @property
    def x0(self):
        return self._X0

