from lys_fem.mf import mfem, util, MFEMLinearModel, MFEMNonlinearModel


class MFEMLinearTestModel(MFEMLinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model)
        self._fec = mfem.H1_FECollection(1, mesh.Dimension())
        self._space = mfem.FiniteElementSpace(mesh, self._fec, 1, mfem.Ordering.byVDIM)
        ess_tdof = self.essential_tdof_list(self._space)
        c = util.generateDomainCoefficient(self._space, model.initialConditions)
        self.x0 = util.initialValue(self._space, c)
        self.K = util.bilinearForm(self._space, ess_tdof, domainInteg=mfem.DiffusionIntegrator())
        self.b = util.linearForm(self._space, ess_tdof, self.K, self.x0)
    
    @property
    def solution(self):
        gf = mfem.GridFunction(self._space)
        gf.SetFromTrueDofs(self.x0)
        return gf


class MFEMNonlinearTestModel(MFEMNonlinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model)
        self._fec = mfem.H1_FECollection(1, mesh.Dimension())
        self._space = mfem.FiniteElementSpace(mesh, self._fec, 1, mfem.Ordering.byVDIM)
        self.ess_tdof_list = self.essential_tdof_list(self._space)
        self.x0 = util.initialValue(self._space, util.generateDomainCoefficient(self._space, model.initialConditions))

    def update(self, u):
        uc, duc = self._createCoef(u)
        self.K = util.bilinearForm(self._space, self.ess_tdof_list, domainInteg=mfem.DiffusionIntegrator(uc))
        self.grad_Kx = self.K - util.bilinearForm(self._space, self.ess_tdof_list, domainInteg=mfem.MixedScalarWeakDerivativeIntegrator(duc))
        self.b = util.linearForm(self._space, self.ess_tdof_list, self.K, self.x0)
    
    @property
    def solution(self):
        x = mfem.GridFunction(self._space)
        x.SetFromTrueDofs(self.x0)
        return x

    def _createCoef(self, u):
        self.u_gf = mfem.GridFunction(self._space)
        self.u_gf.SetFromTrueDofs(u)
        self.u_c = mfem.GridFunctionCoefficient(self.u_gf)

        self._du_gf = mfem.GridFunction(self._space)
        self.u_gf.GetDerivative(1, 0, self._du_gf)
        self._du_c = mfem.GridFunctionCoefficient(self._du_gf)

        return self.u_c, self._du_c
