from lys_fem.ngs import NGSModel, util
from lys_fem.ngs.util import NGSFunction, grad, dx, ds, coef
from . import ExternalMagneticField, Demagnetization, UniaxialAnisotropy, GilbertDamping

class NGSLLGModel(NGSModel):
    def __init__(self, model, mesh, mat, order=2):
        super().__init__(model, mesh)
        self._model = model
        self._mat = mat

        init = self._model.initialConditions.coef(self._model.initialConditionTypes[0])
        initialValue = util.generateCoefficient(init, self._mesh)
        dirichlet = util.generateDirichletCondition(self._model)

        for eq in model.equations:
            self.addVariable(eq.variableName, 3, "auto", "auto", region = eq.geometries, order=order)
            self.addVariable(eq.variableName+"_lam", 1, region=eq.geometries, order=2, isScalar=True)

    def weakform(self, vars):
        g, Ms = self._mat["g_LL"], self._mat["M_s"]
        A = coef(2)*self._mat["A_ex"] * g / Ms

        wf = NGSFunction()
        for eq in self._model.equations:
            var1, var2 = vars[eq.variableName], vars[eq.variableName+"_lam"]
            m, test_m = var1.trial, var1.test
            m0 = m.value
            lam, test_lam = var2.trial, var2.test

            wf += m.t.dot(test_m)*dx 

            # Normalization term
            wf += (coef(1e-5) * lam * test_lam + coef(2)*lam*m0.dot(test_m) + (m0.dot(m)-coef(1))*test_lam)*dx

            # Exchange term
            wf -= A * m0.cross(grad(m)).ddot(grad(test_m)) * dx

            if self._model.domainConditions.have(GilbertDamping):
                alpha = self._mat["alpha"]
                for gil in self._model.domainConditions.get(GilbertDamping):
                    #region = self._mesh.Materials(util.generateGeometry(gil.geometries))
                    wf -= alpha * m0.cross(m).dot(test_m)*dx

            B = self.coef(ExternalMagneticField, "B")
            wf += g*m.cross(B).dot(test_m)*dx

            if self._model.domainConditions.have(UniaxialAnisotropy):
                u, Ku = self._mat["u_Ku"], self._mat["Ku"]
                for uni in self._model.domainConditions.get(UniaxialAnisotropy):
                    #region = self._mesh.Materials(util.generateGeometry(uni.geometries))
                    B = coef(2)*Ku/Ms*m.dot(u)*u
                    wf += g*m.cross(B).dot(test_m)*dx

            if self._model.domainConditions.have(Demagnetization):
                for demag in self._model.domainConditions.get(Demagnetization):
                    v2 = vars[demag.values]
                    phi, test_phi = v2.trial, v2.test
                    #region = self._mesh.Materials(util.generateGeometry(eq.geometries))
                    wf += Ms*m.dot(grad(test_phi))*dx

        return wf   

    @property
    def isNonlinear(self):
        return True 