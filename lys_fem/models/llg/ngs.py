from lys_fem.ngs import NGSModel, grad, dx
from . import ExternalMagneticField, Demagnetization, UniaxialAnisotropy, GilbertDamping

class NGSLLGModel(NGSModel):
    def __init__(self, model, mesh, order=2):
        super().__init__(model, mesh)
        self._model = model

        for eq in model.equations:
            self.addVariable(eq.variableName, 3, region = eq.geometries, order=order)
            self.addVariable(eq.variableName+"_lam", 1, initialValue=None, dirichlet=None, region=eq.geometries, order=2, isScalar=True)

    def weakform(self, vars, mat):
        g, Ms = mat["g_LL"], mat["M_s"]
        A = 2*mat["A_ex"] * g / Ms

        wf = 0
        for eq in self._model.equations:
            m, test_m = vars[eq.variableName]
            m0 = m.value
            lam, test_lam = vars[eq.variableName+"_lam"]

            # Left-hand side, normalization, exchange term
            wf += m.t.dot(test_m)*dx
            wf += (1e-5 * lam * test_lam + 2*lam*m0.dot(test_m) + (m0.dot(m)-1)*test_lam)/1e-11*dx
            wf -= A * m0.cross(grad(m)).ddot(grad(test_m)) * dx

            if self._model.domainConditions.have(GilbertDamping):
                alpha = mat["alpha"]
                for gil in self._model.domainConditions.get(GilbertDamping):
                    #region = self._mesh.Materials(util.generateGeometry(gil.geometries))
                    wf -= alpha * m0.cross(m).dot(test_m)*dx

            B = self.coef(ExternalMagneticField, "B")
            wf += g*m.cross(B).dot(test_m)*dx

            if self._model.domainConditions.have(UniaxialAnisotropy):
                u, Ku = mat["u_Ku"], mat["Ku"]
                for uni in self._model.domainConditions.get(UniaxialAnisotropy):
                    #region = self._mesh.Materials(util.generateGeometry(uni.geometries))
                    B = 2*Ku/Ms*m.dot(u)*u
                    wf += g*m.cross(B).dot(test_m)*dx

            if self._model.domainConditions.have(Demagnetization):
                for demag in self._model.domainConditions.get(Demagnetization):
                    phi, test_phi = vars[demag.values]
                    #region = self._mesh.Materials(util.generateGeometry(eq.geometries))
                    wf += Ms*m.dot(grad(test_phi))*dx

        return wf
