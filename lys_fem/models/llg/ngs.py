from lys_fem.ngs import NGSModel, grad, dx
from . import ExternalMagneticField, UniaxialAnisotropy, GilbertDamping

class NGSLLGModel(NGSModel):
    def __init__(self, model, mesh, vars, order=2):
        super().__init__(model, mesh, vars)
        self._model = model

        for eq in model.equations:
            self.addVariable(eq.variableName, 3, region = eq.geometries, order=order)
            self.addVariable(eq.variableName+"_lam", 1, initialValue=None, dirichlet=None, region=eq.geometries, order=2, isScalar=True)

    def weakform(self, vars, mat):
        g, Ms = 1.760859770e11, mat["Ms"]
        A = 2*mat["Aex"] * g / Ms

        wf = 0
        for eq in self._model.equations:
            m, test_m = vars[eq.variableName]
            m0 = m.value
            lam, test_lam = vars[eq.variableName+"_lam"]

            # Left-hand side, normalization, exchange term
            wf += m.t.dot(test_m)*dx
            wf += (1e-5 * lam * test_lam + 2*lam*m0.dot(test_m) + (m0.dot(m)-1)*test_lam)/1e-11*dx
            wf -= A * m0.cross(grad(m)).ddot(grad(test_m)) * dx

            for gil in self._model.domainConditions.get(GilbertDamping):
                wf -= mat["alpha"] * m0.cross(m).dot(test_m)*dx(gil.geometries)

            for ex in self._model.domainConditions.get(ExternalMagneticField):
                B = mat[ex.values]
                wf += g*m.cross(B).dot(test_m)*dx(ex.geometries)
            
            for uni in self._model.domainConditions.get(UniaxialAnisotropy):
                u, Ku = mat["u_Ku"], mat["Ku"]
                B = 2*Ku/Ms*m.dot(u)*u
                wf += g*m.cross(B).dot(test_m)*dx(uni.geometries)
        return wf
