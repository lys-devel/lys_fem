from lys_fem.ngs import NGSModel, grad, dx, util
from . import ExternalMagneticField, UniaxialAnisotropy, MagneticScalarPotential

class NGSLLGModel(NGSModel):
    def __init__(self, model, mesh, vars, order=2):
        super().__init__(model, mesh, vars)
        self._model = model

        for eq in model.equations:
            self.addVariable(eq.variableName, 3, region = eq.geometries, order=order)
            self.addVariable(eq.variableName+"_lam", 1, initialValue=None, dirichlet=None, region=eq.geometries, order=1, isScalar=True, L2=True)

    def weakform(self, vars, mat):
        g, mu0, Ms = mat.const.g_e, mat.const.mu_0, mat["Ms"]
        A = 2*mat["Aex"] * g / Ms
        alpha = mat["alpha"]

        wf = 0
        for eq in self._model.equations:
            m, test_m = vars[eq.variableName]
            m0 = m.value
            lam, test_lam = vars[eq.variableName+"_lam"]
            scale = util.max(mat.const.dti, 1)#1e11

            # Left-hand side, normalization, exchange term
            wf += (m.t + 2*lam*m0*scale).dot(test_m)*dx
            wf += (-1e-5*lam + (m0.dot(m)-1))*test_lam*scale*dx
            wf += -A * m0.cross(grad(m)).ddot(grad(test_m))*dx
            wf += -alpha * m.cross(m.t).dot(test_m)*dx

            for ex in self._model.domainConditions.get(ExternalMagneticField):
                B = mat[ex.values]
                wf += g*m.cross(B).dot(test_m)*dx(ex.geometries)
            
            for uni in self._model.domainConditions.get(UniaxialAnisotropy):
                u, Ku = mat["u_Ku"], mat["Ku"]
                B = 2*Ku/Ms*m.dot(u)*u
                wf += g*m.cross(B).dot(test_m)*dx(uni.geometries)

            for sc in self._model.domainConditions.get(MagneticScalarPotential):
                phi = mat[sc.values]
                wf += g*m.cross(-mu0*grad(phi.value)).dot(test_m)*dx(sc.geometries)

        return wf