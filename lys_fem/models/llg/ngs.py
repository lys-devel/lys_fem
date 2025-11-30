from lys_fem.ngs import NGSModel, grad, dx, util
from lys_fem.util import g_e, e, mu_B, mu_0, dti, k_B, Ve
from . import ExternalMagneticField, UniaxialAnisotropy, CubicAnisotropy, MagneticScalarPotential, CubicMagnetoStriction, SpinTransferTorque, ThermalFluctuation, CubicMagnetoRotationCoupling, BarnettEffect

class NGSLLGModel(NGSModel):
    def __init__(self, model):
        super().__init__(model)
        self._model = model

    def weakform(self, vars, mat):
        if self._model.constraint == "Alouges":
            return self._weakform_alouges(vars, mat)
        else:
            return self._weakform_default(vars, mat)

    def _weakform_default(self, vars, mat):
        Ms = mat["Ms"]
        A = 2*mat["Aex"] * g_e / Ms
        alpha = mat["alpha_LLG"]

        wf = 0
        m, test_m = vars[self._model.variableName]

        # Left-hand side, normalization, exchange term
        wf += m.t.dot(test_m)*dx
        wf += A * grad(m).cross(m).ddot(grad(test_m))*dx
        wf += -alpha * m.cross(m.t).dot(test_m)*dx

        if self._model.constraint == "Lagrange":
            lam, test_lam = vars[self._model.variableName+"_lam"]
            scale = util.max(dti, 1)#1e11
            wf += 2*lam*m.dot(test_m)*scale*dx
            wf += (-1e-5*lam + (m.dot(m)-1))*test_lam*scale*dx

        for ex in self._model.domainConditions.get(ExternalMagneticField):
            B = mat[ex.values]
            wf += g_e*m.cross(B).dot(test_m)*dx(ex.geometries)
        
        for uni in self._model.domainConditions.get(UniaxialAnisotropy):
            u, Ku = mat["u_Ku/norm(u_Ku)"], mat["Ku"]
            B = 2*Ku/Ms*m.dot(u)*u
            wf += g_e*m.cross(B).dot(test_m)*dx(uni.geometries)

        for sc in self._model.domainConditions.get(MagneticScalarPotential):
            phi = mat[sc.values]
            wf += g_e*m.cross(-mu_0*grad(phi)).dot(test_m)*dx(sc.geometries)

        for st in self._model.domainConditions.get(SpinTransferTorque):
            beta = mat["beta_st"]
            u = mu_B/e*Ms * mat[st.values]
            wf += u.dot(grad(m)).dot(test_m) - beta * m.cross(u.dtot(grad(m))).dot(test_m)

        return wf

    def _weakform_alouges(self, vars, mat):
        Ms = mat["Ms"]
        A = 2*mat["Aex"] * g_e / Ms
        alpha = mat["alpha_LLG"]

        wf = 0
        theta = 1
        m, test_m = vars[self._model.variableName]
        lam, test_lam = vars[self._model.variableName+"_lam"]

        # Left-hand side, normalization term
        wf += m.cross(m.t).dot(test_m)*dx + alpha*m.t.dot(test_m)*dx
        wf += (lam*m.dot(test_m) + m.t.dot(m)*test_lam)*dx

        # Exchange term
        wf += A*grad(m).ddot(grad(test_m))*dx
        wf += A*grad(m.t).ddot(grad(test_m))*theta/dti*dx
        wf += A*grad(m).ddot(grad(m))*m.t.dot(test_m)*theta/dti*dx

        for ex in self._model.domainConditions.get(ExternalMagneticField):
            B = mat[ex.values]
            wf += -g_e*B.dot(test_m)*dx(ex.geometries)
            wf += g_e*m.dot(B)*m.t.dot(test_m)*theta/dti*dx(ex.geometries)

        for uni in self._model.domainConditions.get(UniaxialAnisotropy):
            u, Ku = mat["u_Ku/norm(u_Ku)"], mat["Ku"]
            B = 2*Ku/Ms*m.dot(u)*u
            wf += -g_e*B.dot(test_m)*dx(uni.geometries)
            wf += g_e*m.dot(B)*m.t.dot(test_m)*theta/dti*dx(uni.geometries)

        for cu in self._model.domainConditions.get(CubicAnisotropy):
            Kc = mat["Kc"]
            B = -4/Ms*util.einsum("ijkl,j,k,l->i", Kc, m, m, m)
            wf += -g_e*B.dot(test_m)*dx(cu.geometries)

        for cms in self._model.domainConditions.get(CubicMagnetoStriction):
            Ks = mat["K_MS"]
            du = util.grad(mat[cms.values])
            e = (du + du.T)/2
            B = -2/Ms*util.einsum("ijkl,j,kl->i", Ks, m, e)
            wf += -g_e*B.dot(test_m)*dx(cms.geometries)
            #wf += g*m.dot(B)*m.t.dot(test_m)*theta/dti*dx(cms.geometries)

        for cms in self._model.domainConditions.get(CubicMagnetoRotationCoupling):
            Kc = mat["Kc"]
            du = util.grad(mat[cms.values])
            w = (du.T-du)/2
            K = util.einsum("ijkl,Ii->Ijkl", Kc, w) + util.einsum("ijkl,Jj->iJkl", Kc, w) + util.einsum("ijkl,Kk->ijKl", Kc, w) + util.einsum("ijkl,Ll->ijkL", Kc, w)
            B = -4/Ms*util.einsum("ijkl,j,k,l->i", K, m, m, m)
            wf += -g_e*B.dot(test_m)*dx(cms.geometries)

        for bar in self._model.domainConditions.get(BarnettEffect):
            u = mat[bar.values]
            rut = util.rot(u.t)/2
            wf += -rut.dot(test_m) * dx(bar.geometries)
            #wf += g*m.dot(B)*m.t.dot(test_m)*theta/dti*dx(bar.geometries)

        for sc in self._model.domainConditions.get(MagneticScalarPotential):
            phi = mat[sc.values]
            B = -mu0*grad(phi)
            wf += -g_e*B.dot(test_m)*dx(sc.geometries)
            #wf += g*m.dot(B)*m.t.dot(test_m)*theta/dti*dx(sc.geometries)

        for st in self._model.domainConditions.get(SpinTransferTorque):
            beta = mat["beta_st"]
            u = -mu_B/e/Ms*mat[st.values]/(1+beta**2)
            w = u.dot(grad(m))
            wf += -(m.cross(w) + beta*w).dot(test_m)*dx(st.geometries)

        for th in self._model.domainConditions.get(ThermalFluctuation):
            T, R, Ve = mat[th.T], mat[th.R], Ve
            D = alpha*kB*T*dti/(Ms*g_e*Ve)
            B = util.sqrt(2*D)*R
            wf += -g_e*B.dot(test_m)*dx(th.geometries)

        return wf

    def discretize(self, dti):
        if self._model.constraint == "Alouges":
            d = {}
            for v in self.variables:
                if "_lam" in v.name:
                    continue
                trial = util.trial(v)
                mn = util.prev(trial)
                d[trial] = mn
                d[trial.t] = trial
            return d
            
        return super().discretize(dti)

    def updater(self, dti):
        d = super().updater(dti)
        if self._model.constraint == "Projection":
            for v in self.variables:
                trial = util.trial(v)
                d[trial] = trial/util.norm(trial)
        elif self._model.constraint == "Alouges":
            for v in self.variables:
                if "_lam" in v.name:
                    continue
                trial = util.trial(v)
                m = util.prev(trial) + trial/dti
                d[trial] = m/util.norm(m)
                d[trial.t] = trial
        return d
