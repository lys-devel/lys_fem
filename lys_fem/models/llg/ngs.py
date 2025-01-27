from lys_fem.ngs import NGSModel, grad, dx, util
from . import ExternalMagneticField, UniaxialAnisotropy, CubicAnisotropy, MagneticScalarPotential, CubicMagnetoStriction, SpinTransferTorque, ThermalFluctuation

class NGSLLGModel(NGSModel):
    def __init__(self, model, vars):
        super().__init__(model, vars)
        self._model = model

        for eq in model.equations:
            if self._model.constraint == "Alouges":
                self.addVariable(eq.variableName, 3, region = eq.geometries, type="v", order=model.order, fetype=model.type)
                self.addVariable(eq.variableName+"_lam", 1, initialValue=None, dirichlet=None, region=eq.geometries, order=model.order-1, fetype=model.type, isScalar=True)

            elif self._model.constraint == "Lagrange":
                self.addVariable(eq.variableName, 3, region = eq.geometries, order=model.order)
                self.addVariable(eq.variableName+"_lam", 1, initialValue=None, dirichlet=None, region=eq.geometries, order=0, isScalar=True, fetype="L2")
            
            else:
                self.addVariable(eq.variableName, 3, region = eq.geometries, order=model.order)

    def weakform(self, vars, mat):
        if self._model.constraint == "Alouges":
            return self._weakform_alouges(vars, mat)
        else:
            return self._weakform_default(vars, mat)

    def _weakform_default(self, vars, mat):
        g, e, mu_B, mu0, Ms = mat.const.g_e, mat.const.e, mat.const.mu_B, mat.const.mu_0, mat["Ms"]
        A = 2*mat["Aex"] * g / Ms
        alpha = mat["alpha"]

        wf = 0
        for eq in self._model.equations:
            m, test_m = vars[eq.variableName]

            # Left-hand side, normalization, exchange term
            wf += m.t.dot(test_m)*dx
            wf += A * grad(m).cross(m).ddot(grad(test_m))*dx
            wf += -alpha * m.cross(m.t).dot(test_m)*dx

            if self._model.constraint == "Lagrange":
                lam, test_lam = vars[eq.variableName+"_lam"]
                scale = util.max(mat.const.dti, 1)#1e11
                wf += 2*lam*m.dot(test_m)*scale*dx
                wf += (-1e-5*lam + (m.dot(m)-1))*test_lam*scale*dx

            for ex in self._model.domainConditions.get(ExternalMagneticField):
                B = mat[ex.values]
                wf += g*m.cross(B).dot(test_m)*dx(ex.geometries)
            
            for uni in self._model.domainConditions.get(UniaxialAnisotropy):
                u, Ku = mat["u_Ku/norm(u_Ku)"], mat["Ku"]
                B = 2*Ku/Ms*m.dot(u)*u
                wf += g*m.cross(B).dot(test_m)*dx(uni.geometries)

            for sc in self._model.domainConditions.get(MagneticScalarPotential):
                phi = mat[sc.values]
                wf += g*m.cross(-mu0*grad(phi)).dot(test_m)*dx(sc.geometries)

            for st in self._model.domainConditions.get(SpinTransferTorque):
                beta = mat["beta_st"]
                u = mu_B/e*Ms * mat[st.values]
                wf += u.dot(grad(m)).dot(test_m) - beta * m.cross(u.dtot(grad(m))).dot(test_m)

        return wf

    def _weakform_alouges(self, vars, mat):
        g, e, mu_B, mu0, Ms, kB, dti = mat.const.g_e, mat.const.e, mat.const.mu_B, mat.const.mu_0, mat["Ms"], mat.const.k_B, mat.const.dti
        A = 2*mat["Aex"] * g / Ms
        alpha = mat["alpha"]

        wf = 0
        theta = 1
        for eq in self._model.equations:
            m, test_m = vars[eq.variableName]
            lam, test_lam = vars[eq.variableName+"_lam"]

            # Left-hand side, normalization term
            wf += m.cross(m.t).dot(test_m)*dx + alpha*m.t.dot(test_m)*dx
            wf += (lam*m.dot(test_m) + m.t.dot(m)*test_lam)*dx

            # Exchange term
            wf += A*grad(m).ddot(grad(test_m))*dx
            wf += A*grad(m.t).ddot(grad(test_m))*theta/dti*dx
            wf += A*grad(m).ddot(grad(m))*m.t.dot(test_m)*theta/dti*dx

            for ex in self._model.domainConditions.get(ExternalMagneticField):
                B = mat[ex.values]
                wf += -g*B.dot(test_m)*dx(ex.geometries)
                wf += g*m.dot(B)*m.t.dot(test_m)*theta/dti*dx(ex.geometries)

            for uni in self._model.domainConditions.get(UniaxialAnisotropy):
                u, Ku = mat["u_Ku/norm(u_Ku)"], mat["Ku"]
                B = 2*Ku/Ms*m.dot(u)*u
                wf += -g*B.dot(test_m)*dx(uni.geometries)
                wf += g*m.dot(B)*m.t.dot(test_m)*theta/dti*dx(uni.geometries)

            for cu in self._model.domainConditions.get(CubicAnisotropy):
                c1, c2, c3, Kc = mat["u_Kc[0]/norm(u_Kc[0])"], mat["u_Kc[1]/norm(u_Kc[1])"], mat["u_Kc[2]/norm(u_Kc[2])"], mat["Kc"]
                B =  -2*Kc/Ms*(c2.dot(m)**2+c3.dot(m)**2)*c1.dot(m)*c1
                B += -2*Kc/Ms*(c3.dot(m)**2+c1.dot(m)**2)*c2.dot(m)*c2
                B += -2*Kc/Ms*(c1.dot(m)**2+c2.dot(m)**2)*c3.dot(m)*c3
                wf += -g*B.dot(test_m)*dx(cu.geometries)
                #wf += g*m.dot(B)*m.t.dot(test_m)*theta/dti*dx(cu.geometries)

            for sc in self._model.domainConditions.get(MagneticScalarPotential):
                phi = mat[sc.values]
                B = -mu0*grad(phi)
                wf += -g*B.dot(test_m)*dx(sc.geometries)
                #wf += g*m.dot(B)*m.t.dot(test_m)*theta/dti*dx(sc.geometries)

            for cms in self._model.domainConditions.get(CubicMagnetoStriction):
                C, l100, l111 = mat["C"], mat["lam100"], mat["lam111"]
                du = util.grad(mat[cms.values])
                e = (du + du.T)/2
                B1, B2 = -3*l100*(C[0,1,0,0]-C[0,0,0,0])/Ms, - 6*l111*C[0,1,0,1]/Ms
                B = B1*e.diag()*m + B2*e.offdiag()*m
                wf += -g*B.dot(test_m)*dx(cms.geometries)
                print(wf)
                #wf += g*m.dot(B)*m.t.dot(test_m)*theta/dti*dx(cms.geometries)

            for st in self._model.domainConditions.get(SpinTransferTorque):
                beta = mat["beta_st"]
                u = -mu_B/e/Ms*mat[st.values]/(1+beta**2)
                w = u.dot(grad(m))
                wf += -(m.cross(w) + beta*w).dot(test_m)*dx(st.geometries)

            for th in self._model.domainConditions.get(ThermalFluctuation):
                T, R, Ve = mat[th.T], mat[th.R], mat.const.Ve
                D = alpha*kB*T*dti/(Ms*g*Ve)
                B = util.sqrt(2*D)*R
                wf += -g*B.dot(test_m)*dx(th.geometries)

        return wf

    def discretize(self, tnt, sols, dti):
        if self._model.constraint == "Alouges":
            d = {}
            for v in self.variables:
                if "_lam" in v.name:
                    continue
                trial, _ = tnt[v]
                mn = sols.X(v)
                d[trial] = mn
                d[trial.t] = trial
            return d
            
        return super().discretize(tnt, sols, dti)

    def updater(self, tnt, sols, dti):
        d = super().updater(tnt, sols, dti)
        if self._model.constraint == "Projection":
            for v in self.variables:
                trial, _ = tnt[v]
                d[trial] = trial/util.norm(trial)
        elif self._model.constraint == "Alouges":
            for v in self.variables:
                if "_lam" in v.name:
                    continue
                trial = tnt[v][0]
                m = sols.X(v) + trial/dti
                d[trial] = m/util.norm(m)
                d[trial.t] = trial
        return d
