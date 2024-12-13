from lys_fem.ngs import NGSModel, grad, dx, util, time
from . import ExternalMagneticField, UniaxialAnisotropy, CubicAnisotropy, MagneticScalarPotential, SpinTransferTorque

class NGSLLGModel(NGSModel):
    def __init__(self, model, mesh, vars):
        super().__init__(model, mesh, vars)
        self._model = model

        for eq in model.equations:
            self.addVariable(eq.variableName, 3, region = eq.geometries, order=model.order)

            if self._model.constraint == "Alouges":
                self.addVariable(eq.variableName+"_v", 3, initialValue=None, region = eq.geometries, order=model.order)
                self.addVariable(eq.variableName+"_lam", 1, initialValue=None, dirichlet=None, region=eq.geometries, order=0, isScalar=True, L2=True)

            if self._model.constraint == "Lagrange":
                self.addVariable(eq.variableName+"_lam", 1, initialValue=None, dirichlet=None, region=eq.geometries, order=0, isScalar=True, L2=True)

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
            m0 = m.value

            # Left-hand side, normalization, exchange term
            wf += m.t.dot(test_m)*dx
            wf += A * grad(m).cross(m0).ddot(grad(test_m))*dx
            wf += -alpha * m0.cross(m.t).dot(test_m)*dx

            if self._model.constraint == "Lagrange":
                lam, test_lam = vars[eq.variableName+"_lam"]
                scale = util.max(mat.const.dti, 1)#1e11
                wf += 2*lam*m0.dot(test_m)*scale*dx
                wf += (-1e-5*lam + (m0.dot(m0)-1))*test_lam*scale*dx

            for ex in self._model.domainConditions.get(ExternalMagneticField):
                B = mat[ex.values]
                wf += g*m.cross(B).dot(test_m)*dx(ex.geometries)
            
            for uni in self._model.domainConditions.get(UniaxialAnisotropy):
                u, Ku = mat["u_Ku/norm(u_Ku)"], mat["Ku"]
                B = 2*Ku/Ms*m0.dot(u)*u
                wf += g*m.cross(B).dot(test_m)*dx(uni.geometries)

            for sc in self._model.domainConditions.get(MagneticScalarPotential):
                phi = mat[sc.values]
                wf += g*m0.cross(-mu0*grad(phi)).dot(test_m)*dx(sc.geometries)

            for st in self._model.domainConditions.get(SpinTransferTorque):
                beta = mat["beta_st"]
                u = mu_B/e*Ms * mat[st.values]
                wf += u.dot(grad(m)).dot(test_m) - beta * m.cross(u.dtot(grad(m))).dot(test_m)

        return wf

    def _weakform_alouges(self, vars, mat):
        g, e, mu_B, mu0, Ms, dti = mat.const.g_e, mat.const.e, mat.const.mu_B, mat.const.mu_0, mat["Ms"], mat.const.dti
        A = 2*mat["Aex"] * g / Ms
        alpha = mat["alpha"]

        wf = 0
        theta = 1
        for eq in self._model.equations:
            m, test_m = vars[eq.variableName]
            v, test_v = vars[eq.variableName+"_v"]
            lam, test_lam = vars[eq.variableName+"_lam"]

            # Left-hand side, normalization term
            wf += m.cross(v).dot(test_v)*dx + alpha*v.dot(test_v)*dx
            wf += 1e-5*lam*test_lam*dx+(lam*m.dot(test_v) + v.dot(m)*test_lam)*dx

            # Exchange term
            wf += A*grad(m).ddot(grad(test_v))*dx
            wf += A*grad(v).ddot(grad(test_v))*theta/dti*dx
            wf += A*grad(m).ddot(grad(m))*v.dot(test_v)*theta/dti*dx

            for ex in self._model.domainConditions.get(ExternalMagneticField):
                B = mat[ex.values]
                wf += -g*B.dot(test_v)*dx(ex.geometries)
                wf += g*m.dot(B)*v.dot(test_v)*theta/dti*dx(ex.geometries)

            for uni in self._model.domainConditions.get(UniaxialAnisotropy):
                u, Ku = mat["u_Ku/norm(u_Ku)"], mat["Ku"]
                B = 2*Ku/Ms*m.dot(u)*u
                wf += -g*B.dot(test_v)*dx(uni.geometries)
                wf += g*m.dot(B)*v.dot(test_v)*theta/dti*dx(uni.geometries)

            for cu in self._model.domainConditions.get(CubicAnisotropy):
                c1, c2, c3, Kc = mat["u_Kc[0]/norm(u_Kc[0])"], mat["u_Kc[1]/norm(u_Kc[1])"], mat["u_Kc[2]/norm(u_Kc[2])"], mat["Kc"]
                B =  -2*Kc/Ms*(c2.dot(m)**2+c3.dot(m)**2)*c1.dot(m)*c1
                B += -2*Kc/Ms*(c3.dot(m)**2+c1.dot(m)**2)*c2.dot(m)*c2
                B += -2*Kc/Ms*(c1.dot(m)**2+c2.dot(m)**2)*c3.dot(m)*c3
                wf += -g*B.dot(test_v)*dx(cu.geometries)
                wf += g*m.dot(B)*v.dot(test_v)*theta/dti*dx(cu.geometries)

            for sc in self._model.domainConditions.get(MagneticScalarPotential):
                phi = mat[sc.values]
                B = -mu0*grad(phi)
                wf += -g*B.dot(test_v)*dx(sc.geometries)
                wf += g*m.dot(B)*v.dot(test_v)*theta/dti*dx(sc.geometries)

            for st in self._model.domainConditions.get(SpinTransferTorque):
                beta = mat["beta_st"]
                u = -mu_B/e/Ms*mat[st.values]/(1+beta**2)
                w = u.dot(grad(m))
                wf += -(m.cross(w) + beta*w).dot(test_v)*dx(st.geometries)

        return wf

    def discretize(self, tnt, sols, dti):
        if self._model.discretization == "LLG Asym":
            d = {}
            for v in self.variables:
                if "_lam" in v.name:
                    continue
                trial = tnt[v][0]
                w = 0
                mn, gn = sols.X(v), sols.grad(v)
                d = time.BackwardEuler.generateWeakforms(v, trial, sols, dti)
                d[trial.value] = (1-w)*trial + w*mn
                d[grad(trial)] = (1-w)*grad(trial) + w*gn
            return d
        return super().discretize(tnt, sols, dti)

    def updater(self, tnt, sols, dti):
        d = super().updater(tnt, sols, dti)
        if self._model.constraint == "Projection":
            for v, (trial, test) in tnt.items():
                d[trial] = trial/util.norm(trial)
        elif self._model.constraint == "Alouges":
            for v in self.variables:
                if "_lam" in v.name or "_v" in v.name:
                    continue
                trial = tnt[v][0]
                vel = [trial_v for vv, (trial_v, test_v) in tnt.items() if vv.name==v.name+"_v"][0]
                d[trial] = (trial + vel/dti)/util.norm(trial + vel/dti)
        return d
