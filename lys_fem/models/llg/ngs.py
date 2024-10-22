from lys_fem.ngs import NGSModel, grad, dx, util, time
from . import ExternalMagneticField, UniaxialAnisotropy, MagneticScalarPotential, SpinTransferTorque

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
            wf += -A * m0.cross(grad(m)).ddot(grad(test_m))*dx
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
                u, Ku = mat["u_Ku"], mat["Ku"]
                B = 2*Ku/Ms*m0.dot(u)*u
                wf += g*m.cross(B).dot(test_m)*dx(uni.geometries)

            for sc in self._model.domainConditions.get(MagneticScalarPotential):
                phi = mat[sc.values]
                wf += g*m0.cross(-mu0*grad(phi.value)).dot(test_m)*dx(sc.geometries)

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
        for eq in self._model.equations:
            m, test_m = vars[eq.variableName]
            v, test_v = vars[eq.variableName+"_v"]
            lam, test_lam = vars[eq.variableName+"_lam"]

            # Left-hand side, normalization, exchange term
            wf += m.cross(v).dot(test_v)*dx + alpha*v.dot(test_v)*dx
            wf += (lam*m.dot(test_v) + v.dot(m)*test_lam)*dx
            wf += A*grad(m).ddot(grad(test_v))*dx
            wf += A*grad(v).ddot(grad(test_v))/dti*dx

            for ex in self._model.domainConditions.get(ExternalMagneticField):
                B = mat[ex.values]
                wf += -g*B.dot(test_v)*dx(ex.geometries)

            for uni in self._model.domainConditions.get(UniaxialAnisotropy):
                u, Ku = mat["u_Ku"], mat["Ku"]
                B = 2*Ku/Ms*m.dot(u)*u
                wf += -g*B.dot(test_v)*dx(uni.geometries)

            for sc in self._model.domainConditions.get(MagneticScalarPotential):
                phi = mat[sc.values]
                B = -mu0*grad(phi.value)
                wf += -g*B.dot(test_v)*dx(sc.geometries)

        return wf

    def discretize(self, sols, dti):
        if self._model.discretization == "LLG Asym":
            d = {}
            for v in self.variables:
                if "_lam" in v.name:
                    continue
                w = 1
                mn, gn = sols.X()[v.name], sols.grad()[v.name]
                d = time.BackwardEuler.generateWeakforms(v, sols, dti)
                d[v.trial.value] = (1-w)*v.trial + w*mn
                d[grad(v.trial)] = (1-w)*grad(v.trial) + w*gn
            return d
        
        if self._model.discretization == "Alouges theta":
            d = {}
            for v in self.variables:
                if "_lam" in v.name or "_v" in v.name:
                    continue
                mn, gn = sols.X()[v.name], sols.grad()[v.name]
                d[grad(v.trial)] = gn
            return d
        return super().discretize(sols, dti)

    def updater(self, sols, dti):
        d = super().updater(sols, dti)
        if self._model.constraint == "Projection":
            for v in self.variables:
                d[v.trial] = v.trial/util.norm(v.trial)
        elif self._model.constraint == "Alouges":
            for v in self.variables:
                if "_lam" in v.name or "_v" in v.name:
                    continue
                vel = [vv for vv in self.variables if vv.name==v.name+"_v"][0]
                d[v.trial] = (v.trial + vel.trial/dti)/util.norm(v.trial + vel.trial/dti)
        return d
