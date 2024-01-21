from ngsolve import grad, dx, ds
from lys_fem.ngs import NGSModel, util, dti
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
            self.addVariable(eq.variableName+"x", 1, [dirichlet[0]], initialValue[0], eq.geometries, order=order)
            self.addVariable(eq.variableName+"y", 1, [dirichlet[1]], initialValue[1], eq.geometries, order=order)
            self.addVariable(eq.variableName+"z", 1, [dirichlet[2]], initialValue[2], eq.geometries, order=order)
            self.addVariable(eq.variableName+"_lam", 1, region=eq.geometries, order=2)

    def bilinearform(self, tnt, sols):
        g = self._mat["g_LL"]
        Ms = self._mat["M_s"]
        A = 2*self._mat["A_ex"] * g / Ms

        wf = 0
        for eq in self._model.equations:
            mx, test_mx = tnt[eq.variableName+"x"]
            my, test_my = tnt[eq.variableName+"y"]
            mz, test_mz = tnt[eq.variableName+"z"]
            m = (mx, my, mz)
            test_m = (test_mx, test_my, test_mz)
            lam, test_lam = tnt[eq.variableName+"_lam"]
            mx0 = sols[eq.variableName+"x"]
            my0 = sols[eq.variableName+"y"]
            mz0 = sols[eq.variableName+"z"]
            m0 = (mx0, my0, mz0)

            wf += util.dot(m, test_m)*dti*dx 
            wf += (1e-5 * lam *test_lam + 2*lam*util.dot(m, test_m) + (util.dot(m,m)-1)*test_lam)*dx
            # Exchange term
            wf += -A * (my*grad(mz) - mz*grad(my)) * grad(test_mx) * dx
            wf += -A * (mz*grad(mx) - mx*grad(mz)) * grad(test_my) * dx
            wf += -A * (mx*grad(my) - my*grad(mx)) * grad(test_mz) * dx

            if self._model.domainConditions.have(GilbertDamping):
                alpha = self._mat["alpha"]
                for gil in self._model.domainConditions.get(GilbertDamping):
                    region = self._mesh.Materials(util.generateGeometry(gil.geometries))
                    wf += -alpha * util.dot(util.cross(m, (mx-mx0, my-my0, mz-mz0)), test_m)*dti*dx(definedon=region)

            c = self._model.domainConditions.coef(ExternalMagneticField)
            if c is not None:
                B = util.generateCoefficient(c, self._mesh)
                wf += g*util.dot(util.cross(m, B), test_m)*dx(definedon=util.generateGeometry(eq.geometries))

            if self._model.domainConditions.have(UniaxialAnisotropy):
                Ku = self._mat["Ku"]
                u = self._mat["u_Ku"]
                for uni in self._model.domainConditions.get(UniaxialAnisotropy):
                    region = self._mesh.Materials(util.generateGeometry(uni.geometries))
                    B = 2*Ku/Ms*util.dot(m,u)*u
                    wf += g*util.dot(util.cross(m, B), test_m)*dx(definedon=region)

            if self._model.domainConditions.have(Demagnetization):
                for demag in self._model.domainConditions.get(Demagnetization):
                    phi, test_phi = tnt[demag.values]
                    region = self._mesh.Materials(util.generateGeometry(eq.geometries))
                    wf += Ms*m*grad(test_phi)*dx(definedon=region)

        return wf
    
    def linearform(self, tnt, sols):
        wf = util.generateCoefficient(0) * dx
        for eq in self._model.equations:
            mx, test_mx = tnt[eq.variableName+"x"]
            my, test_my = tnt[eq.variableName+"y"]
            mz, test_mz = tnt[eq.variableName+"z"]
            mx0 = sols[eq.variableName+"x"]
            my0 = sols[eq.variableName+"y"]
            mz0 = sols[eq.variableName+"z"]
            wf += util.dot((test_mx, test_my, test_mz), (mx0, my0, mz0)) * dti * dx(definedon=self._mesh.Materials(util.generateGeometry(eq.geometries)))
        return wf

    @property
    def isNonlinear(self):
        return True 