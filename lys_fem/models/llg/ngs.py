from ngsolve import grad, dx, ds, div, specialcf
from lys_fem.ngs import NGSModel, util, dti
from . import ExternalMagneticField, Demagnetization, UniaxialAnisotropy, GilbertDamping

class NGSLLGModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh)
        self._model = model
        self._mat = mat
        for eq in model.equations:
            self.addVariable(eq.variableName, eq.variableDimension, "auto", "auto", eq.geometries)
            self.addVariable(eq.variableName+"_lam", 1, region=eq.geometries)

    def bilinearform(self, tnt, sols):
        g = util.generateCoefficient(1.760859770e11)
        Ms = self._mat["M_s"]

        wf = 0
        for eq in self._model.equations:
            m, test_m = tnt[eq.variableName]
            lam, test_lam = tnt[eq.variableName+"_lam"]
            m0 = sols[eq.variableName]
            wf += m*test_m*dti*dx 
            wf += (2*lam*m * test_m + (m*m-1)*test_lam)*dx

            if self._model.domainConditions.have(GilbertDamping):
                alpha = self._mat["alpha"]
                for gil in self._model.domainConditions.get(GilbertDamping):
                    region = self._mesh.Materials(util.generateGeometry(gil.geometries))
                    wf += -alpha * util.CrossProduct(m, m-m0)*dti*test_m*dx(definedon=region)

            if self._model.domainConditions.have(ExternalMagneticField):
                B = util.generateGeometryCoefficient(self._mesh, self._model.domainConditions.get(ExternalMagneticField))
                wf += g*util.CrossProduct(m, B)*test_m*dx(definedon=util.generateGeometry(eq.geometries))

            if self._model.domainConditions.have(UniaxialAnisotropy):
                Ku = self._mat["Ku"]
                for uni in self._model.domainConditions.get(UniaxialAnisotropy):
                    region = self._mesh.Materials(util.generateGeometry(uni.geometries))
                    B = 2*(Ku*m)*(Ku/Ku.Norm())/Ms
                    wf += g*util.CrossProduct(m, B)*test_m*dx(definedon=region)

            if self._model.domainConditions.have(Demagnetization):
                for demag in self._model.domainConditions.get(Demagnetization):
                    phi, test_phi = tnt[demag.values]
                    region = self._mesh.Materials(util.generateGeometry(eq.geometries))
                    wf += Ms*m*grad(test_phi)*dx(definedon=region)

        return wf
    
    def linearform(self, tnt, sols):
        wf = util.generateCoefficient(0) * dx
        for eq in self._model.equations:
            m, test_m = tnt[eq.variableName]
            s = sols[eq.variableName]
            wf += s * test_m * dti * dx(definedon=self._mesh.Materials(util.generateGeometry(eq.geometries)))
        return wf

    @property
    def isNonlinear(self):
        return True 