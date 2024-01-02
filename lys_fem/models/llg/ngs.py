from ngsolve import grad, dx, ds
from lys_fem.ngs import NGSModel, util, dti
from . import ExternalMagneticField

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

        wf = 0
        for eq in self._model.equations:
            m, test_m = tnt[eq.variableName]
            lam, test_lam = tnt[eq.variableName+"_lam"]
            wf += m*test_m*dti*dx 
            wf += (2*lam*m * test_m + (m*m-1)*test_lam)*dx

            if self._model.domainConditions.have(ExternalMagneticField):
                B = util.generateGeometryCoefficient(self._mesh, self._model.domainConditions.get(ExternalMagneticField))
                wf += g*util.CrossProduct(m, B)*test_m*dx(definedon=util.generateGeometry(eq.geometries))

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