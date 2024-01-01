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

    @property
    def bilinearform(self):
        TnT_list = self.TnT()
        g = util.generateCoefficient(1.760859770e11)

        wf = 0
        for i, eq in enumerate(self._model.equations):
            [m,lam],[test_m, test_lam] = TnT_list[0][i:i+2], TnT_list[1][i:i+2]
            wf += m*test_m*dti*dx 
            wf += (2*lam*m * test_m + (m*m-1)*test_lam)*dx

            if self._model.domainConditions.have(ExternalMagneticField):
                B = util.generateGeometryCoefficient(self._mesh, self._model.domainConditions.get(ExternalMagneticField))
                wf += g*util.CrossProduct(m, B)*test_m*dx(definedon=util.generateGeometry(eq.geometries))

        return wf
    
    @property
    def linearform(self):
        TnT_list = self.TnT()
        wf = 0
        for i, eq in enumerate(self._model.equations):
            [m,lam],[test_m, test_lam] = TnT_list[0][i:i+2], TnT_list[1][i:i+2]
            wf += self.sol[2*i] * test_m * dti * dx
        return wf

    @property
    def isNonlinear(self):
        return True 