from ngsolve import H1, GridFunction, grad, dx, ds
from lys_fem.ngs import NGSModel, util, dti

class NGSLLGModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh)
        self._model = model
        self._mat = mat

        self._fes_lam = H1(mesh, order=1, dirichlet=[])
        self._sol_lam = GridFunction(self._fes_lam)
        self._sol_lam.Set(util.generateCoefficient(0))

    @property
    def bilinearform(self):
        [m,lam],[test_m, test_lam] =self.TnT()
        g = util.generateCoefficient(1.760859770e11)
        B = util.generateCoefficient([0,0,1])

        wf_m = m*test_m*dti + g*util.CrossProduct(m, B)*test_m + 2*lam*m * test_m
        wf_lam = (m*m-1)*test_lam
        return (wf_m + wf_lam) * dx
    
    @property
    def linearform(self):
        [m,lam],[test_m, test_lam] =self.TnT()
        wf = self.sol[0] * test_m * dti * dx
        return wf

    @property
    def spaces(self):
        return super().spaces + [self._fes_lam]

    @property
    def variableNames(self):
        return super().variableNames + ["lam_LLG"]

    @property
    def sol(self):
        return super().sol + [self._sol_lam]

    @property
    def isNonlinear(self):
        return True 