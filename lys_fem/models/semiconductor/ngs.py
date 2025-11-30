from lys_fem.ngs import NGSModel, grad, dx
from . import DirichletBoundary, InitialCondition


class NGSSemiconductorModel(NGSModel):
    def __init__(self, model):
        super().__init__(model)
        self._model = model

    def weakform(self, vars, mat):
        q, kB = 1.602176634e-19, 1.3806488e-23
        mu_n, mu_p, Nd, Na = mat["mu_n"], mat["mu_p"] , mat["N_d"], mat["N_a"]

        wf = 0
        n, test_n = vars[self._model.variableName+"_e"]
        p, test_p = vars[self._model.variableName+"_h"]
        phi, test_phi = vars[self._model.phi]
        if self._model.T is None:
            T = mat["T"]
        else:
            T = vars[self._model.T][0]
        D_n, D_p = mu_n*kB*T/q, mu_p*kB*T/q

        # lhs, drift current, diffusion current terms
        wf += (n.t.dot(test_n) + p.t.dot(test_p))*dx
        wf += grad(phi).dot(-n*mu_n*grad(test_n) + p*mu_p*grad(test_p))*dx
        wf += D_n*grad(n).dot(grad(test_n))*dx + D_p*grad(p).dot(grad(test_p))*dx

        wf -= q*(p-n+Nd-Na)*test_phi * dx 

        return wf
