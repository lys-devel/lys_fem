from lys_fem.ngs import NGSModel, grad, dx
from . import DirichletBoundary, InitialCondition


class NGSSemiconductorModel(NGSModel):
    def __init__(self, model, mesh, vars, order=2):
        super().__init__(model, mesh, vars)
        self._model = model

        init = self._model.initialConditions.coef(InitialCondition)
        dirichlet = self._model.boundaryConditions.coef(DirichletBoundary)
        if dirichlet is None:
            dirichlet = ["auto", "auto"]

        for eq in model.equations:
            self.addVariable(eq.variableName+"_e", 1, dirichlet[0], init[0], region = eq.geometries, order=order, isScalar=True)
            self.addVariable(eq.variableName+"_h", 1, dirichlet[1], init[1], region = eq.geometries, order=order, isScalar=True)

    def weakform(self, vars, mat):
        q, kB = 1.602176634e-19, 1.3806488e-23
        mu_n, mu_p, Nd, Na = mat["mu_n"], mat["mu_p"] , mat["N_d"], mat["N_a"]

        wf = 0
        for eq in self._model.equations:
            n, test_n = vars[eq.variableName+"_e"]
            p, test_p = vars[eq.variableName+"_h"]
            phi, test_phi = vars[eq.potName]
            if eq.tempName is None:
                T = mat["T"]
            else:
                T = vars[eq.tempName][0].value
            D_n, D_p = mu_n*kB*T/q, mu_p*kB*T/q

            # lhs, drift current, diffusion current terms
            wf += (n.t.dot(test_n) + p.t.dot(test_p))*dx
            wf += grad(phi).dot(-n.value*mu_n*grad(test_n) + p.value*mu_p*grad(test_p))*dx
            wf += D_n*grad(n).dot(grad(test_n))*dx + D_p*grad(p).dot(grad(test_p))*dx

            wf -= q*(p-n+Nd-Na)*test_phi * dx 

        return wf
