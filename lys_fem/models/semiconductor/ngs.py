from lys_fem.ngs import NGSModel, util, grad, dx


class NGSSemiconductorModel(NGSModel):
    def __init__(self, model, mesh, order=2):
        super().__init__(model, mesh)
        self._model = model

        init = self._model.initialConditions.coef(self._model.initialConditionTypes[0])
        initialValue = util.generateCoefficient(init, mesh)
        dirichlet = util.generateDirichletCondition(self._model)

        for eq in model.equations:
            self.addVariable(eq.variableName+"_e", 1, [dirichlet[0]], initialValue[0], region = eq.geometries, order=order, scale=init.scale)
            self.addVariable(eq.variableName+"_h", 1, [dirichlet[1]], initialValue[1], region = eq.geometries, order=order, scale=init.scale)

    def weakform(self, vars, mat):
        mu_n, mu_p, q, kB, Nd, Na = mat["mu_e"], mat["mu_h"], mat["q"], mat["k_B"], mat["N_d"], mat["N_a"]

        wf = 0
        for eq in self._model.equations:
            n, test_n = vars[eq.variableName+"_e"]
            p, test_p = vars[eq.variableName+"_h"]
            phi, test_phi = vars[eq.potName]
            if eq.tempName is None:
                T = mat["T"]
            else:
                T, _ = vars[eq.tempName][0].value
            D_n, D_p = mu_n*kB*T/q, mu_p*kB*T/q

            # lhs, drift current, diffusion current terms
            wf += (n.t.dot(test_n) + p.t.dot(test_p))*dx
            wf += (-n.value*mu_n*grad(test_n) + p.value*mu_p*grad(test_p)).dot(grad(phi))*dx
            wf += D_n*grad(n).dot(grad(test_n))*dx + D_p*grad(p).dot(grad(test_p))*dx

            wf -= q*(p-n+Nd-Na)*test_phi * dx 

        return wf   
