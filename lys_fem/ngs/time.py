from . import util

class BackwardEuler: 
    def generateWeakforms(self, wf, model, sols, dti):
        # Replace time derivative
        d = {}
        for v, x0, v0 in zip(model.variables, sols.X(), sols.V()):
            d[v.trial.t] = (v.trial - x0)*dti
            d[v.trial.tt] = (v.trial - x0)*dti**2 - v0*dti
        wf = wf.replace(d)
        return wf
        
    def updateSolutions(self, x, sols, dti):
        X0 = sols[0][0]
        v = util.GridFunction(sols.finiteElementSpace)
        v.vec.data = dti*(x.vec - X0.vec)
        return (x, v, None)

    @property
    def use_a(self):
        return False


class BDF2:
    n = util.Parameter("n", 0)

    def generateWeakforms(self, wf, model, sols, dti):
        # Replace time derivative
        d = {}
        n = util.min(util.stepn, 0)
        for v, x0, x1 in zip(model.variables, sols.X(), sols.X(1)):
            d[v.trial.t] = ((2+n)*v.trial-(2+2*n)*x0+n*x1)/2*dti
        wf = wf.replace(d)
        return wf

    def updateSolutions(self, x, sols, dti):
        X0 = sols[0][0]
        X1 = sols[1][0]
        v = util.GridFunction(sols.finiteElementSpace)
        if util.stepn.get() > 1:
            v.vec.data = dti*3/2*x.vec - dti*4/2*X0.vec + dti*0.5*X1.vec
        else:
            v.vec.data = dti*(x.vec - X0.vec)
        return (x, v, None)

    @property
    def use_a(self):
        return False


class NewmarkBeta:
    def __init__(self, params="tapezoidal"):
        if params == "tapezoidal":
            self._params = [1/4, 1/2]
        else:
            self._params = params
       
    def generateWeakforms(self, wf, model, sols, dti):
        b, g = self._params
        d = {}
        for v, x0, v0, a0 in zip(model.variables, sols.X(), sols.V(), sols.A()):
            d[v.trial.t] = (v.trial - x0)*(g/b)*dti + v0*(1-g/b) + a0*(1-0.5*g/b)/dti
            d[v.trial.tt] = (v.trial - x0)*(1/b)*dti*dti - v0*(1/b)*dti + a0*(1-0.5/b)
        wf = wf.replace(d)
        return wf
        
    def updateSolutions(self, x, sols, dti):
        X0, V0, A0 = sols[0]
        X0, V0, A0 = X0.vec, V0.vec, A0.vec

        b, g = self._params
        a = util.GridFunction(sols.finiteElementSpace)
        v = util.GridFunction(sols.finiteElementSpace)

        a.vec.data = (1/b*dti**2)*(x.vec-X0) - (dti/b)*V0 + (1-0.5/b)*A0
        v.vec.data = V0 + (1-g)/dti*A0 + g/dti*a.vec
        return (x, v, a)
    
    @property
    def use_a(self):
        return True
    

class GeneralizedAlpha(NewmarkBeta):
    def __init__(self, rho="tapezoidal"):
        am = (2*rho-1)/(rho+1)
        af = rho/(rho+1)
        beta = 0.25*(1+af-am)**2
        gamma = 0.5+af-am
        self._params = [am, af, beta, gamma]
        super().__init__()
      
    def generateWeakforms(self, model, sols, dti):
        X, X0, V0, A0 = self.trials, sols.X(), sols.V(), sols.A()
        am, af, b, g = self._params
        if model.isNonlinear:
           M, C, K, F = model.weakforms(X, (X-X0)*dti, X)
           return M + C + K, F
        else:
            M1, C1, K1, F1 = model.weakforms(X, X*dti, X*dti**2)
            v = (X0*(g/b)*dti + (g/b-1)*V0 + (0.5*g/b-1)*A0/dti)*(1-af) + af*V0
            a = (X0/b*dti**2 + V0*dti/b + (0.5/b-1)*A0)*(1-am) + am * A0
            M2, C2, K2, F2 = model.weakforms(X0, v, a)
            return (1-am)/b*M1 + (1-af)*(g/b)*C1 + (1-af)*K1, F1 + M2 + C2 - af*K2

