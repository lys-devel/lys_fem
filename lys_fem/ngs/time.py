from . import util

class ForwardEuler:
    @staticmethod
    def generateWeakforms(v, trial, sols, dti):
        d = {}
        x0, v0, g0 = sols.X(v), sols.V(v), sols.grad(v)
        d[trial.t] = (trial - x0)*dti
        d[trial.tt] = (trial - x0)*dti**2 - v0*dti
        d[trial] = x0
        d[trial.value] = x0
        d[util.grad(trial)] = g0
        return d
    

class BackwardEuler: 
    @staticmethod
    def generateWeakforms(v, trial, sols, dti):
        d = {}
        x0, v0 = sols.X(v), sols.V(v)
        d[trial.t] = (trial - x0)*dti
        d[trial.tt] = (trial - x0)*dti**2 - v0*dti
        return d


class BDF2:
    @staticmethod
    def generateWeakforms(v, trial, sols, dti):
        n = util.min(util.stepn, 1)
        x0, x1 = sols.X(v), sols.X(v,1)
        d = {trial.t: ((2+n)*trial-(2+2*n)*x0+n*x1)/2*dti}
        return d


class NewmarkBeta:       
    @staticmethod
    def generateWeakforms(v, trial, sols, dti):
        b, g = [1/4, 1/2]
        d = {}
        x0, v0, a0 = sols.X(v), sols.V(v), sols.A(v)
        d[trial.t] = (trial - x0)*(g/b)*dti + v0*(1-g/b) + a0*(1-0.5*g/b)/dti
        d[trial.tt] = (trial - x0)*(1/b)*dti*dti - v0*(1/b)*dti + a0*(1-0.5/b)
        return d
    

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

