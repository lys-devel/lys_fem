import numpy as np
from ngsolve import Parameter, BilinearForm, LinearForm, CoefficientFunction

from . import util


class _Operator:
    def __init__(self, wf, fes, nonlinear):
        self._blf, self._lf = BilinearForm(fes), LinearForm(fes)
        self._blf += wf.lhs.eval()
        self._lf += wf.rhs.eval()
        self._fes = fes
        self._nl = nonlinear
        self._init  = False

    def __call__(self, x):
        self._lf.Assemble()
        if self._nl:
            return self._blf.Apply(x) + self._lf.vec
        else:
            return self._blf.mat * x  + self._lf.vec

    def Jacobian(self, x):
        if self._nl:
            self._blf.AssembleLinearization(x)
            return self._blf.mat.Inverse(self._fes.FreeDofs(), "pardiso")
        else:
            return self._inv
        
    def update(self):
        if self._nl:
            return
        if not self._init:
            self._blf.Assemble()
            self._inv = self._blf.mat.Inverse(self._fes.FreeDofs(), "pardiso")
            self._init = True
    
    @property
    def isNonlinear(self):
        return self._nl


class _Solution:
    def __init__(self, model, use_a, nlog=2):
        self._model = model
        fes = model.finiteElementSpace
        self._sols = [(util.GridFunction(fes), util.GridFunction(fes), util.GridFunction(fes)) for n in range(nlog)]
        self.update(model.initialValue(use_a))

    def update(self, xva):
        # Store old data
        for j in range(3):
            for n in range(1, len(self._sols)):
                self._sols[-n][j].vec.data = self._sols[-n+1][j].vec
        # Set new one
        for xi, yi in zip(self._sols[0], xva):
            if yi is not None:
                xi.vec.data = yi.vec

    def __getitem__(self, n):
        return self._sols[n]

    def copy(self):
        g = util.GridFunction(self._model.finiteElementSpace)
        g.vec.data = self._sols[0][0].vec
        return g

    def X(self, n=0):
        return self.__toFunc(self[n][0])

    def V(self, n=0):
        return self.__toFunc(self[n][1], "t")

    def A(self, n=0):
        return self.__toFunc(self[n][2], "tt")

    def __toFunc(self, x, pre=""):
        res = []
        n = 0
        for v in self._model.variables:
            if v.size == 1 and v.isScalar:
                res.append(util.NGSFunction(x.components[n], v.name+pre+"0"))
            else:
                res.append(util.NGSFunction(CoefficientFunction(tuple(x.components[n:n+v.size])), v.name+pre+"0"))
            n+=v.size           
        return res


class NGSTimeIntegrator:
    def __init__(self, model):
        self._model = model
        self._dti = Parameter(-1)

        self._sols = _Solution(model, self.use_a)
        self._x = self._sols.copy()

        wf = model.weakforms()
        wf = self.generateWeakforms(wf, model, self._sols, util.NGSFunction(self._dti,"dti"))
        self._op = _Operator(wf, model.finiteElementSpace, model.isNonlinear)

    def solve(self, solver, dti=0):
        if self._dti.Get() != dti:
            self._dti.Set(dti)
            self._op.update()
        with np.errstate(divide='ignore', invalid="ignore"):
            self._x.vec.data = solver.solve(self._op, self._x.vec.CreateVector(copy=True))
        self._sols.update(self.updateSolutions(self._x, self._sols, self._dti.Get()))
   
    @property
    def solution(self):
        return self._sols.copy()
    
    @property
    def use_a(self):
        return True


class BackwardEuler(NGSTimeIntegrator): 
    def generateWeakforms(self, wf, model, sols, dti):
        # Replace time derivative
        d = {}
        for v, x0, v0 in zip(model.variables, sols.X(), sols.V()):
            d[v.trial.t] = (v.trial - x0)*dti
            d[v.trial.tt] = (v.trial - x0)*dti*dti - v0*dti
        wf.replace(d)
        return wf
        
    def updateSolutions(self, x, sols, dti):
        X0 = sols[0][0]
        v = util.GridFunction(self._model.finiteElementSpace)
        v.vec.data = dti*(x.vec - X0.vec)
        return (x, v, None)

    @property
    def use_a(self):
        return False

class NewmarkBeta(NGSTimeIntegrator):
    def __init__(self, model, params="tapezoidal"):
        if params == "tapezoidal":
            self._params = [1/4, 1/2]
        else:
            self._params = params
        super().__init__(model)
       
    def generateWeakforms(self, wf, model, sols, dti):
        b, g = self._params
        d = {}
        for v, x0, v0, a0 in zip(model.variables, sols.X(), sols.V(), sols.A()):
            d[v.trial.t] = (v.trial - x0)*util.coef(g/b)*dti + v0*util.coef(1-g/b) + a0*util.coef(1-0.5*g/b)/dti
            d[v.trial.tt] = (v.trial - x0)*util.coef(1/b)*dti*dti - v0*util.coef(1/b)*dti + a0*util.coef(1-0.5/b)
        wf.replace(d)
        return wf
        
    def updateSolutions(self, x, sols, dti):
        X0, V0, A0 = sols[0]
        X0, V0, A0 = X0.vec, V0.vec, A0.vec

        b, g = self._params
        a = util.GridFunction(self._model.finiteElementSpace)
        v = util.GridFunction(self._model.finiteElementSpace)

        a.vec.data = (1/b*dti**2)*(x.vec-X0) - (dti/b)*V0 + (1-0.5/b)*A0
        v.vec.data = V0 + (1-g)/dti*A0 + g/dti*a.vec
        return (x, v, a)
    

class GeneralizedAlpha(NewmarkBeta):
    def __init__(self, model, rho="tapezoidal"):
        am = (2*rho-1)/(rho+1)
        af = rho/(rho+1)
        beta = 0.25*(1+af-am)**2
        gamma = 0.5+af-am
        self._params = [am, af, beta, gamma]
        super().__init__(model)
      
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

