import numpy as np
from ngsolve import Parameter, GridFunction, BilinearForm, LinearForm


class _Operator:
    def __init__(self, blf, lf, fes, nonlinear):
        self._blf = blf
        self._lf = lf
        self._fes = fes
        self._nl = nonlinear
        self._init  = False

    def __call__(self, x):
        self._lf.Assemble()
        if self._nl:
            return self._blf.Apply(x) - self._lf.vec
        else:
            return self._blf.mat * x  - self._lf.vec

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
    def __init__(self, model, nlog=2):
        self._model = model
        fes = model.finiteElementSpace
        self._sols = [(GridFunction(fes), GridFunction(fes), GridFunction(fes)) for n in range(nlog)]
        self._isSingle = len(model.variables) == 1

    def update(self, xva):
        for j in range(3):
            for n in range(1, len(self._sols)):
                self.__set(self._sols[-n][j], self._sols[-n+1][j])
        for xi, yi in zip(self._sols[0], xva):
            if yi is not None:
                self.__set(xi, yi)

    def __set(self,x,y):
        if self._isSingle:
            if not isinstance(y, GridFunction):
                y = y[0]
            x.Set(y)
        else:
            if isinstance(y, GridFunction):
                y = y.components
            for xi, yi in zip(x.components, y):
                xi.Set(yi)

    def __getitem__(self, n):
        return self._sols[n]

    def copy(self):
        g = GridFunction(self._model.finiteElementSpace)
        self.__set(g, self._sols[0][0])
        return g

    def X(self, n=0):
        x = self[n][0]
        return np.array([x] if self._isSingle else x.components)

    def V(self, n=0):
        x = self[n][1]
        return np.array([x] if self._isSingle else x.components)

    def A(self, n=0):
        x = self[n][2]
        return np.array([x] if self._isSingle else x.components)


class NGSTimeIntegrator:
    def __init__(self, model):
        self._model = model
        self._dti = Parameter(-1)

        self._sols = _Solution(model)
        x,v = model.initialValue, model.initialVelocity
        self._sols.update((x, v, self.initialAcceleration(model, x, v)))
        self._x = self._sols.copy()

        a,f = self.generateWeakforms(model, self._sols, self._dti)
        A,F = BilinearForm(self._model.finiteElementSpace), LinearForm(self._model.finiteElementSpace)
        if a != 0:
            A += a
        if f != 0:
            F += f
        self._op = _Operator(A, F, model.finiteElementSpace, model.isNonlinear)

    @property
    def trials(self):
        return np.array([self._model.finiteElementSpace.TnT()[0]] if len(self._model.variables) == 1 else self._model.finiteElementSpace.TnT()[0])
    
    def solve(self, solver, dti=0):
        if self._dti.Get() != dti:
            self._dti.Set(dti)
            self._op.update()
        self._x.vec.data = solver.solve(self._op, self._x.vec.CreateVector(copy=True))
        self._sols.update(self.updateSolutions(self._x, self._sols, self._dti.Get()))
   
    @property
    def solution(self):
        return self._sols.copy()
        result = self._model.materialSolution
        vs = self._model.variables
        X = self._sols.copy()

        comps = [X] if len(vs) == 1 else X.components
        for v, xv in zip(vs, comps):
            xv.vec.data *= v.scale
            if len(X.shape) == 0:
                result[v.name] = xv 
            else:
                result[v.name] = xv
        return result


class BackwardEuler(NGSTimeIntegrator):
    def initialAcceleration(self, model, x, v):
        return None
    
    def generateWeakforms(self, model, sols, dti):
        X, X0, V0 = self.trials, sols.X(), sols.V()
        M1, C1, K1, F1 = model.weakforms(X, X*dti, X*dti**2)
        M2, C2, K2, F2 = model.weakforms(X, X0*dti, X0*dti**2+V0*dti)
        if model.isNonlinear:
           return M1 - M2 + C1 - C2 + K1, F1
        else:
            return M1 + C1 + K1, F1 + C2 + M2
        
    def updateSolutions(self, x, sols, dti):
        X0 = sols[0][0]
        v = GridFunction(self._model.finiteElementSpace)
        v.vec.data = dti*(x.vec - X0.vec)
        return (x, v, None)
    

class GeneralizedAlpha(NGSTimeIntegrator):
    def __init__(self, model, rho="tapezoidal"):
        if isinstance(rho, float):
            am = (2*rho-1)/(rho+1)
            af = rho/(rho+1)
            beta = 0.25*(1+af-am)**2
            gamma = 0.5+af-am
            self._params = [am, af, beta, gamma]
        elif rho=="tapezoidal":
            self._params = [0, 0, 1/4, 1/2]
        super().__init__(model)

    def initialAcceleration(self, model, x, v):
        fes = model.finiteElementSpace
        X = self.trials
        M, C, K, F = model.weakforms(X, X, X)
        Feff = LinearForm(fes)
        Feff += F
        Keff = BilinearForm(fes)
        Keff += K
        Ceff = BilinearForm(fes)
        Ceff += C
        Meff = BilinearForm(fes)
        Meff += M

        rhs = Feff.vec - Ceff.Apply(v.vec) - Keff.Apply(x.vec)
        Meff.AssembleLinearization(x.vec)

        a = GridFunction(fes)
        a.vec.data  = (Meff.mat.Inverse(model.finiteElementSpace.FreeDofs(), "pardiso") * rhs)
        return a
        
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
        
    def updateSolutions(self, x, sols, dti):
        X0, V0, A0 = sols[0]
        X0, V0, A0 = X0.vec, V0.vec, A0.vec

        am, af, b, g = self._params
        a = GridFunction(self._model.finiteElementSpace)
        v = GridFunction(self._model.finiteElementSpace)

        a.vec.data = (1/b*dti**2)*(x.vec-X0) - (dti/b)*V0 + (1-0.5/b)*A0
        v.vec.data = V0 + (1-g)/dti*A0 + g/dti*a.vec
        return (x, v, a)
    
