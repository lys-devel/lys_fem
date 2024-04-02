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

    def __set(self, x,y):
        if self._isSingle:
            x.Set(y)
        else:
            for xi, yi in zip(x.components, y.components):
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
        return np.array([x] if self._isSingle == 1 else x.components)

    def A(self, n=0):
        x = self[n][2]
        return np.array([x] if self._isSingle == 1 else x.components)


class NGSTimeIntegrator:
    def __init__(self, model):
        self._model = model
        self._dti = Parameter(-1)

        self._sols = _Solution(model)
        self._sols.update((model.initialValue, model.initialVelocity, None))
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
        self._sols.update(self.updateSolutions(self._x, self._sols, self._dti))
   
    @property
    def solution(self):
        result = self._model.materialSolution
        vs = self._model.variables
        x, v, a = self._sols[0]

        comps = [x] if len(vs) == 1 else x.components
        for v, xv in zip(vs, comps):
            if len(x.shape) == 0:
                result[v.name] = np.array(xv.vec) * v.scale 
            else:
                result[v.name] = [np.array(xi.vec) * v.scale for xi in xv.components]
        return result


class BackwardEuler(NGSTimeIntegrator):
    def generateWeakforms(self, model, sols, dti):
        X, X0 = self.trials, sols.X()
        if model.isNonlinear:
           M, C, K, F = model.weakforms(X, (X-X0)*dti, X)
           return M+C+K, F
        else:
            M1, C1, K1, F1 = model.weakforms(X, X*dti, X)
            M2, C2, K2, F2 = model.weakforms(X, X0*dti, X)
            return C1 + K1, F1 + C2
        
    def updateSolutions(self, x, sols, dti):
        return (x, None, None)
    
    def __calc_xtt(self, x, v):
        self.update(x.vec)
        M, C, K, F = self.weakforms
        rhs = F.vec - C.apply(v.vec) - K.apply(x.vec)
        a = M.inverse(self._model.finiteElementSpace.FreeDofs()) * rhs
        return a