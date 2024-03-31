import numpy as np
from scipy import sparse
from ngsolve.la import SparseMatrixd


class _BLFBase:
    def __add__(self, v):
        return _Binary(self, v, "+")
    
    def __radd__(self, v):
        return self + v
    
    def __mul__(self, v):
        return _Binary(self, v, "*")
    
    def __rmul__(self, v):
        return self*v

    def inverse(self, dofs):
        m = self.mat.tocoo()
        m = SparseMatrixd.CreateFromCOO(m.row, m.col, m.data, *m.shape)
        return m.Inverse(dofs, "pardiso")


class _BLF(_BLFBase):
    def __init__(self, blf, nonlinear):
        if not nonlinear:
            blf.Assemble()
        self._blf = blf
        self._nl = nonlinear

    def apply(self, x):
        if self._nl:
            return self._blf.Apply(x)
        else:
            return self._blf.mat*x
        
    def update(self, x):
        if self._nl:
            self._blf.AssembleLinearization(x)

    @property
    def mat(self):
        row, col, val = self._blf.mat.COO()
        return sparse.csr_matrix((val,(row,col)))


class _Binary(_BLFBase):
    def __init__(self, b1, b2, type):
        self._b1 = b1
        self._b2 = b2
        self._type = type

    def apply(self, x):
        if self._type == "*":
            return self._b2 * self._b1.apply(x)
        elif self._type == "+":
            return self._b1.apply(x) + self._b2.apply(x)
        
    @property
    def mat(self):
        m1 = self._b1.mat 
        if self._type == "*":
            m = m1 * self._b2
        if self._type == "+":
            m = m1 + self._b2.mat
        return m


class _Solution(list):
    def __init__(self, item=None, nlog=2):
        super().__init__(item)
        self._nlog = nlog

    def append(self, item):
        super().append(item)
        if len(self) > self._nlog:
            self.pop(0)


class NGSTimeIntegrator:
    def __init__(self, model, calc_xtt=True):
        self._model = model

        M,C,K,F = model.weakforms()
        M = _BLF(M, self.isNonlinear)
        C = _BLF(C, self.isNonlinear)
        K = _BLF(K, self.isNonlinear)
        self._wfs = [M, C, K, F]

        x,v,a = model.initialValue, model.initialVelocity, None
        if calc_xtt:
            a = self.__calc_xtt(x,v)
        self._sols = _Solution([(x.vec,v.vec,a)])
        self._x = x

    def __calc_xtt(self, x, v):
        self.update(x.vec)
        M, C, K, F = self.weakforms
        rhs = F.vec - C.apply(v.vec) - K.apply(x.vec)
        a = M.inverse(self._model.finiteElementSpace.FreeDofs()) * rhs
        return a
    
    def solve(self, solver, dti=0):
        self._dti = dti
        self._x.vec.data = solver.solve(self, self._x.vec.CreateVector(copy=True))
        self._sols.append(self.update_xva(self._x.vec))

    def update(self, x):
        M,C,K,F = self.weakforms
        if self.isNonlinear:
            M.update(x)
            C.update(x)
            K.update(x)
        F.Assemble()

    @property
    def solution(self):
        result = self._model.materialSolution
        vs = self._model.variables

        comps = [self._x] if len(vs) == 1 else self._x.components
        for v, xv in zip(vs, comps):
            if len(self._x.shape) == 0:
                result[v.name] = np.array(xv.vec) * v.scale 
            else:
                result[v.name] = [np.array(xi.vec) * v.scale for xi in xv.components]
        return result

    @property
    def weakforms(self):
        return self._wfs

    @property
    def isNonlinear(self):
        return self._model.isNonlinear


class BackwardEuler(NGSTimeIntegrator):
    def __init__(self, model):
        super().__init__(model, calc_xtt=False)

    def __call__(self, x):
        self.update(x)
        M,C,K,F = self.weakforms
        dti = self._dti
        x0, v0, a0 = self._sols[-1]
        return (C*dti).apply(x- x0) + K.apply(x) - F.vec 

    def Jacobian(self, x):
        self.update(x)
        M,C,K,F = self.weakforms
        dti = self._dti
        return (C*dti + K).inverse(self._model.finiteElementSpace.FreeDofs())
    
    def update_xva(self, x):
        return x, None, None