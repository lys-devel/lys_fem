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


class NGSTimeIntegrator:
    def __init__(self, model):
        self._model = model

        M,C,K,F = model.weakforms
        M = _BLF(M, self.isNonlinear)
        C = _BLF(C, self.isNonlinear)
        K = _BLF(K, self.isNonlinear)

        self._wfs = [M, C, K, F]
        self._x = model.getSolution()

    def solve(self, solver, dti=0):
        self._dti = dti
        self._x1 = self._model.getSolution()
        solver.solve(self, self._x)
        self._model.setSolution(self._x)

    def update(self, x):
        M,C,K,F = self.weakforms
        if self.isNonlinear:
            M.update(x)
            C.update(x)
            K.update(x)
        F.Assemble()

    @property
    def weakforms(self):
        return self._wfs

    @property
    def isNonlinear(self):
        return self._model.isNonlinear


class BackwardEuler(NGSTimeIntegrator):
    def __call__(self, x):
        self.update(x)
        M,C,K,F = self.weakforms
        dti = self._dti
        x0 = self._x1.vec
        return (C*dti).apply(x- x0) + K.apply(x) - F.vec 

    def Jacobian(self, x):
        M,C,K,F = self.weakforms
        dti = self._dti
        return (C*dti + K).inverse(self._model.finiteElementSpace.FreeDofs())
    