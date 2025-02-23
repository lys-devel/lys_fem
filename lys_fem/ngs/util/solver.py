import time

import ngsolve
import ngsolve.ngs2petsc as n2p
import petsc4py.PETSc as psc


class LinearSolver:
    def __init__(self, fes, blf, solver, prec=None, iter=None, rtol=None, parallel=False):
        self._fes = fes
        self._blf = blf
        self._solver = solver
        self._prec = prec
        self._iter = iter
        self._rtol = rtol
        self._par = parallel

        self._inv = None
        self._res = fes.gridFunction()

    def update(self):
        if self._solver in ["pardiso", "mumps", "umfpack", "pardisospd", "sparsecholesky", "masterinverse"]:
            self._inv = None
        else:
            if self._inv is None:
                self._inv = _petsc(self._fes, self._blf, self._solver, self._prec, iter=self._iter, tol=self._rtol, parallel=self._par)
            else:
                self._inv.setMatrix(self._blf.mat)

    def __mul__(self, other):
        start = time.time()
        if self._inv is None:
            self._inv = self._blf.mat.Inverse(self._fes.FreeDofs(self._blf.condense), self._solver)
        if self._blf.condense:
            ext = ngsolve.IdentityMatrix() + self._blf.harmonic_extension
            extT = ngsolve.IdentityMatrix() + self._blf.harmonic_extension_trans
            self._res.vec.data = ext * (self._inv * (extT * other)) + self._blf.inner_solve * other
        else:
            self._res.vec.data = self._inv * other
        if isinstance(self._inv, _petsc):
            self.msg = self._inv.msg
        else:
            self.msg = "[Direct solver] Time = {:.2f}".format(time.time()-start)
        return self._res.vec


class _petsc:
    def __init__(self, fes, blf, solver, prec, iter=10000, tol=1e-9, parallel=False):
        self._fes = fes
        self._blf = blf
        self._iter=iter
        self._tol = tol
        self._par = parallel
        self.__initialize(blf.mat, solver, prec)

    def __initialize(self, mat, solver, prec):
        self._dofs = self._fes.FreeDofs(self._blf.condense)
        self._psc_mat = n2p.CreatePETScMatrix(mat, self._dofs)
        if self._par:
            self._vecmap = n2p.VectorMapping(mat.row_pardofs, self._dofs)

        ksp = psc.KSP()
        ksp.create()
        ksp.setOperators(self._psc_mat)
        ksp.setType(solver)
        ksp.setComputeSingularValues(True)
        if prec == "gmres":
            ksp.setGMRESRestart(2000)
        ksp.getPC().setType(prec)
        ksp.setTolerances(rtol=self._tol, atol=0, divtol=1e16, max_it=self._iter)
        self._ksp = ksp

    def setMatrix(self, mat):
        self._psc_mat = n2p.CreatePETScMatrix(mat, self._dofs)
        self._ksp.setOperators(self._psc_mat)

    def __mul__(self, other):
        start = time.time()
        f, u = self._psc_mat.createVecs()
        gfu = self._fes.gridFunction()
        gfu.vec.data = other
        if self._par:
            self._vecmap.N2P(gfu.vec, f)
        else:
            f.getArray()[:] = gfu.vec.FV().NumPy()[self._dofs]
        self._ksp.solve(f, u)
        gfu.vec.FV().NumPy()[:] = 0
        if self._par:
            self._vecmap.P2N(u, gfu.vec)
        else:
            gfu.vec.FV().NumPy()[self._dofs] = u.getArray()
        ma, mi = self._ksp.computeExtremeSingularValues()
        self.msg = "[Iterative solver] Iter = "+ str(self._ksp.its)+", Condition = {:.2f}".format(ma/mi) + ", Time = {:.2f}".format(time.time()-start)
        if self._ksp.its == self._iter:
            raise NGSConvergenceError("[Iterative solver] NOT converged.")
        return gfu.vec


class NGSConvergenceError(RuntimeError):
    pass
