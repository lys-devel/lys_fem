import time
import numpy as np

import ngsolve
import ngsolve.ngs2petsc as n2p
import petsc4py.PETSc as psc

from .space import CompressedFESpace


class Solver:
    def __init__(self, fes, wf, linear, nonlinear={}, parallel=False):
        self._fes = fes
        self._linear = linear
        self._nonlinear = nonlinear

        self._blf = fes.BilinearForm(wf.lhs, condense = linear.get("condense",False), symmetric = linear.get("symmetric",False))
        self._lf = fes.LinearForm(wf.rhs)
        self._inv = LinearSolver(fes, self._blf, parallel=parallel, **linear)
        self._nl = NonlinearSolver(**nonlinear)

    def solve(self, x):
        msg = self.__update()
        xi = self._fes.gridFunction()
        self.__syncGridFunction(x, "->", xi)
        xi.vec.data = self._nl.newton(self, xi.vec.CreateVector(copy=True))
        self.__syncGridFunction(x, "<-", xi)
        msg += self._nl.msg
        return msg

    def __update(self):
        self._lf.update()
        start = time.time()
        if self._blf.update():
            self._inv.update()
            return "\t[Assemble] Time = {:.2f}\n".format(time.time()-start)
        return ""

    def __syncGridFunction(self, glb, direction, loc):
        if not isinstance(self._fes, CompressedFESpace):
            if direction == "->":
                loc.vec.data = glb.vec
            elif direction == "<-":
                glb.vec.data = loc.vec
        else:
            if direction == "->":
                loc.vec.FV().NumPy()[:] = glb.vec.FV().NumPy()[self._fes.mask]
            elif direction == "<-":
                glb.vec.FV().NumPy()[self._fes.mask] = loc.vec.FV().NumPy()

    def __call__(self, x):
        return self._blf * x + self._lf

    def Jacobian(self, x):
        if self._blf.linearize(x):
            self._inv.update()
        return self._inv

    @property
    def isNonlinear(self):
        return self._blf.isNonlinear

    def __str__(self):
        res = ""
        res += "\t\tTotal degree of freedoms: " + str(self._fes.ndof) + "\n"
        res += "\t\tNonlinear: " + str(self.isNonlinear) + "\n"
        res += "\t\tTime dependent: LHS = " + str(self._blf.isTimeDependent) + ", RHS = " + str(self._lf.isTimeDependent) + "\n" 
        if isinstance(self._fes, CompressedFESpace):
            res += "\t\tSymbols: " + str(self._fes.symbols) + "\n"
        res += "\t\tLinear Solver: " + self._linear.get("solver") + "\n"
        res += "\t\tStatic condensation: " + str(self._linear.get("condense", False)) + "\n"
        res += "\t\tSymmetric: " + str(self._linear.get("symmetric", False)) + "\n"
        if "prec" in self._linear:
            res += "\t\tPreconditioner: " + str(self._linear.get("prec")) + "\n"
            res += "\t\tMax iteration:" + str(self._linear.get("iter")) + "\n"
            res += "\t\tRelative tolerance:" + str(self._linear.get("rtol")) + "\n"
        return res


class NonlinearSolver:
    def __init__(self, eps=1e-5, max_iter=30, gamma=1):
        self._eps = eps
        self._iter = max_iter
        self._gamma = gamma

    def newton(self, F, x):
        self.msg = ""
        max_iter = 1 if not F.isNonlinear else self._iter
        dx = x.CreateVector()
        for i in range(max_iter):
            J = F.Jacobian(x)
            dx.data = J*F(x)
            dx.data *= self._gamma
            x -= dx
            R = np.sqrt(np.divide(dx.InnerProduct(dx), x.InnerProduct(x)))
            self.msg += "\t"+J.msg + "\n"+"\tResidual R = {:.2e}\n".format(R)
            if R < self._eps:
                if i!=0:
                    self.msg += "\t[Newton solver] Converged in " + str(i) + " steps.\n"
                return x
        if max_iter !=1:
            raise ConvergenceError("[Newton solver] NOT Converged in " + str(i) + " steps.")
        return x


class LinearSolver:
    def __init__(self, fes, blf, solver, prec=None, iter=None, rtol=None, parallel=False, **kwargs):
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
            raise ConvergenceError("[Iterative solver] NOT converged. Condition = {:.2f}".format(ma/mi))
        return gfu.vec


class ConvergenceError(RuntimeError):
    pass
