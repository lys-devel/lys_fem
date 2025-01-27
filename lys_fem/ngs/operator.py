import time
import numpy as np

import ngsolve
import ngsolve.ngs2petsc as n2p
import petsc4py.PETSc as psc

from . import mpi, util

class Operator:
    def __init__(self, mesh, model, sols, step, type="discretized"):
        self._fes = util.FiniteElementSpace(model.variables, mesh, step.variables, symmetric=step.symmetric, condense=step.condensation)
        self._step = step

        wf = model.weakforms(type=type, sols=sols, symbols=step.variables)
        self._lhs, self._rhs = wf.lhs, wf.rhs
        self.__form()

        self._nl = self._lhs.isNonlinear
        self._tdep_lhs = self._lhs.isTimeDependent
        self._tdep_rhs = self._rhs.isTimeDependent
        self._init  = False
        self._psc = None

    def __form(self):
        self._blf = self._fes.BilinearForm()
        if self._lhs.valid:
            self._blf += self._lhs.eval(self._fes)
        self._lf = self._fes.LinearForm()
        if self._rhs.valid:
            self._lf += self._rhs.eval(self._fes)
  
    def __call__(self, x):
        if self.isNonlinear or self._blf.condense:
            return self._blf.Apply(x) + self._lf.vec
        else:
            return self._blf.mat * x  + self._lf.vec

    def Jacobian(self, x):
        if self.isNonlinear:
            self._blf.AssembleLinearization(x)
            self._inv = self.__inverse(self._blf.mat)
        return self._inv

    @property
    def isNonlinear(self):
        return self._nl

    def solve(self, x):
        self.__update()
        xi = self._fes.gridFunction()
        self.__syncGridFunction(x, "->", xi)
        xi.vec.data = newton(self, xi.vec.CreateVector(copy=True), eps=self._step.newton_eps, max_iter=self._step.newton_maxiter, gamma=self._step.newton_damping)
        self.__syncGridFunction(x, "<-", xi)

    def __update(self):
        if (self._tdep_rhs or not self._init):
            self._lf.Assemble()
        if (self._tdep_lhs or not self._init) and not self.isNonlinear:
            start = time.time()
            self._blf.Assemble()
            self._inv = self.__inverse(self._blf.mat)
            mpi.print_("\t[Assemble] Time = {:.2f}".format(time.time()-start))
        self._init = True

    def __inverse(self, mat):
        if self._step.solver in ["pardiso", "mumps", "umfpack", "pardisospd", "sparsecholesky", "masterinverse"]:
            start = time.time()
            inv = mat.Inverse(self._fes.FreeDofs(), self._step.solver)
            mpi.print_("\t[Direct solver] Time = {:.2f}".format(time.time()-start))
        else:
            if self._psc is None:
                self._psc = _petsc(self._fes, mat, self._step.solver, self._step.preconditioner, iter=self._step.linear_maxiter, tol=self._step.linear_rtol)
            else:
                self._psc.setMatrix(mat)
            inv = self._psc
        return _Inv(self._fes, inv, self._blf)
    
    def __syncGridFunction(self, glb, direction, loc):
        if self._step.variables is None:
            if direction == "->":
                loc.vec.data = glb.vec
            elif direction == "<-":
                glb.vec.data = loc.vec
        else:
            if direction == "->":
                loc.vec.FV().NumPy()[:] = glb.vec.FV().NumPy()[self._fes.mask]
            elif direction == "<-":
                glb.vec.FV().NumPy()[self._fes.mask] = loc.vec.FV().NumPy()

    def __str__(self):
        res = ""
        res += "\t\tTotal (Free) degree of freedoms: " + str(self._fes.ndof) + " ("+ str(sum(self._fes.FreeDofs())) + ")" + "\n"
        res += "\t\tNonlinear: " + str(self.isNonlinear) + "\n"
        res += "\t\tTime dependent: LHS = " + str(self._tdep_lhs) + ", RHS = " + str(self._tdep_rhs) + "\n" 
        res += "\t\tStatic condensation: " + str(self._step.condensation) + "\n"
        if self._step.variables is not None:
            res += "\t\tSymbols: " + str(self._step.variables) + "\n"
        res += "\t\tSolver: " + str(self._step.solver) + "\n"
        if self._step.preconditioner is not None:
            res += "\t\tPreconditioner: " + str(self._step.preconditioner) + "\n"
            res += "\t\tMax iteration:" + str(self._step.linear_maxiter) + "\n"
            res += "\t\tRelative tolerance:" + str(self._step.linear_rtol) + "\n"
        return res


class _petsc:
    def __init__(self, fes, mat, solver, prec, iter=10000, tol=1e-9):
        self._fes = fes
        self._iter=iter
        self._tol = tol
        self.__initialize(mat, solver, prec)

    def __initialize(self, mat, solver, prec):
        self._dofs = self._fes.FreeDofs()
        self._psc_mat = n2p.CreatePETScMatrix(mat, self._dofs)
        if mpi.isParallel():
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
        if mpi.isParallel():
            self._vecmap.N2P(gfu.vec, f)
        else:
            f.getArray()[:] = gfu.vec.FV().NumPy()[self._dofs]
        self._ksp.solve(f, u)
        gfu.vec.FV().NumPy()[:] = 0
        if mpi.isParallel():
            self._vecmap.P2N(u, gfu.vec)
        else:
            gfu.vec.FV().NumPy()[self._dofs] = u.getArray()
        ma, mi = self._ksp.computeExtremeSingularValues()
        mpi.print_("\t[Iterative solver] Iter =", self._ksp.its, ", Condition = {:.2f}".format(ma/mi), ", Time = {:.2f}".format(time.time()-start))
        if self._ksp.its == self._iter:
            raise ConvergenceError("[Iterative solver] NOT converged.")
        return gfu.vec


class _Inv:
    def __init__(self, fes, inv, blf):
        self._inv = inv
        self._blf = blf
        self._res = fes.gridFunction()

    def __mul__(self, other):
        start = time.time()
        if self._blf.condense:
            ext = ngsolve.IdentityMatrix() + self._blf.harmonic_extension
            extT = ngsolve.IdentityMatrix() + self._blf.harmonic_extension_trans
            self._res.vec.data = ext * (self._inv * (extT * other)) + self._blf.inner_solve * other
        else:
            self._res.vec.data = self._inv * other
        if not isinstance(self._inv, _petsc):
            mpi.print_("\t[Direct Inv. Mult] Time = {:.2f}".format(time.time()-start))
        return self._res.vec


def newton(F, x, eps=1e-5, max_iter=30, gamma=1):
    if not F.isNonlinear:
        max_iter=1
    dx = x.CreateVector()
    for i in range(max_iter):
        dx.data = F.Jacobian(x)*F(x)
        dx.data *= gamma
        x -= dx
        R = np.sqrt(np.divide(dx.InnerProduct(dx), x.InnerProduct(x)))
        mpi.print_("\tResidual R =", R)
        if R < eps:
            if i!=0:
                mpi.print_("\t[Newton solver] Converged in", i, "steps.")
            return x
    if max_iter !=1:
        raise ConvergenceError("[Newton solver] NOT Converged in " + str(i) + " steps.")
    return x


class ConvergenceError(RuntimeError):
    def __init__(self, msg):
        mpi.print_(msg)
        super().__init__(msg)
