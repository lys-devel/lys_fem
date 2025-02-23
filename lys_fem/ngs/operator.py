import time
import numpy as np

from . import mpi, util

class Operator:
    def __init__(self, fes, model, sols, step, type="discretized"):
        self._step = step

        self._fes = fes.compress(step.variables)
        wf = model.weakforms(type=type, sols=sols, symbols=step.variables)
        self._blf = self._fes.BilinearForm(wf.lhs, step.condensation, step.symmetric)
        self._lf = self._fes.LinearForm(wf.rhs)
        self._inv = util.LinearSolver(self._fes, self._blf, step.solver, step.preconditioner, step.linear_maxiter, step.linear_rtol, parallel=mpi.isParallel())
  
    def __call__(self, x):
        return self._blf * x + self._lf

    def Jacobian(self, x):
        if self._blf.linearize(x):
            self._inv.update()
        return self._inv

    @property
    def isNonlinear(self):
        return self._blf.isNonlinear

    def solve(self, x):
        self.__update()
        xi = self._fes.gridFunction()
        self.__syncGridFunction(x, "->", xi)
        xi.vec.data = newton(self, xi.vec.CreateVector(copy=True), eps=self._step.newton_eps, max_iter=self._step.newton_maxiter, gamma=self._step.newton_damping)
        self.__syncGridFunction(x, "<-", xi)

    def __update(self):
        self._lf.update()
        start = time.time()
        if self._blf.update():
            self._inv.update()
            mpi.print_("\t[Assemble] Time = {:.2f}".format(time.time()-start))
    
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
        res += "\t\tTotal degree of freedoms: " + str(self._fes.ndof) + "\n"
        res += "\t\tNonlinear: " + str(self.isNonlinear) + "\n"
        res += "\t\tTime dependent: LHS = " + str(self._blf.isTimeDependent) + ", RHS = " + str(self._lf.isTimeDependent) + "\n" 
        res += "\t\tStatic condensation: " + str(self._step.condensation) + "\n"
        if self._step.variables is not None:
            res += "\t\tSymbols: " + str(self._step.variables) + "\n"
        res += "\t\tSolver: " + str(self._step.solver) + "\n"
        if self._step.preconditioner is not None:
            res += "\t\tPreconditioner: " + str(self._step.preconditioner) + "\n"
            res += "\t\tMax iteration:" + str(self._step.linear_maxiter) + "\n"
            res += "\t\tRelative tolerance:" + str(self._step.linear_rtol) + "\n"
        return res


def newton(F, x, eps=1e-5, max_iter=30, gamma=1):
    if not F.isNonlinear:
        max_iter=1
    dx = x.CreateVector()
    for i in range(max_iter):
        J = F.Jacobian(x)
        dx.data = J*F(x)
        mpi.print_("\t"+J.msg)
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
