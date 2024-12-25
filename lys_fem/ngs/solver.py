import os
import time
import shutil
import numpy as np
import ngsolve
import ngsolve.ngs2petsc as n2p
import petsc4py.PETSc as psc

from . import mpi, util

def generateSolver(fem, mesh, model, load=False):
    solvers = {"Stationary Solver": StationarySolver, "Relaxation Solver": RelaxationSolver, "Time Dependent Solver": TimeDependentSolver}
    result = []
    for i, s in enumerate(fem.solvers):
        sol = solvers[s.className]
        solver = sol(s, mesh, model, dirname="Solver" + str(i), load=load)
        solver.fem = fem
        result.append(solver)
    return result


class _Sol:
    def __init__(self, value):
        if isinstance(value, util.FiniteElementSpace):
            self._fes = value
            self._sols = (self._fes.gridFunction(), self._fes.gridFunction(), self._fes.gridFunction())
        else:
            self._fes = value[0].finiteElementSpace
            self._sols = value

    def __getitem__(self, index):
        return self._sols[index]
    
    def set(self, xva):
        if isinstance(xva, _Sol):
            xva = xva._sols
        for xi, yi in zip(self._sols, xva):
            if yi is not None:
                xi.vec.data = yi.vec
            else:
                xi.vec.data *= 0

    def project(self, fes):
        g = fes.gridFunction()
        for v in self._fes.model.variables:
            if v.type == "x":
                g.setComponent(v, util.SolutionFunction(v, self, 0).eval(fes)/v.scale)
            if v.type == "v":
                g.setComponent(v, util.SolutionFunction(v, self, 1).eval(fes)/v.scale)
            if v.type == "a":
                g.setComponent(v, util.SolutionFunction(v, self, 2).eval(fes)/v.scale)
        return g
    
    def save(self, path):
        self._sols[0].Save(path, parallel=mpi.isParallel())
        self._sols[1].Save(path+"_v", parallel=mpi.isParallel())
        self._sols[2].Save(path+"_a", parallel=mpi.isParallel())

    @staticmethod
    def load(fes, path, parallel=None):
        if parallel is None:
            parallel = mpi.isParallel()
        x, v, a = (fes.gridFunction(), fes.gridFunction(), fes.gridFunction())
        if os.path.exists(path):
            x.Load(path, parallel)
        if os.path.exists(path+"_v"):
            v.Load(path+"_v", parallel)
        if os.path.exists(path+"_a"):
            a.Load(path+"_a", parallel)
        return _Sol((x,v,a))

    @property
    def finiteElementSpace(self):
        return self._fes

    @property
    def replaceDict(self):
        """
        Returns a dictionary that replace trial functions with corresponding solutions.
        """
        return {trial: util.SolutionFunction(v, self, 0) for v, (trial, test) in self._fes.model.TnT.items()}


class _Solution:
    """
    Solution class stores the solutions and the time derivatives as grid function.
    The NGSFunctions based on the grid function is also provided by this class.
    """
    def __init__(self, fes, model, dirname=None, nlog=2):
        self._model = model
        self._sols = [_Sol(fes) for _ in range(nlog)]
        self._nlog = nlog

        if dirname is not None:
            self._dirname = dirname
        else:
            self._dirname = None

    def __getitem__(self, index):
        return self._sols[index]

    def initialize(self, fes, use_a):
        self.__update(_Sol(self._model.initialValue(fes, use_a)))
        if mpi.isRoot and self._dirname is not None:
            if os.path.exists(self._dirname):
                shutil.rmtree(self._dirname)
            os.makedirs(self._dirname, exist_ok=True)
            self.__save(0)

    def reset(self):
        zero = self._sols[0][0].finiteElementSpace.gridFunction()
        for _ in range(len(self._sols)):
            self.__update(_Sol((self._sols[0][0], zero, zero)))

    def updateSolution(self, x0, saveIndex=None):
        tdep = self._model.updater(self)
        fes = x0.finiteElementSpace

        x, v, a = fes.gridFunction(), fes.gridFunction(), fes.gridFunction()
        x.vec.data = x0.vec

        fs = x0.toNGSFunctions(self._model, "_new")
        tnt = self._model.TnT
        d = {trial: fs[v.name] for v, (trial, test) in tnt.items()}

        for var, (trial, test) in tnt.items():
            if trial in tdep:
                x.setComponent(var, tdep[trial].replace(d).eval(fes))
        for var, (trial, test) in tnt.items():
            if trial.t in tdep:
                v.setComponent(var, tdep[trial.t].replace(d).eval(fes))
        for var, (trial, test) in tnt.items():
            if trial.tt in tdep:
                a.setComponent(var, tdep[trial.tt].replace(d).eval(fes))

        self.__update(_Sol((x,v,a)))

        if self._dirname is not None and saveIndex is not None:
            self.__save(saveIndex)

    def __update(self, xva):
        for n in range(1, len(self._sols)):
            self._sols[-n].set(self._sols[-n+1])
        self._sols[0].set(xva)

    def __save(self, index):
        self._sols[0].save(self._dirname + "/ngs" + str(index))

    def X(self, var, n=0):
        return util.SolutionFunction(var, self._sols[n], 0)

    def V(self, var, n=0):
        return util.SolutionFunction(var, self._sols[n], 1)

    def A(self, var, n=0):
        return util.SolutionFunction(var, self._sols[n], 2)

    def error(self, var):
        val = self.grad(var)
        g = self._fes.gridFunction()
        g.setComponent(var, val.eval())
        g = g.toNGSFunctions(self._model, pre="_g")[var.name]
        return ngsolve.Integrate(((g-val)**2).eval(), self._mesh, ngsolve.VOL, element_wise=True)
    

class _Operator:
    def __init__(self, wf, mesh, model, sols, step):
        self._fes = util.FiniteElementSpace(model, mesh, step.variables, symmetric=step.symmetric, condense=step.condensation)
        self._step = step

        wf = self.__prepareWeakform(wf, model, sols, step.variables)
        self._lhs, self._rhs = wf.lhs, wf.rhs
        self.__update()

        self._nl = self._lhs.isNonlinear
        self._tdep_lhs = self._lhs.isTimeDependent
        self._tdep_rhs = self._rhs.isTimeDependent
        self._init  = False

    def __update(self, bilinear=True, linear=True):
        if bilinear:
            self._blf = self._fes.BilinearForm()
            if self._lhs.valid:
                self._blf += self._lhs.eval(self._fes)
        if linear:
            self._lf = self._fes.LinearForm()
            if self._rhs.valid:
                self._lf += self._rhs.eval(self._fes)

    def __prepareWeakform(self, wf, model, sols, symbols):
        d = dict(model.discretize(sols))
        if symbols is not None:
            for v, (trial, test) in model.TnT.items():
                if v.name not in symbols:
                    d[trial] = sols.X(v)
                    d[trial.t] = sols.V(v)
                    if trial.tt in wf:
                        d[trial.tt] = sols.A(v)
                    d[test] = 0
                    d[util.grad(test)] = 0
        return wf.replace(d)
    
    def __call__(self, x):
        if self.isNonlinear:
            return self._blf.Apply(x) + self._lf.vec
        else:
            return self._blf.mat * x  + self._lf.vec

    def Jacobian(self, x):
        if self.isNonlinear:
            self._blf.AssembleLinearization(x)
            self._inv = self.__inverse(self._blf.mat)
        return self._inv

    def update(self):
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
            inv = _petsc(self._fes, mat, self._step.solver, self._step.preconditioner, iter=self._step.linear_maxiter, tol=self._step.linear_rtol)
        return _Inv(self._fes, inv, self._blf)

    @property
    def isNonlinear(self):
        return self._nl

    @property
    def finiteElementSpace(self):
        return self._fes

    def syncGridFunction(self, glb, direction, loc):
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
        ksp.getPC().setType(prec)
        ksp.setTolerances(rtol=self._tol, atol=0, divtol=1e16, max_it=self._iter)
        self._ksp = ksp

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
        mpi.print_("\t[Iterative solver] Iter =", self._ksp.its, ", Time = {:.2f}".format(time.time()-start))
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


class SolverBase:
    def __init__(self, obj, mesh, model, dirname, timeDep=False, variableStep=False, load=False):
        self._obj = obj
        self._fes = util.FiniteElementSpace(model, mesh)
        self._model = model
        self._mat = model.materials
        self._mat.const.dti.tdep = variableStep
        self._dirname = "Solutions/" + dirname

        wf = model.weakforms()
        self._sols = _Solution(self._fes, model, "Solutions/" + dirname)
        self._sols.initialize(self._fes, use_a = any([v.tt in wf for v, _ in model.TnT.values()]) and timeDep)
        self._ops = [_Operator(wf, mesh, model, self._sols, step) for step in obj.steps]
        self._xis = [op.finiteElementSpace.gridFunction() for op in self._ops]
        if not load:
            util.stepn.set(-1)

    @np.errstate(divide='ignore', invalid="ignore")
    def solve(self, dti=0):
        start = time.time()
        if dti==0:
            self._sols.reset()
        self._mat.updateSolutionFields(int(util.stepn.get()))
        self._mat.const.dti.set(dti)
        util.stepn.set(int(util.stepn.get() + 1))

        E0 = self.__calcDiff()
        self._step()
        E = self.__calcDiff()
        mpi.print_("\tTotal step time: {:.2f}".format(time.time()-start))
        mpi.print_()
        return np.divide(np.linalg.norm(E-E0), np.linalg.norm(E))

    def _step(self):
        x = self._sols[0].project(self._fes)
        for i, (op, xi, step) in enumerate(zip(self._ops, self._xis, self._obj.steps)):
            mpi.print_("\t=======Solver step", i+1, "=======")
            if step.deformation is not None:
                self.__applyDeformation(step)
            op.update()
            op.syncGridFunction(x, "->", xi)
            xi.vec.data = newton(op, xi.vec.CreateVector(copy=True), eps=step.newton_eps, max_iter=step.newton_maxiter, gamma=step.newton_damping)
            op.syncGridFunction(x, "<-", xi)
            mpi.print_()
        self.solutions.updateSolution(x, saveIndex=int(util.stepn.get()+1))

    def __applyDeformation(self, step):
        deform = {v.name: xx for v, xx in zip(self._model.variables, self._sols.X())}[step.deformation]
        space = ngsolve.VectorH1(self._mesh)
        gf = util.GridFunction(space, deform.eval())
        self._mesh.SetDeformation(gf)
        mpi.print_("deformation set")

    def __calcDiff(self):
        expr = self._obj.diff_expr
        if expr is None or expr=="":
            expr = self._model.variables[0].name
        d = self._sols[0].replaceDict
        return self._mat[expr].replace(d).integrate(self._fes)

    def exportRefinedMesh(self):
        err = self.solutions.error([v for v in self._model.variables if v.name=="x"][0])
        size = np.array([[0.02] for e in err])
        file = self._dirname + "/mesh"+str(int(util.stepn.get()+2))+".msh"
        self.fem.mesher.exportRefinedMesh(self.fem.geometries.generateGeometry(), self._mesh.tags, size, file)
        return file

    @property
    def solutions(self):
        return self._sols

    @property
    def name(self):
        return self._obj.className

    @property
    def obj(self):
        return self._obj
    
    def __str__(self):
        res = ""
        for i, op in enumerate(self._ops):
            res += "\tStep " + str(1+i) + "\n"
            res += str(op)
        return res


class StationarySolver(SolverBase):
    def execute(self):
        self.solve()


class RelaxationSolver(SolverBase):
    def __init__(self, obj, mesh, model, **kwargs):
        super().__init__(obj, mesh, model, timeDep=True, variableStep=True, **kwargs)
        self._tSolver = obj

    def execute(self):
        t = 0
        dt, dx_ref = self._tSolver.dt0, self._tSolver.dx
        for i in range(1,self._tSolver.maxiter):
            util.t.set(t)
            dx, dt = self._solve(dt)
            t = t + dt
            mpi.print_("Step", i, ", t = {:3e}".format(t), ", dt = {:3e}".format(dt), ", dx = {:3e}".format(dx))
            if dx != 0:
                dt *= min(np.sqrt(dx_ref/abs(dx)), self._tSolver.maxStep)
            if dt < self._tSolver.dt0:
                dt = self._tSolver.dt0
            if dt > self._tSolver.dt_max:
                dt = self._tSolver.dt_max
            if dt == self._tSolver.dt_max and dx < self._tSolver.tolerance:
                mpi.print_("[Relaxation solver] Converged in", i, "steps")
                return
                
    def _solve(self, dt):
        try:
            return self.solve(1/dt), dt
        except ConvergenceError:
            mpi.print_("Convergence problem detected. Time step is changed to " + str(dt/2))
            util.stepn.set(int(util.stepn.get()-1))
            return self._solve(dt/2)


class TimeDependentSolver(SolverBase):
    def __init__(self, obj, mesh, model, **kwargs):
        super().__init__(obj, mesh, model, timeDep=True, **kwargs)
        self._tSolver = obj

    def execute(self):
        t = 0
        for i, dt in enumerate(self._tSolver.getStepList()):
            util.t.set(t)
            dx = self.solve(1/dt)
            t = t + dt
            mpi.print_("Timestep", i, ", t = {:3e}".format(t), ", dx = {:3e}".format(dx))


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
