import os
import shutil
import numpy as np

import ngsolve
from ngsolve import BilinearForm, LinearForm, CoefficientFunction, VectorH1, TaskManager

from . import mpi, util

def generateSolver(fem, mesh, model, load=False):
    solvers = {"Stationary Solver": StationarySolver, "Relaxation Solver": RelaxationSolver, "Time Dependent Solver": TimeDependentSolver}
    result = []
    for i, s in enumerate(fem.solvers):
        sol = solvers[s.className]
        result.append(sol(s, mesh, model, dirname="Solver" + str(i), load=load))
    return result


class _Solution:
    def __init__(self, model, nlog=2):
        self._model = model
        fes = model.finiteElementSpace
        self._sols = [(util.GridFunction(fes), util.GridFunction(fes), util.GridFunction(fes)) for n in range(nlog)]
        self._coefs = [[self[n][i].toNGSFunctions(model, "t"*i+"_n") for i in range(3)] for n in range(nlog)]
        self._grads = [self[n][0].toGradFunctions(model, "_n") for n in range(nlog)]
        self._use_a = False

    def initialize(self):
        self.update(self._model.initialValue(self._use_a))

    def reset(self):
        for i in range(self.nlog):
            self.update((self._sols[0][0], None, None))

    def update(self, xva):
        # Store old data
        for j in range(3):
            for n in range(1, len(self._sols)):
                self._sols[-n][j].vec.data = self._sols[-n+1][j].vec
        # Set new one
        for xi, yi in zip(self._sols[0], xva):
            if yi is not None:
                xi.vec.data = yi.vec
            else:
                xi.vec.data *= 0

    def __getitem__(self, n):
        return self._sols[n]

    def copy(self):
        g = util.GridFunction(self._model.finiteElementSpace)
        g.vec.data = self._sols[0][0].vec
        return g
    
    def save(self, path):
        self._sols[0][0].Save(path, parallel=mpi.isParallel())

    def load(self, path, parallel):
        self._sols[0][0].Load(path, parallel = parallel)

    def X(self, n=0):
        return self._coefs[n][0]

    def V(self, n=0):
        return self._coefs[n][1]

    def A(self, n=0):
        self._use_a=True
        return self._coefs[n][2]
    
    def grad(self, n=0):
        return self._grads[n]

    @property
    def nlog(self):
        return len(self._sols)

    @property
    def finiteElementSpace(self):
        return self._model.finiteElementSpace


class _Operator:
    def __init__(self, model, diff, sols, symbols, solver="pardiso", prec=None):
        wf = self.__prepareWeakform(model, diff, sols, symbols)
        self._model = model
        self._symbols = symbols
        self._solver = solver
        self._fes = model.finiteElementSpace
        self.__setCoupling()

        self._blf, self._lf = BilinearForm(self._fes, condense=self._cond), LinearForm(self._fes)
        lhs, rhs = wf.lhs, wf.rhs
        if lhs.valid:
            self._blf += lhs.eval()
        if rhs.valid:
            self._lf += rhs.eval()
        if prec is not None:
            self._prec = ngsolve.Preconditioner(self._blf, prec)
        self._nl = lhs.isNonlinear
        self._tdep_lhs = lhs.isTimeDependent
        self._tdep_rhs = rhs.isTimeDependent
        self._init  = False

    def __prepareWeakform(self, model, diff, sols, symbols):
        wf = model.weakforms()
        d = dict(diff)
        if symbols is not None:
            X, V, A, G = sols.X(), sols.V(), sols.A(), sols.grad()
            for v in model.variables:
                if v.name not in symbols:
                    d[v.trial] = X[v.name]
                    d[v.trial.t] = V[v.name]
                    d[v.trial.tt] = A[v.name]
                    if util.grad(v.trial) not in d:
                        d[util.grad(v.trial)] = G[v.name]
                    d[v.test] = 0
                    d[util.grad(v.test)] = 0
        wf = wf.replace(d)
        return wf

    def __call__(self, x):
        if self.isNonlinear:
            return self._blf.Apply(x) + self._lf.vec
        else:
            return self._blf.mat * x  + self._lf.vec

    def Jacobian(self, x):
        if self.isNonlinear:
            self._blf.AssembleLinearization(x)
            return self.__inverse(self._blf.mat)
        else:
            return self._inv
        
    def update(self):
        if (self._tdep_rhs or not self._init):
            self._lf.Assemble()
        if (self._tdep_lhs or not self._init) and not self.isNonlinear:
            self._blf.Assemble()
            self._inv = self.__inverse(self._blf.mat)
        self._init = True

    def __inverse(self, mat):
        if self._solver in ["pardiso", "pardisospd", "mumps", "sparsecholesky", "masterinverse", "umfpack"]:
            inv = mat.Inverse(self._dofs, self._solver)
            if self._cond:
                ext = ngsolve.IdentityMatrix() + self._blf.harmonic_extension
                extT = ngsolve.IdentityMatrix() + self._blf.harmonic_extension_trans
                inv =  ext @ inv @ extT + self._blf.inner_solve
            return inv
        if self._solver == "CG":
            return ngsolve.CGSolver(mat, self._prec.mat)
        if self._solver == "GMRES":
            return ngsolve.GMRESSolver(mat, self._prec.mat)

    def __setCoupling(self):
        if isinstance(self._fes, ngsolve.ProductSpace):
            self._cond = any([isinstance(c, ngsolve.L2) for c in self._fes.components]) and self._symbols is None
        else:
            self._cond = False

        self._dofs = ngsolve.BitArray(self._fes.FreeDofs(coupling=self._cond))
        if self._symbols is None:
            return
        n = 0
        for v in self._model.variables:
            if v.name not in self._symbols:
                for j in range(n, n+v.size):
                    self._dofs[self._fes.Range(j)]=False
            n += v.size
        return

    @property
    def isNonlinear(self):
        return self._nl
    

class SolverBase:
    def __init__(self, obj, mesh, model, dirname, variableStep=False, load=False):
        self._obj = obj
        self._mesh = mesh
        self._model = model
        self._diff_expr = obj.diff_expr
        self._mat = model.materials
        self._mat.const.dti.tdep = variableStep
        if not load:
            self.__prepareDirectory(dirname)

        self._sols = _Solution(model)
        disc = model.discretize(self._sols, self._mat.const.dti)
        self._tdep = model.updater(self._sols, self._mat.const.dti)
        self._ops = [_Operator(model, disc, self._sols, step.variables, solver=step.solver, prec=step.preconditioner) for step in obj.steps]
        util.stepn.set(-1)
        self._sols.initialize()
        if not load:
            self.exportSolution(0)

    @np.errstate(divide='ignore', invalid="ignore")
    def solve(self, dti=0):
        if dti==0:
            self._sols.reset()
        self._mat.updateSolutionFields(int(util.stepn.get()))
        self._mat.const.dti.set(dti)
        util.stepn.set(int(util.stepn.get() + 1))

        x, x0 = self._sols.copy(), self._sols.copy()
        for op, step in zip(self._ops, self._obj.steps):
            if step.deformation is not None:
                deform = {v.name: xx for v, xx in zip(self._model.variables, self._sols.X())}[step.deformation]
                space = VectorH1(self._mesh)
                gf = util.GridFunction(space, deform.eval())
                self._mesh.SetDeformation(gf)
                print("deformation set")
            op.update()
            x.vec.data = newton(op, x.vec.CreateVector(copy=True), eps=step.newton_eps, max_iter=step.newton_maxiter, gamma=step.newton_damping)
        self.__updateSolution(x)
        self.exportSolution(int(util.stepn.get()+1))
        return self.__calcDifference(self._sols.copy(), x0)

    def __calcDifference(self, x, x0):
        # TODO: This should be based on solution?.
        if self._diff_expr is None:
            self._diff_expr = self._model.variables[0].name
        x, x0 = x.toNGSFunctions(self._model), x0.toNGSFunctions(self._model)
        xd, x0d = dict(self._mat), dict(self._mat)
        xd.update({v.name: x[v.name] for v in self._model.variables})
        x0d.update({v.name: x0[v.name] for v in self._model.variables})

        x = util.eval(self._diff_expr, xd).eval()
        x0 = util.eval(self._diff_expr, x0d).eval()
        diff = ngsolve.Integrate(x-x0, self._mesh)
        x = ngsolve.Integrate(x, self._mesh)
        return np.divide(np.linalg.norm(diff), np.linalg.norm(x))

    def __prepareDirectory(self, dirname):
        self._dirname = "Solutions/" + dirname
        if mpi.isRoot:
            if os.path.exists(self._dirname):
                shutil.rmtree(self._dirname)
        os.makedirs(self._dirname, exist_ok=True)

    def __updateSolution(self, x0):
        x = util.GridFunction(self._model.finiteElementSpace)
        v = util.GridFunction(self._model.finiteElementSpace)
        a = util.GridFunction(self._model.finiteElementSpace)
        x.vec.data = x0.vec

        fs = x0.toNGSFunctions(self._model, "_new")
        d = {v.trial: fs[v.name] for v in self._model.variables}

        for var in self._model.variables:
            if var.trial in self._tdep:
                x.setComponent(var, self._tdep[var.trial].replace(d).eval(), self._model)
        for var in self._model.variables:
            if var.trial.t in self._tdep:
                v.setComponent(var, self._tdep[var.trial.t].replace(d).eval(), self._model)
        for var in self._model.variables:
            if var.trial.tt in self._tdep:
                a.setComponent(var, self._tdep[var.trial.tt].replace(d).eval(), self._model)

        self._sols.update((x,v,a))

    def exportMesh(self, m):
        m.ngmesh.Save(self._dirname + "/mesh0.vol")

    def exportSolution(self, index):
        self._sols.save(self._dirname + "/ngs" + str(index))

    def importSolution(self, index, parallel, dirname=None):
        if dirname is None:
            dirname=self._dirname
        self._sols.load((dirname + "/ngs" + str(index)), parallel)

    @property
    def solutions(self):
        return self._sols

    @property
    def name(self):
        return self._obj.className

    @property
    def obj(self):
        return self._obj


class StationarySolver(SolverBase):
    def execute(self):
        self.solve()


class RelaxationSolver(SolverBase):
    def __init__(self, obj, mesh, model, **kwargs):
        super().__init__(obj, mesh, model, variableStep=True, **kwargs)
        self._tSolver = obj

    def execute(self):
        t = 0
        dt, dx_ref = self._tSolver.dt0, self._tSolver.dx
        for i in range(1,self._tSolver.maxiter):
            util.t.set(t)
            dx = self.solve(1/dt)
            t = t + dt
            mpi.print_("Step", i, ", t = {:3e}".format(t), ", dt = {:3e}".format(dt), ", dx = {:3e}".format(dx))
            if dt == np.inf:
                return
            if dx != 0:
                dt *= min(np.sqrt(dx_ref/dx), self._tSolver.maxStep)
            if dt > self._tSolver.dt0*10**self._tSolver.factor:
                if self._tSolver.inf:
                    print("inf")
                    dt = np.inf
                else:
                    return

class TimeDependentSolver(SolverBase):
    def __init__(self, obj, mesh, model, **kwargs):
        super().__init__(obj, mesh, model, **kwargs)
        self._tSolver = obj

    def execute(self):
        t = 0
        for i, dt in enumerate(self._tSolver.getStepList()):
            util.t.set(t)
            dx = self.solve(1/dt)
            mpi.print_("Timestep", i, ", t = {:3e}".format(t), ", dx = {:3e}".format(dx))
            t = t + dt


def newton(F, x, eps=1e-5, max_iter=30, gamma=1):
    if not F.isNonlinear:
        max_iter=1
    dx = x.CreateVector()
    for i in range(max_iter):
        with TaskManager():
            dx.data = F.Jacobian(x)*F(x)
        dx.data *= gamma
        x -= dx
        R = np.sqrt(np.divide(dx.InnerProduct(dx), x.InnerProduct(x)))
        #print("R =", R)
        if R < eps:
            if i!=0:
                mpi.print_("[Newton solver] Converged in", i, "steps.")
            return x
    if max_iter !=1:
        raise RuntimeError("[Newton solver] NOT Converged in " + str(i) + " steps.")
    return x
