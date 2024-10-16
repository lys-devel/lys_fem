import os
import shutil
import numpy as np

import ngsolve
from ngsolve import BilinearForm, LinearForm, CoefficientFunction, VectorH1, TaskManager, SetNumThreads

from . import mpi, time, util

def generateSolver(fem, mesh, model):
    solvers = {"Stationary Solver": StationarySolver, "Relaxation Solver": RelaxationSolver, "Time Dependent Solver": TimeDependentSolver}
    result = []
    for i, s in enumerate(fem.solvers):
        sol = solvers[s.className]
        result.append(sol(s, mesh, model, "Solver" + str(i)))
    return result


class _Solution:
    def __init__(self, model, use_a, nlog=2):
        self._model = model
        fes = model.finiteElementSpace
        self._sols = [(util.GridFunction(fes), util.GridFunction(fes), util.GridFunction(fes)) for n in range(nlog)]
        self._coefs = [[self.__toFunc(self[n][i], "t"*i) for i in range(3)] for n in range(nlog)]
        self._grads = [self.__toGrad(self[n][0]) for n in range(nlog)]
        self.update(model.initialValue(use_a))

    def __toFunc(self, x, pre=""):
        res = {}
        n = 0
        for v in self._model.variables:
            if v.size == 1 and v.isScalar:
                res[v.name] = util.NGSFunction(v.scale*x.components[n], name=v.name+pre+"0", tdep=True)
            else:
                res[v.name] = util.NGSFunction(v.scale*CoefficientFunction(tuple(x.components[n:n+v.size])), name=v.name+pre+"0", tdep=True)
            n+=v.size           
        return res

    def __toGrad(self, x):
        res = {}
        n = 0
        for v in self._model.variables:
            if v.size == 1 and v.isScalar:
                res[v.name] = util.NGSFunction(v.scale*ngsolve.grad(x.components[n]), name="grad("+v.name+"0)", tdep=True)
            else:
                g = [ngsolve.grad(x.components[i]) for i in range(n,n+v.size)]
                res[v.name] = util.NGSFunction(v.scale*CoefficientFunction(tuple(g)), name="grad("+v.name+"0)", tdep=True)
            n+=v.size           
        return res

    def update(self, xva):
        # Store old data
        for j in range(3):
            for n in range(1, len(self._sols)):
                self._sols[-n][j].vec.data = self._sols[-n+1][j].vec
        # Set new one
        for xi, yi in zip(self._sols[0], xva):
            if yi is not None:
                xi.vec.data = yi.vec

    def __getitem__(self, n):
        return self._sols[n]

    def copy(self):
        g = util.GridFunction(self._model.finiteElementSpace)
        g.vec.data = self._sols[0][0].vec
        return g
    
    def save(self, path):
        self._sols[0][0].Save(path, parallel=mpi.isParallel())

    def X(self, n=0):
        return self._coefs[n][0]

    def V(self, n=0):
        return self._coefs[n][1]

    def A(self, n=0):
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
        self._nl_lhs = lhs.isNonlinear
        self._nl_rhs = rhs.isNonlinear
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
                    d[util.grad(v.trial)] = G[v.name]
                    d[v.test] = 0
                    d[util.grad(v.test)] = 0
        wf = wf.replace(d)
        return wf

    def __call__(self, x):
        with TaskManager():
            if self._nl_rhs:
                self._lf.Assemble()
            if self._nl_lhs:
                return self._blf.Apply(x) + self._lf.vec
            else:
                return self._blf.mat * x  + self._lf.vec

    def Jacobian(self, x):
        with TaskManager():
            if self._nl_lhs:
                self._blf.AssembleLinearization(x)
                #print("Condition Number:", 1/np.linalg.cond(self._blf.mat.ToDense()))
                res = self.__inverse(self._blf.mat)
                return res
            else:
                return self._inv
        
    def update(self, dti):
        self.__setCoupling()
        with TaskManager():
            if (self._tdep_rhs or not self._init) and not self._nl_rhs:
                self._lf.Assemble()
            if (self._tdep_lhs or not self._init) and not self._nl_lhs:
                self._blf.Assemble()
                self._inv = self.__inverse(self._blf.mat)
        self._init = True

    def __inverse(self, mat):
        with TaskManager():
            if self._solver in ["pardiso", "pardisospd", "mumps", "sparsecholesky", "masterinverse", "umfpack"]:
                inv = mat.Inverse(self._fes.FreeDofs(coupling=self._cond), self._solver)
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
            cond = [isinstance(c, ngsolve.L2) for c in self._fes.components]
        else:
            cond = [isinstance(self._fes, ngsolve.L2)]
        if self._symbols is None:
            self._cond = any(cond)
            return
        for i, c in enumerate(self._model.coupling):
            self._fes.SetCouplingType(i, c)
        n = 0
        for v in self._model.variables:
            if v.name not in self._symbols:
                for j in range(n, n+v.size):
                    cond[j] = False
                    self._fes.SetCouplingType(self._fes.Range(j), ngsolve.COUPLING_TYPE.UNUSED_DOF)
            n += v.size
        self._cond = False
        ngsolve.Compress(self._fes)
        
    @property
    def isNonlinear(self):
        return self._nl_lhs or self._nl_rhs
    


class SolverBase:
    def __init__(self, obj, mesh, model, dirname, variableStep=False):
        self._obj = obj
        self._mesh = mesh
        self._model = model
        self._mat = model.materials
        self._mat.const.dti.tdep = variableStep
        self.__prepareDirectory(dirname)

        self._integ = self.__prepareIntegrator(obj)
        self._sols = _Solution(model, self._integ.use_a)
        self._disc = model.discretize(self._sols, self._mat.const.dti)
        self._ops = [_Operator(model, self._disc, self._sols, step.variables, solver=step.solver, prec=step.preconditioner) for step in obj.steps]
        self._x = self._sols.copy()
        util.stepn.set(-1)

    @np.errstate(divide='ignore', invalid="ignore")
    def solve(self, dti=0):
        self._mat.updateSolutionFields(int(util.stepn.get()))
        util.stepn.set(int(util.stepn.get() + 1))
        x0 = self._x.vec.CreateVector(copy=True)
        self._mat.const.dti.set(dti)
        for op, step in zip(self._ops, self._obj.steps):
            if step.deformation is not None:
                deform = {v.name: x0 for v, x0 in zip(self._model.variables, self._sols.X())}[step.deformation]
                space = VectorH1(self._mesh)
                gf = util.GridFunction(space, deform.eval())
                self._mesh.SetDeformation(gf)
                print("deformation set")
            op.update(dti)
            self._x.vec.data = newton(op, self._x.vec.CreateVector(copy=True), eps=step.newton_eps, max_iter=step.newton_maxiter, gamma=step.newton_damping)
            self.__updateSolution(self._x, self._disc, self._sols, self._mat.const.dti)
        return self.__calcDifference(self._x.vec, x0)

    def __calcDifference(self, x, x0):
        x0.data = x0.data - x.data
        return np.divide(x0.InnerProduct(x0), x.InnerProduct(x))
            
    def __prepareIntegrator(self, obj):
        if obj.method == "ForwardEuler":
            return time.ForwardEuler()
        if obj.method == "BackwardEuler":
            return time.BackwardEuler()
        elif obj.method == "BDF2":
            return time.BDF2()
        elif obj.method == "NewmarkBeta":
            return time.NewmarkBeta()
        else:
            return time.GeneralizedAlpha(obj.method)

    def __prepareDirectory(self, dirname):
        self._dirname = "Solutions/" + dirname
        if mpi.isRoot:
            if os.path.exists(self._dirname):
                shutil.rmtree(self._dirname)
        os.makedirs(self._dirname, exist_ok=True)

    def __updateSolution(self, x, disc, sols, dti):
        v = util.GridFunction(self._model.finiteElementSpace)
        a = util.GridFunction(self._model.finiteElementSpace)

        n = 0
        for var in self._model.variables:
            # Calculate range for variable
            if v.isSingle:
                r = slice(None)
            else:
                r = slice(self._model.finiteElementSpace.Range(n).start, self._model.finiteElementSpace.Range(n+var.size-1).stop)
            n += var.size

            # Parameters
            d = {var.trial: x.vec[r], dti: dti.get(), util.stepn: util.stepn.get()}
            for i in range(sols.nlog):
                d[sols.X(i)[var.name]] = sols[i][0].vec[r]
                d[sols.V(i)[var.name]] = sols[i][1].vec[r]
                d[sols.A(i)[var.name]] = sols[i][2].vec[r]

            # Update by replacing discretization
            if var.trial.t in disc:
                v.vec.data[r] = disc[var.trial.t].replace(d, type="value")
            if var.trial.tt in disc:
                a.vec.data[r] = disc[var.trial.tt].replace(d, type="value")

        sols.update((x,v,a))

    def exportMesh(self, m):
        m.ngmesh.Save(self._dirname + "/mesh0.vol")

    def exportSolution(self, index):
        self._sols.save(self._dirname + "/ngs" + str(index))

    @property
    def solution(self):
        return self._sols

    @property
    def name(self):
        return self._obj.className

    @property
    def obj(self):
        return self._obj


class StationarySolver(SolverBase):
    def execute(self):
        #self.exportMesh(self._mesh)
        self.exportSolution(0)

        self.solve()
        self.exportSolution(1)


class RelaxationSolver(SolverBase):
    def __init__(self, obj, mesh, model, dirname):
        super().__init__(obj, mesh, model, dirname, variableStep=True)
        self._tSolver = obj

    def execute(self):
        #self.exportMesh(self._mesh)
        self.exportSolution(0)

        t = 0
        dt, dx_ref = self._tSolver.dt0, self._tSolver.dx
        for i in range(1,100):
            util.t.set(t)
            dx = self.solve(1/dt)
            t = t + dt
            mpi.print_("Step", i, ", t = {:3e}".format(t), ", dt = {:3e}".format(dt), ", dx = {:3e}".format(dx))
            self.exportSolution(i)
            if dt == np.inf:
                break
            dt *= min(np.sqrt(dx_ref/dx), 2)
            if dt > self._tSolver.dt0*10**self._tSolver.factor:
                print("inf")
                dt = np.inf


class TimeDependentSolver(SolverBase):
    def __init__(self, obj, mesh, model, dirname):
        super().__init__(obj, mesh, model, dirname)
        self._tSolver = obj

    def execute(self):
        #self.exportMesh(self._mesh)
        self.exportSolution(0)

        t = 0
        for i, dt in enumerate(self._tSolver.getStepList()):
            util.t.set(t)
            dx = self.solve(1/dt)
            mpi.print_("Timestep", i, ", t = {:3e}".format(t), ", dx = {:3e}".format(dx))
            self.exportSolution(i + 1)
            t = t + dt


def newton(F, x, eps=1e-5, max_iter=30, gamma=1):
    if not F.isNonlinear:
        max_iter=1
    dx = x.CreateVector()
    for i in range(max_iter):
        with ngsolve.TaskManager():
            dx.data = F.Jacobian(x)*F(x)
        dx.data *= gamma
        x -= dx
        R = np.sqrt(np.divide(dx.InnerProduct(dx), x.InnerProduct(x)))
        print("R =", R)
        if R < eps:
            if i!=0:
                mpi.print_("[Newton solver] Converged in", i, "steps.")
            return x
    if max_iter !=1:
        raise RuntimeError("[Newton solver] NOT Converged in " + str(i) + " steps.")
    return x
