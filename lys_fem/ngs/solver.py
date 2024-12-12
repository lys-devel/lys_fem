import os
import shutil
import numpy as np
import ngsolve

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

    def X(self, var, n=0):
        return self._coefs[n][0][var.name]

    def V(self, var, n=0):
        return self._coefs[n][1][var.name]

    def A(self, var, n=0):
        self._use_a=True
        return self._coefs[n][2][var.name]
    
    def grad(self, var, n=0):
        return self._grads[n][var.name]

    @property
    def nlog(self):
        return len(self._sols)

    @property
    def finiteElementSpace(self):
        return self._model.finiteElementSpace
    
    @property
    def replaceDict(self):
        """
        Returns a dictionary that replace trial functions with corresponding solutions.
        """
        tnt = self._model.TnT
        sols = self[0][0].toNGSFunctions(self._model)
        grads = self[0][0].toGradFunctions(self._model)
        d = {trial: sols[v.name] for v, (trial, test) in tnt.items()}
        d.update({util.grad(trial): grads[v.name] for v, (trial, test) in tnt.items()})
        return d


class _Operator:
    def __init__(self, model, sols, symbols, solver="pardiso", prec=None, cond=False, sym=False):
        self._model = model
        self._symbols = symbols
        self._solver = solver

        self._fes = self.__compressed_fes(model.finiteElementSpace, symbols)
        #for i in range(model.finiteElementSpace.ndof):
        #    print(i, model.finiteElementSpace.CouplingType(i))
        #print(self._fes.ndof, self._fes.FreeDofs())
        #for i in range(self._fes.ndof):
        #    print(i, self._fes.CouplingType(i))

        self._blf, self._lf = ngsolve.BilinearForm(self._fes, condense=cond, symmetric=sym), ngsolve.LinearForm(self._fes)
        wf = self.__prepareWeakform(model, sols, symbols)
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

    def __compressed_fes(self, fes, symbols):
        #print(fes.ndof, fes.FreeDofs())
        if symbols is None:
            return fes

        dofs = ngsolve.BitArray(fes.FreeDofs())
        n = 0
        for v in self._model.variables:
            for j in range(n, n+v.size):
                dofs[fes.Range(j)]=v.name in symbols
            n += v.size
        self._mask = dofs
        return ngsolve.Compress(fes, active_dofs=dofs)

    def __prepareWeakform(self, model, sols, symbols):
        tnt = util.TnT_dict(self._model.variables, self._fes)
        wf = model.weakforms(tnt)
        d = dict(model.discretize(tnt, sols))
        if symbols is not None:
            for v, (trial, test) in tnt.items():
                if v.name not in symbols:
                    d[trial] = sols.X(v)
                    d[trial.t] = sols.V(v)
                    if trial.tt in wf:
                        d[trial.tt] = sols.A(v)
                    d[util.grad(trial)] = sols.grad(v)
                    d[test] = 0
                    d[util.grad(test)] = 0
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
            inv = mat.Inverse(self._fes.FreeDofs(self._blf.condense), self._solver)
        if self._solver == "CG":
            inv = ngsolve.CGSolver(mat, self._prec.mat)
        if self._solver == "GMRES":
            inv = ngsolve.GMRESSolver(mat, self._prec.mat)
        return _Inv(inv, self._blf)

    @property
    def isNonlinear(self):
        return self._nl

    @property
    def finiteElementSpace(self):
        return self._fes

    def syncGridFunction(self, glb, direction, loc):
        if self._symbols is None:
            if direction == "->":
                loc.vec.data = glb.vec
            elif direction == "<-":
                glb.vec.data = loc.vec
        else:
            if direction == "->":
                loc.vec.FV().NumPy()[:] = glb.vec.FV().NumPy()[self._mask]
            elif direction == "<-":
                glb.vec.FV().NumPy()[self._mask] = loc.vec.FV().NumPy()


class _Inv:
    def __init__(self, inv, blf):
        self._inv = inv
        self._blf = blf

    def __mul__(self, other):
        if self._blf.condense:
            ext = ngsolve.IdentityMatrix() + self._blf.harmonic_extension
            extT = ngsolve.IdentityMatrix() + self._blf.harmonic_extension_trans
            inv =  ext @ (self._inv + self._blf.inner_solve) @ extT
            return inv*other
        else:
            return self._inv * other


class SolverBase:
    def __init__(self, obj, mesh, model, dirname, variableStep=False, load=False):
        self._obj = obj
        self._mesh = mesh
        self._model = model
        self._mat = model.materials
        self._mat.const.dti.tdep = variableStep
        if not load:
            self.__prepareDirectory(dirname)

        self._sols = _Solution(model)
        self._ops = [_Operator(model, self._sols, step.variables, solver=step.solver, prec=step.preconditioner, cond = step.condensation, sym=step.symmetric) for step in obj.steps]
        util.stepn.set(-1)
        self._sols.initialize()
        self._diff = self.__calcDiff(obj.diff_expr)
        if not load:
            self.exportSolution(0)

    @np.errstate(divide='ignore', invalid="ignore")
    def solve(self, dti=0):
        if dti==0:
            self._sols.reset()
        self._mat.updateSolutionFields(int(util.stepn.get()))
        self._mat.const.dti.set(dti)
        util.stepn.set(int(util.stepn.get() + 1))

        E0 = self._diff.integrate(self._mesh)
        x = self._sols.copy()
        for op, step in zip(self._ops, self._obj.steps):
            if step.deformation is not None:
                deform = {v.name: xx for v, xx in zip(self._model.variables, self._sols.X())}[step.deformation]
                space = ngsolve.VectorH1(self._mesh)
                gf = util.GridFunction(space, deform.eval())
                self._mesh.SetDeformation(gf)
                print("deformation set")
            op.update()
            g = util.GridFunction(op.finiteElementSpace)
            op.syncGridFunction(x, "->", g)
            g.vec.data = newton(op, g.vec.CreateVector(copy=True), eps=step.newton_eps, max_iter=step.newton_maxiter, gamma=step.newton_damping)
            op.syncGridFunction(x, "<-", g)
        self.__updateSolution(x)
        self.exportSolution(int(util.stepn.get()+1))
        E = self._diff.integrate(self._mesh)
        return np.divide(np.linalg.norm(E-E0), np.linalg.norm(E))

    def __calcDiff(self, expr):
        if expr is None or expr=="":
            expr = self._model.variables[0].name
        d = self._sols.replaceDict
        return self._mat[expr].replace(d)

    def __prepareDirectory(self, dirname):
        self._dirname = "Solutions/" + dirname
        if mpi.isRoot:
            if os.path.exists(self._dirname):
                shutil.rmtree(self._dirname)
        os.makedirs(self._dirname, exist_ok=True)

    def __updateSolution(self, x0):
        fes = self._model.finiteElementSpace
        tnt = self._model.TnT
        self._tdep = self._model.updater(tnt, self._sols)

        x = util.GridFunction(fes)
        v = util.GridFunction(fes)
        a = util.GridFunction(fes)
        x.vec.data = x0.vec

        fs = x0.toNGSFunctions(self._model, "_new")
        d = {trial: fs[v.name] for v, (trial, test) in tnt.items()}

        for var, (trial, test) in tnt.items():
            if trial in self._tdep:
                x.setComponent(var, self._tdep[trial].replace(d).eval(), self._model)
        for var, (trial, test) in tnt.items():
            if trial.t in self._tdep:
                v.setComponent(var, self._tdep[trial.t].replace(d).eval(), self._model)
        for var, (trial, test) in tnt.items():
            if trial.tt in self._tdep:
                a.setComponent(var, self._tdep[trial.tt].replace(d).eval(), self._model)

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
                dt *= min(np.sqrt(dx_ref/abs(dx)), self._tSolver.maxStep)
            if dt < self._tSolver.dt0:
                dt = self._tSolver.dt0
            if dt > 1e-9:
                dt = 1e-9
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
        with ngsolve.TaskManager():
            dx.data = F.Jacobian(x)*F(x)
        dx.data *= gamma
        x -= dx
        R = np.sqrt(np.divide(dx.InnerProduct(dx), x.InnerProduct(x)))
        mpi.print_("R =", R)
        if R < eps:
            if i!=0:
                mpi.print_("[Newton solver] Converged in", i, "steps.")
            return x
    if max_iter !=1:
        raise RuntimeError("[Newton solver] NOT Converged in " + str(i) + " steps.")
    return x
