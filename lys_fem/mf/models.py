import sympy as sp
import numpy as np

from ..fem import DirichletBoundary
from . import mfem
from .weakform import t, dt, prev, grad, WeakformParser

modelList = {}


def addMFEMModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, mesh, mat):
    return [modelList[m.name](m, mesh, mat) for m in fem.models]


class MFEMModel:
    def __init__(self, model):
        self._model = model

    @property
    def variableName(self):
        return self._model.variableName

    @property
    def dirichletCondition(self):
        conditions = [b for b in self._model.boundaryConditions if isinstance(b, DirichletBoundary)]
        bdr_dir = {i: [] for i in range(self._model.variableDimension())}
        for b in conditions:
            for axis, check in enumerate(b.components):
                if check:
                    bdr_dir[axis].extend(b.boundaries.getSelection())
        return bdr_dir

    def timeDiscretizedWeakForm(self, type="TimeDependent"):
        wf = self.weakform
        trials = self.trialFunctions
        if type=="Stationary":
            return discretizeTime_stationary(wf, trials)
        else:
            if self.__checkTimeDerivativeOrder(wf, trials) == 2:
                return discretizeTime_generalizedAlpha(wf, trials, 0.9)
            else:
                return discretizeTime_backwardEuler(wf, trials)

    def __checkTimeDerivativeOrder(self, weakform, trials):
        trials_t2 = [trial.diff(t).diff(t) for trial in trials]
        p = sp.poly(weakform, *trials_t2)
        for order, value in p.as_dict().items():
            if sum(order) > 0:
                return 2
        return 1


def discretizeTime_stationary(wf, trials):
    for trial in trials:
        wf = wf.subs({trial.diff(t).diff(t): 0, trial.diff(t): 0})
    return wf

def discretizeTime_backwardEuler(wf, trials):
    for trial in trials:
        wf = wf.subs({trial.diff(t).diff(t): 0, trial.diff(t): (trial-prev(trial))/dt})
    return wf


class CompositeModel:
    def __init__(self, mesh, models, type):
        self._mesh = mesh
        self._models = models
        self._type = type

        wf = self.weakform
        self._nonlinear = self.__checkNonlinear(wf, self.trialFunctions)
        self._parser = WeakformParser(wf, self.trialFunctions, self.coefficient)

    def __checkNonlinear(self, wf, trials):
        vars = []
        for trial in trials:
            vars.append(trial)
            for gt in grad(trial):
                vars.append(gt)
        p = sp.poly(wf, *vars)
        for order in p.as_dict().keys():
            if sum(order) > 1:
                return True
        return False

    def solve(self, solver, dt=1):
        self.dt = dt
        if not self.isNonlinear:
            solver.setMaxIteration(1)
        if not hasattr(self, "_x"):
            self._x = self.__getFromGridFunctions()
        self._update_t = True
        self._x = solver.solve(self, self._x)
        self.__setToGridFunctions(self._x)
        return self.solution

    def __getFromGridFunctions(self):
        x = mfem.BlockVector(self._block_offset)
        for i, trial in enumerate(self.trialFunctions):
            trial.mfem.x.GetTrueDofs(x.GetBlock(i))
        return x
    
    def __setToGridFunctions(self, x):
        x0 = mfem.BlockVector(x, self._block_offset)
        for i, t in enumerate(self.trialFunctions):
            t.mfem.x.SetFromTrueDofs(x0.GetBlock(i))

    def update(self, x):
        # Translate x to grid functions
        x = mfem.BlockVector(x, self._block_offset)
        x_gfs = []
        for i, trial in enumerate(self.trialFunctions):
            x_gf = mfem.GridFunction(trial.mfem.space)
            x_gf.SetFromTrueDofs(x.GetBlock(i))
            x_gfs.append(x_gf)      

        # Parse matrices by updated coefficients
        coeffs = self.__updateCoefficients(x_gfs)
        self.K, self.b, self._J = self._parser.update(x_gfs, self.dt, coeffs)

    def __updateCoefficients(self, x_gfs):
        coeffs = {}
        # prepare coefficient for trial functions and its derivative.
        for gf, trial in zip(x_gfs, self.trialFunctions):
            coeffs[trial.mfem.name] = mfem.generateCoefficient(gf, trial.mfem.dimension)
            for d, gt in enumerate(trial.mfem.gradNames):
                gfd = mfem.GridFunction(trial.mfem.space)
                gf.GetDerivative(1, d, gfd)
                coeffs[gt] = mfem.generateCoefficient(gfd, trial.mfem.dimension)

        # coefficient for dt and previous value
        if self._update_t:
            coeffs["dt"] = mfem.generateCoefficient(self.dt, self._mesh.SpaceDimension())
            for trial in self.trialFunctions:
                p = prev(trial)
                coeffs[str(p)] = mfem.generateCoefficient(trial.mfem.x, self._mesh.SpaceDimension())
            self._update_t = False
        return coeffs

    def __call__(self, x):
        K, b = self.K, self.b
        res = mfem.Vector(x.Size())
        K.Mult(x, res)
        res -= b
        return res

    def grad(self, x):
        return self._J

    @property
    def _block_offset(self):
        offsets =mfem.intArray([0]+[t.mfem.space.GetTrueVSize() for t in self.trialFunctions])
        offsets.PartialSum()
        return offsets

    @property
    def weakform(self):
        return sum(m.timeDiscretizedWeakForm(self._type) for m in self._models)

    @property
    def trialFunctions(self):
        result = []
        for m in self._models:
            result.extend(m.trialFunctions)
        return result

    @property
    def coefficient(self):
        result = {}
        for m in self._models:
            result.update(m.coefficient)
        return result
    
    @property
    def isNonlinear(self):
        return self._nonlinear

    @property
    def solution(self):
        return {t.mfem.name.replace("trial_", ""): t.mfem.x.getData() for t in self.trialFunctions}

    # --------------For debug only----------------

    def printVector(self, vec):
        for i, t in enumerate(self.trialFunctions):
            gf = mfem.GridFunction(t.mfem.space)
            self._mesh.GetNodes(gf)
            print(str(t).replace("trial_", ""), [v[0] for v in zip(vec.GetBlock(i).GetDataArray())])

    def dualToPrime(self, vec):
        self.vb = mfem.BlockVector(vec, self._block_offset)
        self._res_dtp = mfem.BlockVector(self._block_offset)
        self._primes = [trial.mfem.dualToPrime(self.vb.GetBlock(i)) for i, trial in enumerate(self.trialFunctions)]
        for i, p in enumerate(self._primes):
            self._res_dtp.GetBlock(i).Set(1, p)
        return self._res_dtp

    def getNodalValue(self, vec):
        vb = mfem.BlockVector(vec, self._block_offset)
        res = []
        for i, tri in enumerate(self.trialFunctions):
            gf = mfem.GridFunction(tri.mfem.space)
            gf.SetFromTrueDofs(vb.GetBlock(i))
            r = gf.getData()[:,0]
            res.append(r)
        return np.array(res)