import numpy as np
from .base import FEMObject


class SolverStep:
    """
    Solver step that determines which variable is solved. Settings for linear/nonlinear solver is also specified.

    Args:
        vars(list of str): The name of variables that is solved in this step.
        solver(str): The name of the linear solver.
        prec(str): The name of the preconditioner. It is used only for iterative solvers.
        symmetric(bool): If true, the linear solver assume that the stiffness matrix to be symmetric.
        condensation(bool): If true, static condensation will be applied.
        maxiter(int): The maximum iteration for nonlinear newton solver. It will be negrected if the problem is linear.
        damping(float): The damping factor for newton solver.
        eps(float): The convergence criteria for newton solver.
        deformation(str): The name of a variable that is used for deformation of mesh.
    """
    def __init__(self, vars=None, solver="pardiso", prec=None, symmetric=False, condensation=False, maxiter=30, damping=1, eps=1e-5, deformation=None):
        self._vars = vars
        self._deform = deformation
        self._solver = solver
        self._prec = prec
        self._sym = symmetric
        self._cond = condensation
        self._maxiter = maxiter
        self._damping = damping
        self._eps = eps

    @property
    def variables(self):
        return self._vars

    @property
    def deformation(self):
        return self._deform

    @property
    def solver(self):
        return self._solver

    @property
    def preconditioner(self):
        return self._prec
    
    @property
    def symmetric(self):
        return self._sym
    
    @property
    def condensation(self):
        return self._cond
    
    @property
    def newton_maxiter(self):
        return self._maxiter
    
    @property
    def newton_damping(self):
        return self._damping
    
    @property
    def newton_eps(self):
        return self._eps
    
    
    def saveAsDictionary(self):
        return {"vars": self._vars, "deformation": self._deform, "solver": self._solver, "prec": self._prec, "symmetric": self._sym, "condensation": self._cond, "eps": self._eps, "damping": self._damping, "maxiter": self._maxiter}
    
    @classmethod
    def loadFromDictionary(cls, d):
        return SolverStep(**d)


class FEMSolver(FEMObject):
    def __init__(self, steps=None, diff_expr=None, **kwargs):
        if steps is None:
            steps = [SolverStep(**kwargs)]
        self._steps = steps
        self._diff_expr = diff_expr

    def saveAsDictionary(self):
        d = {"solver": self.className, "steps": [s.saveAsDictionary() for s in self._steps], "diff_expr": self._diff_expr}
        return d
        
    @property
    def steps(self):
        return self._steps

    @property
    def diff_expr(self):
        return self._diff_expr

    @staticmethod
    def loadFromDictionary(d):
        cls_list = set(sum(solvers.values(), []))
        cls_dict = {m.className: m for m in cls_list}
        solver = cls_dict[d["solver"]]
        del d["solver"]

        d["steps"] = [SolverStep.loadFromDictionary(s) for s in d["steps"]]
        return solver(**d)


class StationarySolver(FEMSolver):
    className = "Stationary Solver"

    def widget(self, fem):
        from ..gui import StationarySolverWidget
        return StationarySolverWidget(fem, self)


class RelaxationSolver(FEMSolver):
    className = "Relaxation Solver"

    def __init__(self, dt0 = 1e-9, dx = 1e-1, factor=5, maxiter=100, maxStep=2, inf=False, **kwargs):
        super().__init__(**kwargs)
        self._dt0 = dt0
        self._dx = dx
        self._factor = factor
        self._maxStep = maxStep
        self._maxiter = maxiter
        self._inf = inf

    @property
    def dt0(self):
        return self._dt0
    
    @property
    def dx(self):
        return self._dx
    
    @property
    def factor(self):
        return self._factor

    @property
    def maxiter(self):
        return self._maxiter

    @property
    def maxStep(self):
        return self._maxStep

    @property
    def inf(self):
        return self._inf

    def widget(self, fem):
        from ..gui import RelaxationSolverWidget
        return RelaxationSolverWidget(fem, self)

    def saveAsDictionary(self):
        d = super().saveAsDictionary()
        d["dt0"] = self._dt0
        d["dx"] = self._dx
        d["factor"] = self._factor
        d["maxiter"] = self._maxiter
        d["maxStep"] = self._maxStep
        d["inf"] = self._inf
        return d


class TimeDependentSolver(FEMSolver):
    className = "Time Dependent Solver"

    def __init__(self, step=1, stop=100, **kwargs):
        super().__init__(**kwargs)
        self._step = step
        self._stop = stop

    def widget(self, fem):
        from ..gui import TimeDependentSolverWidget
        return TimeDependentSolverWidget(fem, self)

    def getStepList(self):
        return np.array([self._step]* (int(self._stop / self._step)+1))

    def saveAsDictionary(self):
        d = super().saveAsDictionary()
        d["step"] = self._step
        d["stop"] = self._stop
        return d


solvers = {"Stationary": [StationarySolver, RelaxationSolver], "Time dependent": [TimeDependentSolver]}

