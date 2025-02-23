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
        rtol(float): The tolerance for iterative solvers. It will be neglected if the linear solver is direct.
        iter(int): The maximum iteration for iterative solver. It will be neglected if the linear solver is direct.
        maxiter(int): The maximum iteration for nonlinear newton solver. It will be negrected if the problem is linear.
        damping(float): The damping factor for newton solver.
        eps(float): The convergence criteria for newton solver.
    """
    def __init__(self, vars=None, solver="pardiso", prec=None, symmetric=False, condensation=False, rtol=1e-6, iter=5000, maxiter=30, damping=1, eps=1e-5, **kwargs):
        self._vars = vars
        self._solver = solver
        self._prec = prec
        self._sym = symmetric
        self._cond = condensation
        self._maxiter = maxiter
        self._damping = damping
        self._eps = eps
        self._iter = iter
        self._rtol = rtol

    @property
    def linear(self):
        return {"solver": self._solver, "prec": self._prec, "iter": self._iter, "rtol": self._rtol, "symmetric": self._sym, "condense": self._cond}

    @property
    def nonlinear(self):
        return {"eps": self._eps, "gamma": self._damping, "max_iter": self._maxiter}

    @property
    def variables(self):
        return self._vars

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
    
    @property
    def linear_maxiter(self):
        return self._iter

    @property 
    def linear_rtol(self):
        return self._rtol
    
    def saveAsDictionary(self):
        return {"vars": self._vars, "solver": self._solver, "prec": self._prec, "symmetric": self._sym, 
                "condensation": self._cond, "eps": self._eps, "damping": self._damping, "maxiter": self._maxiter, "iter": self._iter, "rtol": self._rtol}
    
    @classmethod
    def loadFromDictionary(cls, d):
        return SolverStep(**d)


class AdaptiveMeshRefinement:
    def __init__(self, var, nodes, range=(0, 1e30), maxiter=30):
        self._var = var
        self._nodes = nodes
        self._range = range
        self._maxiter = maxiter

    @property
    def varName(self):
        return self._var
    
    @property
    def nodes(self):
        return self._nodes
    
    @property
    def range(self):
        return self._range
    
    @property
    def maxiter(self):
        return self._maxiter
    
    def saveAsDictionary(self):
        return {"var": self._var, "nodes": self._nodes, "range": self._range, "maxiter": self._maxiter}
    
    @classmethod
    def loadFromDictionary(cls, d):
        return cls(**d)
   

class FEMSolver(FEMObject):
    """
    Args:
        steps (list of SolverStep): Solter steps that specifies detailed solver behavior.
        diff_expr (string expression): The expression of dx, which is used for relaxation etc.
        amr (AdaptiveMeshRefinement): The adaptive mesh refinement parameter.
    """
    def __init__(self, steps=None, diff_expr=None, amr=None, **kwargs):
        if steps is None:
            steps = [SolverStep(**kwargs)]
        self._steps = steps
        self._diff_expr = diff_expr
        self._amr = amr

    def setAdaptiveMeshRefinement(self, var, nodes=1000, maxiter=5, range=(0,1e5)):
        if var is None:
            self._amr = None
        else:
            self._amr = AdaptiveMeshRefinement(var, nodes, range=range, maxiter=maxiter)

    def saveAsDictionary(self):
        d = {"solver": self.className, "steps": [s.saveAsDictionary() for s in self._steps], "diff_expr": self._diff_expr}
        if self._amr is not None:
            d["amr"] = self._amr.saveAsDictionary()
        return d
        
    @property
    def steps(self):
        return self._steps

    @property
    def diff_expr(self):
        return self._diff_expr
    
    @property
    def adaptive_mesh(self):
        return self._amr

    @staticmethod
    def loadFromDictionary(d):
        cls_list = set(sum(solvers.values(), []))
        cls_dict = {m.className: m for m in cls_list}
        solver = cls_dict[d["solver"]]
        del d["solver"]

        d["steps"] = [SolverStep.loadFromDictionary(s) for s in d["steps"]]
        d["amr"] = d.get("amr", None)
        if d["amr"] is not None:
            d["amr"] = AdaptiveMeshRefinement.loadFromDictionary(d["amr"])
        return solver(**d)


class StationarySolver(FEMSolver):
    className = "Stationary Solver"

    def widget(self, fem):
        from ..gui import StationarySolverWidget
        return StationarySolverWidget(fem, self)


class RelaxationSolver(FEMSolver):
    className = "Relaxation Solver"

    def __init__(self, dt0 = 1e-9, dx = 1e-1, dt_max=1e-7, maxiter=100, maxStep=2, tolerance=1e-5, **kwargs):
        super().__init__(**kwargs)
        self._dt0 = dt0
        self._dx = dx
        self._dt_max = dt_max
        self._maxStep = maxStep
        self._maxiter = maxiter
        self._tolerance = tolerance

    @property
    def dt0(self):
        return self._dt0
    
    @property
    def dx(self):
        return self._dx
    
    @property
    def tolerance(self):
        return self._tolerance

    @property
    def maxiter(self):
        return self._maxiter

    @property
    def maxStep(self):
        return self._maxStep

    @property
    def dt_max(self):
        return self._dt_max

    def widget(self, fem):
        from ..gui import RelaxationSolverWidget
        return RelaxationSolverWidget(fem, self)

    def saveAsDictionary(self):
        d = super().saveAsDictionary()
        d["dt0"] = self._dt0
        d["dx"] = self._dx
        d["dt_max"] = self._dt_max
        d["maxiter"] = self._maxiter
        d["maxStep"] = self._maxStep
        d["tolerance"] = self._tolerance
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

