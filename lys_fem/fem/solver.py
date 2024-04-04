import numpy as np
from .base import FEMObject


class FEMSolver(FEMObject):
    def __init__(self, method="BackwardEuler"):
        self._method = method

    def saveAsDictionary(self):
        d = {"solver": self.className, "method": self._method}
        return d
    
    @property
    def method(self):
        return self._method

    @staticmethod
    def loadFromDictionary(d):
        cls_list = set(sum(solvers.values(), []))
        cls_dict = {m.className: m for m in cls_list}
        solver = cls_dict[d["solver"]]
        del d["solver"]
        return solver.loadFromDictionary(d)


class StationarySolver(FEMSolver):
    className = "Stationary Solver"

    def widget(self, fem):
        from ..gui import StationarySolverWidget
        return StationarySolverWidget(fem, self)

    @classmethod
    def loadFromDictionary(cls, d):
        return StationarySolver()


class RelaxationSolver(FEMSolver):
    className = "Relaxation Solver"

    def __init__(self, dt0 = 1e-9, dx = 1e-1, method="BackwardEuler"):
        super().__init__(method)
        self._dt0 = dt0
        self._dx = dx

    @property
    def dt0(self):
        return self._dt0/self.fem.scaling.getScaling("s")

    @property
    def dx(self):
        return self._dx

    def widget(self, fem):
        from ..gui import StationarySolverWidget
        return StationarySolverWidget(fem, self)

    def saveAsDictionary(self):
        d = super().saveAsDictionary()
        d["dt0"] = self._dt0
        d["dx"] = self._dx
        return d

    @classmethod
    def loadFromDictionary(cls, d):
        return RelaxationSolver(d["dt0"], d["dx"], method=d.get("method", "BackwardEuler"))


class TimeDependentSolver(FEMSolver):
    className = "Time Dependent Solver"

    def __init__(self, step=1, stop=100, method="BackwardEuler"):
        super().__init__(method)
        self._step = step
        self._stop = stop

    def widget(self, fem):
        from ..gui import TimeDependentSolverWidget
        return TimeDependentSolverWidget(fem, self)

    def getStepList(self):
        return np.array([self._step]* (int(self._stop / self._step)+1))  / self.fem.scaling.getScaling("s")

    def saveAsDictionary(self):
        d = super().saveAsDictionary()
        d["step"] = self._step
        d["stop"] = self._stop
        return d

    @classmethod
    def loadFromDictionary(cls, d):
        return TimeDependentSolver(d["step"], d["stop"], method=d.get("method", "BackwardEuler"))


solvers = {"Stationary": [StationarySolver, RelaxationSolver], "Time dependent": [TimeDependentSolver]}

