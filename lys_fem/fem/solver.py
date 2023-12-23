from lys.Qt import QtWidgets


class FEMSolver:
    def saveAsDictionary(self, fem):
        d = {"solver": self.name}
        return d

    @staticmethod
    def loadFromDictionary(fem, d):
        cls_list = set(sum(solvers.values(), []))
        cls_dict = {m.name: m for m in cls_list}
        solver = cls_dict[d["solver"]]
        del d["solver"]
        return solver.loadFromDictionary(fem, d)


class StationarySolver(FEMSolver):
    @classmethod
    @property
    def name(cls):
        return "Stationary Solver"

    def widget(self, fem):
        from ..gui import StationarySolverWidget
        return StationarySolverWidget(fem, self)

    @classmethod
    def loadFromDictionary(cls, fem, d):
        return StationarySolver()


class TimeDependentSolver(FEMSolver):
    def __init__(self, step=1, stop=100):
        super().__init__()
        self._step = step
        self._stop = stop

    @classmethod
    @property
    def name(cls):
        return "Time Dependent Solver"

    def widget(self, fem):
        from ..gui import TimeDependentSolverWidget
        return TimeDependentSolverWidget(fem, self)

    def getStepList(self):
        return [self._step] * int(self._stop / self._step)

    def saveAsDictionary(self, fem):
        d = super().saveAsDictionary(fem)
        d["step"] = self._step
        d["stop"] = self._stop
        return d

    @classmethod
    def loadFromDictionary(cls, fem, d):
        return TimeDependentSolver(d["step"], d["stop"])


solvers = {"Stationary": [StationarySolver], "Time dependent": [TimeDependentSolver]}

