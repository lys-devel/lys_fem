from lys.Qt import QtWidgets


class FEMSolver:
    def __init__(self, models=None):
        if models is None:
            models = []
        self._models = models

    def addModel(self, model):
        self._models.append(model)

    def remove(self, index):
        self._models.remove(self._models[index])

    @property
    def models(self):
        return self._models

    def saveAsDictionary(self, fem):
        d = {"solver": self.name}
        d["models"] = [fem.models.index(m) for m in self._models]
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
        models = [fem.models[index]for index in d["models"]]
        return StationarySolver(models)


class TimeDependentSolver(FEMSolver):
    def __init__(self, models=None, step=1, stop=100):
        super().__init__(models)
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
        models = [fem.models[index]for index in d["models"]]
        return TimeDependentSolver(models, d["step"], d["stop"])


solvers = {"Stationary": [StationarySolver], "Time dependent": [TimeDependentSolver]}

