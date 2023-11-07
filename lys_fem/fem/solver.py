from lys.Qt import QtWidgets


class FEMSolver:
    def __init__(self, models=None, subSolvers=None):
        if models is None:
            models = []
        if subSolvers is None:
            subSolvers = []
        self._models = models
        self._subs = subSolvers

    def addModel(self, model, solver):
        self._models.append(model)
        self._subs.append(solver)

    def remove(self, index):
        self._models.remove(self._models[index])
        self._subs.remove(self._subs[index])

    @property
    def subSolvers(self):
        return self._subs

    @property
    def models(self):
        return self._models

    def saveAsDictionary(self, fem):
        d = {"solver": self.name}
        d["models"] = [fem.models.index(m) for m in self._models]
        d["subs"] = [s.saveAsDictionary() for s in self._subs]
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
        subs = [subSolvers[sub["type"]].loadFromDictionary(sub) for sub in d["subs"]]
        return StationarySolver(models, subs)


class TimeDependentSolver(FEMSolver):
    def __init__(self, models=None, tSolvers=None, step=1, stop=100):
        super().__init__(models, tSolvers)
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
        subs = [tSolvers[sub["type"]].loadFromDictionary(sub) for sub in d["subs"]]
        return TimeDependentSolver(models, subs, d["step"], d["stop"])


class FEMSubSolver:
    def widget(self, fem, canvas=None):
        return QtWidgets.QWidget()

    def saveAsDictionary(self):
        return {"type": self.name}

    @classmethod
    def loadFromDictionary(cls, d):
        return cls()


class CGSolver(FEMSubSolver):
    @classmethod
    @property
    def name(cls):
        return "CG Solver"


class NewtonSolver(FEMSubSolver):
    @classmethod
    @property
    def name(cls):
        return "Newton Solver"


class FEMTimeDependentSolver(FEMSubSolver):
    def __init__(self, femSolver=NewtonSolver()):
        self._solver = femSolver

    @property
    def femSolver(self):
        return self._solver

    def saveAsDictionary(self):
        d = super().saveAsDictionary()
        d["solver"] = self._solver.saveAsDictionary()
        return d

    @classmethod
    def loadFromDictionary(cls, d):
        s = d["solver"]
        sol = subSolvers[s["type"]].loadFromDictionary(s)
        return cls(sol)


class GeneralizedAlphaSolver(FEMTimeDependentSolver):
    @classmethod
    @property
    def name(cls):
        return "Generalized Alpha Solver"


class BackwardEulerSolver(FEMTimeDependentSolver):
    @classmethod
    @property
    def name(cls):
        return "Backward Euler Solver"


solvers = {"Stationary": [StationarySolver], "Time dependent": [TimeDependentSolver]}
subSolvers = {c.name: c for c in [CGSolver, NewtonSolver]}
tSolvers = {c.name: c for c in [BackwardEulerSolver, GeneralizedAlphaSolver]}
