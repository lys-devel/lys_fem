from lys.Qt import QtWidgets


class FEMSolver:
    def __init__(self, models, subSolvers):
        self._models = models
        self._subs = subSolvers

    def addModel(self, model, solver):
        self._models.append(model)
        self._subs.adppend(solver)

    @property
    def subSolvers(self):
        return self._subs

    @property
    def models(self):
        return self._models

    def saveAsDictionary(self, fem):
        d = {"solver": self.name}
        d["models"] = [fem.models.index(self.target)]
        d["subs"] = [s.saveAsDictionary() for s in self._subs]
        return d

    @staticmethod
    def loadFromDictionary(fem, d):
        cls_list = set(sum(solvers.values(), []))
        cls_dict = {m.name: m for m in cls_list}
        solver = cls_dict[d["solver"]]
        del d["solver"]
        return solver.loadFromDictionary(fem, d)

    @classmethod
    def _loadModels(cls, fem, d):
        models = [fem.models[index]for index in d["models"]]
        subs = [subSolvers[sub["type"]].loadFromDictionary(sub) for sub in d["subs"]]
        return models, subs


class StationarySolver(FEMSolver):
    @classmethod
    @property
    def name(cls):
        return "Stationary Solver"

    def widget(self):
        return QtWidgets.QWidget()

    @classmethod
    def loadFromDictionary(cls, fem, d):
        models, subs = cls._loadModels(fem, d)
        return StationarySolver(models, subs)


class TimeDependentSolver(FEMSolver):
    def __init__(self, models, tSolvers, step=1, stop=100):
        super().__init__(models, tSolvers)
        self._step = step
        self._stop = stop

    @classmethod
    @property
    def name(cls):
        return "Time Dependent Solver"

    def widget(self):
        return _TimeDependentSolverWidget(self)

    def getStepList(self):
        return [self._step] * int(self._stop / self._step)

    def saveAsDictionary(self, fem):
        d = super().saveAsDictionary()
        d["step"] = self._step
        d["stop"] = self._stop
        return d

    @classmethod
    def loadFromDictionary(cls, fem, d):
        models, subs = cls._loadModels(fem, d)
        return TimeDependentSolver(models, subs, d["step"], d["stop"])


class FEMSubSolver:
    def widget(self, fem, canvas=None):
        return QtWidgets.QWidget()

    def saveAsDictionary(self):
        return {"type": self.name()}

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
        subSolvers[s["type"]].loadFromDictionary(s)
        return cls(d["solver"])


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


class _SolverTargetWidget(QtWidgets.QWidget):
    def __init__(self, solver, fem):
        super().__init__()
        self._solver = solver
        self._fem = fem
        self.__initlayout(fem)

    def __initlayout(self, fem):
        self._targ = QtWidgets.QComboBox()
        self._targ.addItem("")
        self._targ.addItems([m.name for m in fem.models])
        if self._solver.target in fem.models:
            index = fem.models.index(self._solver.target)
            self._targ.setCurrentIndex(index + 1)
        self._targ.currentIndexChanged.connect(self.__change)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Target"))
        layout.addWidget(self._targ)
        self.setLayout(layout)

    def __change(self, value):
        self._solver.target = self._fem.models[value - 1]


class _TimeDependentSolverWidget(QtWidgets.QWidget):
    def __init__(self, solver):
        super().__init__()
        self._solver = solver
        self.__initlayout()

    def __initlayout(self):
        self._step = QtWidgets.QDoubleSpinBox()
        self._step.setRange(0, 100000000)
        self._step.setValue(self._solver._step)
        self._step.valueChanged.connect(self.__change)
        self._stop = QtWidgets.QDoubleSpinBox()
        self._stop.setRange(0, 1000000000)
        self._stop.setValue(self._solver._stop)
        self._stop.valueChanged.connect(self.__change)

        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Step"), 0, 0)
        layout.addWidget(self._step, 0, 1)
        layout.addWidget(QtWidgets.QLabel("Stop"), 1, 0)
        layout.addWidget(self._stop, 1, 1)
        self.setLayout(layout)

    def __change(self):
        self._solver._step = self._step.value()
        self._solver._stop = self._stop.value()


solvers = {"Stationary": [StationarySolver], "Time dependent": [TimeDependentSolver]}
subSolvers = {c.name: c for c in [CGSolver, NewtonSolver]}
tSolvers = {c.name: c for c in [BackwardEulerSolver, GeneralizedAlphaSolver]}
