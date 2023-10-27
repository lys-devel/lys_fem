from lys.Qt import QtWidgets


class FEMSolver:
    def saveAsDictionary(self):
        return {"solver": self.name}

    @staticmethod
    def loadFromDictionary(fem, d):
        cls_list = set(sum(solvers.values(), []))
        cls_dict = {m.name: m for m in cls_list}
        solver = cls_dict[d["solver"]]
        del d["solver"]
        return solver.loadFromDictionary(fem, d)


class StationarySolver(FEMSolver):
    def __init__(self, subs=None):
        if subs is None:
            subs = []
        self._subs = subs

    @classmethod
    @property
    def name(cls):
        return "Stationary Solver"

    @property
    def subSolvers(self):
        return self._subs

    def widget(self):
        return QtWidgets.QWidget()

    def addSubSolver(self, sub):
        obj = sub()
        self._subs.append(obj)
        return obj

    @classmethod
    @property
    def subSolverTypes(cls):
        return [LinearSolver]

    def saveAsDictionary(self, fem):
        d = super().saveAsDictionary()
        d["subs"] = {s.name: s.saveAsDictionary(fem) for s in self._subs}
        return d

    @classmethod
    def loadFromDictionary(cls, fem, d):
        types = {t.name: t for t in cls.subSolverTypes}
        subs = [types[name].loadFromDictionary(fem, s) for name, s in d["subs"].items()]
        return StationarySolver(subs)


class TimeDependentSolver(FEMSolver):
    def __init__(self, step=1, stop=100, subs=None):
        if subs is None:
            subs = []
        self._subs = subs
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

    @property
    def subSolvers(self):
        return self._subs

    def addSubSolver(self, sub):
        obj = sub()
        self._subs.append(obj)
        return obj

    @classmethod
    @property
    def subSolverTypes(cls):
        return [GeneralizedAlphaSolver]

    def saveAsDictionary(self, fem):
        d = super().saveAsDictionary()
        d["subs"] = {s.name: s.saveAsDictionary(fem) for s in self._subs}
        d["step"] = self._step
        d["stop"] = self._stop
        return d

    @classmethod
    def loadFromDictionary(cls, fem, d):
        types = {t.name: t for t in cls.subSolverTypes}
        subs = [types[name].loadFromDictionary(fem, s) for name, s in d["subs"].items()]
        return TimeDependentSolver(d["step"], d["stop"], subs)


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


class LinearSolver(FEMSolver):
    def __init__(self, target=None):
        self.target = target

    def widget(self, fem, canvas=None):
        return _LinearSolverWidget(self, fem)

    def saveAsDictionary(self, fem):
        if self.target in fem.models:
            return {"modelIndex": fem.models.index(self.target)}
        else:
            return {}

    @property
    def variableName(self):
        return self.target.variableName

    @classmethod
    def loadFromDictionary(cls, fem, d):
        if "modelIndex" in d:
            return LinearSolver(fem.models[d["modelIndex"]])
        else:
            return LinearSolver()

    @classmethod
    @property
    def name(cls):
        return "Linear Solver"

    @classmethod
    @property
    def objName(cls):
        return "Linear Solver"


class _LinearSolverWidget(QtWidgets.QWidget):
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


class GeneralizedAlphaSolver(FEMSolver):
    def __init__(self, target=None):
        self.target = target

    def widget(self, fem, canvas=None):
        return _GeneralizedAlphaSolverWidget(self, fem)

    def saveAsDictionary(self, fem):
        if self.target in fem.models:
            return {"modelIndex": fem.models.index(self.target)}
        else:
            return {}

    @property
    def variableName(self):
        return self.target.variableName

    @classmethod
    def loadFromDictionary(cls, fem, d):
        if "modelIndex" in d:
            return GeneralizedAlphaSolver(fem.models[d["modelIndex"]])
        else:
            return GeneralizedAlphaSolver()

    @classmethod
    @property
    def name(cls):
        return "Generalized Alpha Solver"

    @classmethod
    @property
    def objName(cls):
        return "Generalized Alpha Solver"


class _GeneralizedAlphaSolverWidget(QtWidgets.QWidget):
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


solvers = {"Stationary": [StationarySolver], "Time dependent": [TimeDependentSolver]}
