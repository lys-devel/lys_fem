from lys.Qt import QtWidgets
from lys_fem import FEMSolver, addSolver


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


addSolver("Stationary", StationarySolver)
