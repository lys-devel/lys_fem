import os
from lys.Qt import QtWidgets

from ..fem import FEMProject, FEMSolution
from ..widgets import FEMTreeItem


class SolutionTree(FEMTreeItem):
    def __init__(self, canvas):
        super().__init__(canvas=canvas)

    def setSolutionPath(self, path):
        if len(path) == 0:
            return
        self.clear()
        self._obj = FEMProject.fromFile("FEM/" + path + "/input.dic")
        dirs = os.listdir("FEM/" + path + "/Solutions")
        for d, sol in zip(dirs, self._obj.solvers):
            self.append(_SolutionGUI(self, self._obj, d, sol, path="FEM/" + path, solverName=d))


class _SolutionGUI(FEMTreeItem):
    def __init__(self, parent, fem, solution, solver, path, solverName):
        super().__init__(parent=parent, children=[_ModelGUI(self, m, path, solverName) for m in fem.models])
        self._solution = solution
        self._solver = solver

    @property
    def name(self):
        return self._solution + ": " + self._solver.name


class _ModelGUI(FEMTreeItem):
    def __init__(self, parent, model, path, solverName):
        super().__init__(parent=parent)
        self._model = model
        self._path = path
        self._solver = solverName

    @property
    def name(self):
        return self._model.name

    @property
    def widget(self):
        return _FEMSolutionWidget(self.fem(), self.canvas(), self._path, self._solver, self._model)


class _FEMSolutionWidget(QtWidgets.QWidget):
    def __init__(self, fem, canvas, path, solver, model):
        super().__init__()
        self._fem = fem
        self._canvas = canvas
        self._path = path
        self._solver = solver
        self._model = model
        self.__initlayout()

    def __initlayout(self):
        self._list = QtWidgets.QComboBox()
        self._list.addItems(self._model.evalList())
        self._time = QtWidgets.QSpinBox()
        self._time.setRange(0, 10000000)
        self._time.valueChanged.connect(self.__show)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(QtWidgets.QPushButton("Show", clicked=self.__show))

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._list)
        layout.addWidget(self._time)
        layout.addLayout(buttons)
        self.setLayout(layout)

    def __show(self):
        data = self.__loadData()
        with self._canvas.delayUpdate():
            self._canvas.clear()
            for w in data:
                o = self._canvas.append(w)
                o.showEdges(True)

    def __loadData(self):
        var = self._list.currentText()
        sol = FEMSolution(self._path)
        return sol.eval(var, model=self._model, data_number=self._time.value(), solver=self._solver)
