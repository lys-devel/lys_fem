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
            self.append(_SolutionGUI(self, d, sol, path="FEM/" + path))


class _SolutionGUI(FEMTreeItem):
    def __init__(self, parent, solution, solver, path):
        super().__init__(parent=parent)
        self._solution = solution
        self._solver = solver
        self._path = path
        self._obj = None

    @property
    def name(self):
        return self._solution + ": " + self._solver.className

    @property
    def widget(self):
        return _FEMSolutionWidget(self.canvas(), self.solution)

    @ property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Send to shell", self.treeWidget(), triggered=self.__add))
        return self._menu
    
    def __add(self):
        from lys import glb
        glb.shell().addObject(self.solution, name="sol")
 
    @property
    def solution(self):
        if self._obj is None:
            self._obj = FEMSolution(self._path, solver=self._solution)
        return self._obj

    
class _FEMSolutionWidget(QtWidgets.QWidget):
    def __init__(self, canvas, solution):
        super().__init__()
        self._canvas = canvas
        self._sol = solution
        self.__initlayout()

    def __initlayout(self):
        self._text = QtWidgets.QLineEdit()
        self._time = QtWidgets.QSpinBox()
        self._time.setRange(0, 10000000)
        self._time.valueChanged.connect(self.__show)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(QtWidgets.QPushButton("Evaluate", clicked=self.__show))

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._text)
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
        var = self._text.text()
        return self._sol.eval(var, data_number=self._time.value())
