import os
from ..fem import FEMProject
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
            self.append(_SolutionGUI(self, self._obj, d, sol, path="FEM/" + path + "/Solutions/" + d))


class _SolutionGUI(FEMTreeItem):
    def __init__(self, parent, fem, solution, solver, path):
        super().__init__(parent=parent, children=[_ModelGUI(self, m, path) for m in fem.models])
        self._solution = solution
        self._solver = solver

    @property
    def name(self):
        return self._solution + ": " + self._solver.name


class _ModelGUI(FEMTreeItem):
    def __init__(self, parent, model, path):
        super().__init__(parent=parent)
        self._model = model
        self._path = path

    @property
    def name(self):
        return self._model.name

    @property
    def widget(self):
        return self._model.resultWidget(self.fem(), self.canvas(), self._path)
