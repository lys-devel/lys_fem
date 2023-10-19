from lys.Qt import QtWidgets
from ..widgets import FEMTreeItem
from ..fem.solver import solvers


class SolverTree(FEMTreeItem):
    def __init__(self, obj, canvas):
        super().__init__(fem=obj, canvas=canvas)
        self.setSolvers(obj.solvers)

    def setSolvers(self, solvers):
        self.clear()
        self._solvers = solvers
        for s in solvers:
            super().append(_SolverGUI(s, self))

    def remove(self, init):
        i = super().remove(init)
        self._solvers.remove(self._solvers[i])

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        for group, sols in solvers.items():
            sub = self._menu.addMenu(group)
            for s in sols:
                sub.addAction(QtWidgets.QAction("Add " + s.name, self.treeWidget(), triggered=lambda x, y=s: self.__add(y())))
        return self._menu

    def __add(self, s):
        self._solvers.append(s)
        super().append(_SolverGUI(s, self))


class _SolverGUI(FEMTreeItem):
    def __init__(self, solver, parent):
        super().__init__(parent, children=[_SubSolver(d, self) for d in solver.subSolvers])
        self._solver = solver

    def remove(self, init):
        i = super().remove(init)
        self._solver.subSolvers.remove(self._solver.subSolvers[i])

    @property
    def name(self):
        return self._solver.name

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        for i in self._solver.subSolverTypes:
            self._menu.addAction(QtWidgets.QAction("Add " + i.name, self.treeWidget(), triggered=lambda x, y=i: self.__add(y)))
        self._menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return self._menu

    def __add(self, type):
        i = self._solver.addSubSolver(type)
        super().append(_SubSolver(i, self))


class _SubSolver(FEMTreeItem):
    def __init__(self, sub, parent):
        super().__init__(parent)
        self._sub = sub

    @property
    def name(self):
        return self._sub.objName

    @property
    def widget(self):
        return self._sub.widget(self.fem(), self.canvas())

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return self._menu
