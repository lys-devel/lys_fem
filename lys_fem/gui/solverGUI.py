from lys.Qt import QtWidgets
from ..widgets import FEMTreeItem
from ..fem.solver import solvers


class SolverTree(FEMTreeItem):
    def __init__(self, obj, canvas):
        super().__init__(fem=obj, canvas=canvas)
        self.setSolvers(obj.solvers)

    @property
    def children(self):
        return self._children

    def setSolvers(self, solvers):
        self._solvers = solvers
        self._children = [_SolverGUI(s, self) for s in solvers]

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        for group, sols in solvers.items():
            sub = self._menu.addMenu(group)
            for s in sols:
                sub.addAction(QtWidgets.QAction("Add " + s.name, self.treeWidget(), triggered=lambda x, y=s: self.__add(y)))
        return self._menu

    def __add(self, solver):
        self.beginInsertRow(len(self._children))
        s = solver()
        self._solvers.append(s)
        self._children.append(_SolverGUI(s, self))
        self.endInsertRow()

    def remove(self, init):
        i = self._children.index(init)
        self.beginRemoveRow(i)
        self._solvers.remove(self._solvers[i])
        self._children.remove(init)
        self.endRemoveRow()


class _SolverGUI(FEMTreeItem):
    def __init__(self, solver, parent):
        super().__init__(parent)
        self._solver = solver
        self._children = [_SubSolver(d, self) for d in solver.subSolvers]

    @property
    def name(self):
        return self._solver.name

    @property
    def children(self):
        return self._children

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        for i in self._solver.subSolverTypes:
            self._menu.addAction(QtWidgets.QAction("Add " + i.name, self.treeWidget(), triggered=lambda x, y=i: self.__add(y)))
        self._menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return self._menu

    def __add(self, type):
        self.beginInsertRow(len(self._children))
        i = self._solver.addSubSolver(type)
        self._children.append(_SubSolver(i, self))
        self.endInsertRow()

    def remove(self, init):
        i = self._children.index(init)
        self.beginRemoveRow(i)
        self._solver.subSolvers.remove(self._solver.subSolvers[i])
        self._children.remove(init)
        self.endRemoveRow()


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
