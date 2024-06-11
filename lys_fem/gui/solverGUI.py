from lys.Qt import QtWidgets
from lys.widgets import ScientificSpinBox

from ..widgets import FEMTreeItem, ModelSelector
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
                sub.addAction(QtWidgets.QAction("Add " + s.className, self.treeWidget(), triggered=lambda x, y=s: self.__add(y())))
        return self._menu

    def __add(self, s):
        self._solvers.append(s)
        super().append(_SolverGUI(s, self))


class _SolverGUI(FEMTreeItem):
    def __init__(self, solver, parent):
        super().__init__(parent)
        self._solver = solver

    def remove(self, init):
        i = super().remove(init)
        self._solver.subSolvers.remove(self._solver.subSolvers[i])

    @property
    def name(self):
        return self._solver.className

    @property
    def widget(self):
        return self._solver.widget(self.fem())

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return self._menu


class StationarySolverWidget(QtWidgets.QWidget):
    def __init__(self, fem, solver):
        super().__init__()
        pass


class RelaxationSolverWidget(QtWidgets.QWidget):
    def __init__(self, fem, solver):
        super().__init__()
        self._fem = fem
        self._solver = solver
        self.__initlayout()

    def __initlayout(self):
        self._dt0 = ScientificSpinBox()
        self._dt0.setValue(self._solver._dt0)
        self._dt0.valueChanged.connect(self.__change)
        self._dx = ScientificSpinBox()
        self._dx.setValue(self._solver._dx)
        self._dx.valueChanged.connect(self.__change)
        self._factor = QtWidgets.QSpinBox()
        self._factor.setRange(1,1000)
        self._factor.setValue(self._solver._factor)
        self._factor.valueChanged.connect(self.__change)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Initial step (s)"), 0, 0)
        grid.addWidget(self._dt0, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Target"), 1, 0)
        grid.addWidget(self._dx, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Factor"), 2, 0)
        grid.addWidget(self._factor, 2, 1)

        self.setLayout(grid)

    def __change(self):
        self._solver._dt0 = self._dt0.value()
        self._solver._dx = self._dx.value()
        self._solver._factor = self._factor.value()


class TimeDependentSolverWidget(QtWidgets.QWidget):
    def __init__(self, fem, solver):
        super().__init__()
        self._fem = fem
        self._solver = solver
        self.__initlayout()

    def __initlayout(self):
        self._step = ScientificSpinBox()
        self._step.setValue(self._solver._step)
        self._step.valueChanged.connect(self.__change)
        self._stop = ScientificSpinBox()
        self._stop.setValue(self._solver._stop)
        self._stop.valueChanged.connect(self.__change)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Step"), 0, 0)
        grid.addWidget(self._step, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Stop"), 1, 0)
        grid.addWidget(self._stop, 1, 1)

        self.setLayout(grid)

    def __change(self):
        self._solver._step = self._step.value()
        self._solver._stop = self._stop.value()


