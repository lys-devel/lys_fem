from lys.Qt import QtWidgets
from ..widgets import FEMTreeItem, ModelSelector
from ..fem.solver import solvers, subSolvers


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
        super().__init__(parent)
        self._solver = solver

    def remove(self, init):
        i = super().remove(init)
        self._solver.subSolvers.remove(self._solver.subSolvers[i])

    @property
    def name(self):
        return self._solver.name

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
        self._fem = fem
        self._solver = solver
        self.__initLayout()
        self._update()

    def __initLayout(self):
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Model", "FEM Solver"])
        self._tree.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QPushButton("Add", clicked=self.__add))
        h.addWidget(QtWidgets.QPushButton("Remove", clicked=self.__remove))

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self._tree)
        layout.addLayout(h)
        self.setLayout(layout)

    def _update(self):
        self._tree.clear()
        for m, s in zip(self._solver.models, self._solver.subSolvers):
            self._tree.addTopLevelItem(QtWidgets.QTreeWidgetItem([m.name, s.name]))

    def __add(self):
        d = _AddModelDialog(self, self._fem)
        if d.exec_():
            self._solver.addModel(d.model, d.femSolver)
        self._update()

    def __remove(self):
        index = self._tree.currentIndex().row()
        if index != -1:
            self._solver.remove(index)
            self._update()


class TimeDependentSolverWidget(QtWidgets.QWidget):
    def __init__(self, fem, solver):
        super().__init__()
        self._fem = fem
        self._solver = solver
        self.__initlayout()
        self._update()

    def __initlayout(self):
        self._step = QtWidgets.QDoubleSpinBox()
        self._step.setRange(0, 100000000)
        self._step.setValue(self._solver._step)
        self._step.setDecimals(6)
        self._step.valueChanged.connect(self.__change)
        self._stop = QtWidgets.QDoubleSpinBox()
        self._stop.setRange(0, 1000000000)
        self._stop.setDecimals(6)
        self._stop.setValue(self._solver._stop)
        self._stop.valueChanged.connect(self.__change)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Step"), 0, 0)
        grid.addWidget(self._step, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Stop"), 1, 0)
        grid.addWidget(self._stop, 1, 1)

        self._tree = QtWidgets.QTreeWidget()
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["Model", "Time Dep. Solver", "FEM Solver"])
        self._tree.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QPushButton("Add", clicked=self.__add))
        h.addWidget(QtWidgets.QPushButton("Remove", clicked=self.__remove))

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(grid)
        layout.addWidget(self._tree)
        layout.addLayout(h)
        self.setLayout(layout)

    def _update(self):
        self._tree.clear()
        for m, s in zip(self._solver.models, self._solver.subSolvers):
            self._tree.addTopLevelItem(QtWidgets.QTreeWidgetItem([m.name, s.name, s.femSolver.name]))

    def __change(self):
        self._solver._step = self._step.value()
        self._solver._stop = self._stop.value()

    def __add(self):
        d = _AddModelDialog(self, self._fem, tdep=True)
        if d.exec_():
            self._solver.addModel(d.model, d.tSolver)
        self._update()

    def __remove(self):
        index = self._tree.currentIndex().row()
        if index != -1:
            self._solver.remove(index)
            self._update()


class _AddModelDialog(QtWidgets.QDialog):
    def __init__(self, parent, fem, tdep=False):
        super().__init__(parent)
        self._fem = fem
        self.__initLayout(fem, tdep)

    def __initLayout(self, fem, tdep):
        self._model = ModelSelector(fem)
        self._femSolver = QtWidgets.QComboBox()
        self._femSolver.addItems(list(subSolvers.keys()))
        self._tSolver = QtWidgets.QComboBox()
        self._tSolver.addItems(list(tSolvers.keys()))

        g = QtWidgets.QGridLayout()
        g.addWidget(QtWidgets.QLabel("Model"), 0, 0)
        g.addWidget(self._model, 0, 1)
        if tdep:
            g.addWidget(QtWidgets.QLabel("Time Dependent Solver"), 1, 0)
            g.addWidget(self._tSolver, 1, 1)
            g.addWidget(QtWidgets.QLabel("FEM Solver"), 2, 0)
            g.addWidget(self._femSolver, 2, 1)
        else:
            g.addWidget(QtWidgets.QLabel("FEM Solver"), 1, 0)
            g.addWidget(self._femSolver, 1, 1)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QPushButton("O K", clicked=self.accept))
        h.addWidget(QtWidgets.QPushButton("CANCEL", clicked=self.reject))

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(g)
        layout.addLayout(h)
        self.setLayout(layout)

    @property
    def femSolver(self):
        return subSolvers[self._femSolver.currentText()]()

    @property
    def tSolver(self):
        return tSolvers[self._tSolver.currentText()](self.femSolver)

    @property
    def model(self):
        return self._model.getSelectedModel()
