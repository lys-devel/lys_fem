from lys.Qt import QtWidgets
from lys.widgets import ScientificSpinBox

from ..widgets import FEMTreeItem
from ..fem.solver import solvers, SolverStep


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
        self.setSteps(solver)

    def setSteps(self, solver):
        self.clear()
        for step in solver.steps:
            super().append(_SolverStepGUI(solver, step, self))

    def __add(self):
        step = SolverStep()
        self._solver.steps.append(step)
        super().append(_SolverStepGUI(self._solver, step, self))

    def remove(self, item):
        i = super().remove(item)
        self._solver.steps.remove(self._solver.steps[i])

    @property
    def name(self):
        return self._solver.className

    @property
    def widget(self):
        return self._solver.widget(self.fem())

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Add Step", self.treeWidget(), triggered=self.__add))
        self._menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return self._menu


class _SolverStepGUI(FEMTreeItem):
    def __init__(self, solver, step, parent):
        super().__init__(parent)
        self._solver = solver
        self._step = step

    @property
    def name(self):
        return "Step " + str(self._solver.steps.index(self._step)+1)

    @property
    def widget(self):
        return _SolverStepWidget(self._step)
    
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
        self._expr = QtWidgets.QLineEdit()
        self._expr.setText(self._solver.diff_expr)
        self._expr.setPlaceholderText("Auto if blank")
        self._expr.textChanged.connect(self.__change)

        self._dx = ScientificSpinBox()
        self._dx.setValue(self._solver._dx)
        self._dx.valueChanged.connect(self.__change)

        self._dt0 = ScientificSpinBox()
        self._dt0.setValue(self._solver._dt0)
        self._dt0.valueChanged.connect(self.__change)

        self._maxstep = ScientificSpinBox()
        self._maxstep.setValue(self._solver._maxStep)
        self._maxstep.valueChanged.connect(self.__change)

        self._factor = QtWidgets.QSpinBox()
        self._factor.setRange(1,1000)
        self._factor.setValue(self._solver._factor)
        self._factor.valueChanged.connect(self.__change)

        self._maxiter = QtWidgets.QSpinBox()
        self._maxiter.setRange(1,1000)
        self._maxiter.setValue(self._solver._maxiter)
        self._maxiter.valueChanged.connect(self.__change)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Expression of x"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("Target dx"), 1, 0)
        grid.addWidget(QtWidgets.QLabel("Initial step (s)"), 2, 0)
        grid.addWidget(QtWidgets.QLabel("Max step factor"), 3, 0)
        grid.addWidget(QtWidgets.QLabel("Stop Factor"), 4, 0)
        grid.addWidget(QtWidgets.QLabel("Max Iteration"), 5, 0)
        grid.addWidget(self._expr, 0, 1)
        grid.addWidget(self._dx, 1, 1)
        grid.addWidget(self._dt0, 2, 1)
        grid.addWidget(self._maxstep, 3, 1)
        grid.addWidget(self._factor, 4, 1)
        grid.addWidget(self._maxiter, 5, 1)

        self.setLayout(grid)

    def __change(self):
        self._solver._diff_expr = self._expr.text()
        self._solver._dt0 = self._dt0.value()
        self._solver._dx = self._dx.value()
        self._solver._factor = self._factor.value()
        self._solver._maxStep = self._maxstep.value()
        self._solver._maxiter = self._maxiter.value()


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


class _SolverStepWidget(QtWidgets.QWidget):
    def __init__(self, step):
        super().__init__()
        self._step = step
        self.__initLayout(step)

    def __initLayout(self, step):
        g0 = self.__initSolvers(step)
        g1 = self.__initVars(step)
        g2 = self.__initNewton(step)
        g3 = self.__initDeform(step)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(g0)
        layout.addWidget(g1)
        layout.addWidget(g2)
        layout.addWidget(g3)

        self.setLayout(layout)

    def __initNewton(self, step):
        self._maxiter = QtWidgets.QSpinBox()
        self._maxiter.setRange(1, 100000)
        self._maxiter.setValue(step.newton_maxiter)
        self._maxiter.valueChanged.connect(self.__changeNewton)
        self._eps = ScientificSpinBox()
        self._eps.setValue(step.newton_eps)
        self._eps.valueChanged.connect(self.__changeNewton)
        self._damp = ScientificSpinBox()
        self._damp.setValue(step.newton_damping)
        self._damp.valueChanged.connect(self.__changeNewton)

        g = QtWidgets.QGridLayout()
        g.addWidget(QtWidgets.QLabel("Damping factor"), 0, 0)
        g.addWidget(QtWidgets.QLabel("Maximum iteration"), 1, 0)
        g.addWidget(QtWidgets.QLabel("Relative tolerance"), 2, 0)
        g.addWidget(self._damp, 0, 1)
        g.addWidget(self._maxiter, 1, 1)
        g.addWidget(self._eps, 2, 1)

        g1 = QtWidgets.QGroupBox("Newton nonlinear solver")
        g1.setLayout(g)
        return g1

    def __changeNewton(self):
        self._step._damping = self._damp.value()
        self._step._eps = self._eps.value()
        self._step._maxiter = self._maxiter.value()

    def __initSolvers(self, step):
        self._solver = QtWidgets.QComboBox()
        self._solver.addItems(["pardiso", "pardisospd", "masterinverse", "sparsecholesky", "umfpack", "CG", "GMRES"])
        self._solver.setCurrentText(step.solver)
        self._solver.currentTextChanged.connect(self.__changeSolvers)

        self._prec = QtWidgets.QComboBox()
        self._prec.addItems(["None", "local", "direct", "h1amg", "bddc"])
        if step.preconditioner is not None:
            self._prec.setCurrentText(step.preconditioner)
        self._prec.currentTextChanged.connect(self.__changeSolvers)

        self._sym = QtWidgets.QCheckBox("Symmetric")
        self._sym.setChecked(step.symmetric)
        self._sym.toggled.connect(self.__changeSolvers)
        self._cond = QtWidgets.QCheckBox("Static condensation")
        self._cond.setChecked(step.condensation)
        self._cond.toggled.connect(self.__changeSolvers)

        g = QtWidgets.QGridLayout()
        g.addWidget(QtWidgets.QLabel("Linear solver"), 0, 0)
        g.addWidget(self._solver, 0, 1)
        g.addWidget(QtWidgets.QLabel("Preconditioner"), 1, 0)
        g.addWidget(self._prec, 1, 1)
        g.addWidget(self._sym, 2, 0)
        g.addWidget(self._cond, 2, 1)

        g1 = QtWidgets.QGroupBox("Solvers")
        g1.setLayout(g)
        return g1

    def __changeSolvers(self):
        self._step._solver = self._solver.currentText()
        if self._prec.currentText() == "None":
            self._step._prec = None
        else:
            self._step._prec = self._prec.currentText()
        self._step._sym = self._sym.isChecked()
        self._step._cond = self._cond.isChecked()

    def __initVars(self, step):
        self._varType = QtWidgets.QComboBox()
        self._varType.addItems(["All", "Custom"])
        self._vars = QtWidgets.QLineEdit()
        self._vars.setPlaceholderText("u,T,E")
        if step.variables is not None:
            self._varType.setCurrentText("Custom")
            self._vars.setText(", ".join(step.variables))
        else:
            self._vars.setEnabled(False)
        self._varType.currentIndexChanged.connect(self.__changeVars)
        self._vars.textChanged.connect(self.__changeVars)

        h1 = QtWidgets.QHBoxLayout()
        h1.addWidget(QtWidgets.QLabel("Variable names"))
        h1.addWidget(self._vars)

        v1 = QtWidgets.QVBoxLayout()
        v1.addWidget(self._varType)
        v1.addLayout(h1)

        g1 = QtWidgets.QGroupBox("Variables to be solved")
        g1.setLayout(v1)
        return g1

    def __changeVars(self):
        self._vars.setEnabled(self._varType.currentText() != "All")
        if self._varType.currentText() == "all":
            self._step._vars = None
        else:
            self._step._vars = self._vars.text().replace(" ", "").split(",")

    def __initDeform(self, step):
        self._deform_var = QtWidgets.QLineEdit()
        self._deform_var.setPlaceholderText("u")
        self._deform_var.textChanged.connect(self.__changeDeform)

        h1 = QtWidgets.QHBoxLayout()
        h1.addWidget(QtWidgets.QLabel("Variable name"))
        h1.addWidget(self._deform_var)

        self._deform = QtWidgets.QGroupBox("Apply Deformation")
        self._deform.setCheckable(True)
        self._deform.setLayout(h1)
        if step.deformation is None:
            self._deform.setChecked(False)
        else:
            self._deform.setChecked(True)
            self._deform_var.setText(self._deform)
        self._deform.toggled.connect(self.__changeDeform)
        return self._deform
    
    def __changeDeform(self):
        if self._deform.isChecked():
            self._step._deform = self._deform_var.text()
        else:
            self._step._deform = None
