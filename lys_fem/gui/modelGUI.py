from lys.Qt import QtWidgets
from ..fem.model import models
from ..widgets import FEMTreeItem


class ModelTree(FEMTreeItem):
    def __init__(self, obj, canvas):
        super().__init__(fem=obj, canvas=canvas)
        self.setModels(obj.models)

    def setModels(self, models):
        self.clear()
        self._models = models
        for m in models:
            super().append(_ModelGUI(m, self))

    def append(self, m):
        self._models.append(m)
        super().append(_ModelGUI(m, self))

    def remove(self, m):
        i = super().remove(m)
        self._models.remove(self._models[i])

    @ property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        for group, ms in models.items():
            sub = self._menu.addMenu(group)
            for m in ms:
                sub.addAction(QtWidgets.QAction("Add " + m.className, self.treeWidget(), triggered=lambda x, y=m: self.append(y())))
        return self._menu


class _ModelGUI(FEMTreeItem):
    def __init__(self, model, parent):
        super().__init__(parent, children=[_EquationGUI(model, self), _DomainGUI(model, self), _BoundaryGUI(model, self), _InitialConditionGUI(model, self)])
        self._model = model

    @ property
    def name(self):
        return self._model.className

    @ property
    def widget(self):
        return self._model.widget(self.fem(), self.canvas())

    @property
    def menu(self):
        menu = self.parent.menu
        menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return menu


class _EquationGUI(FEMTreeItem):
    def __init__(self, model, parent):
        super().__init__(parent, children=[_EquationCondition(d, self) for d in model.equations])
        self._model = model

    def append(self, type):
        eq = type()
        self._model.equations.append(eq)
        super().append(_EquationCondition(eq, self))

    def remove(self, init):
        i = super().remove(init)
        self._model.equations.remove(self._model.equations[i])

    @ property
    def name(self):
        return "Equations"

    @ property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        for i in self._model.equationTypes:
            self._menu.addAction(QtWidgets.QAction("Add " + i.className, self.treeWidget(), triggered=lambda x, y=i: self.append(y)))
        return self._menu


class _DomainGUI(FEMTreeItem):
    def __init__(self, model, parent):
        super().__init__(parent, children=[_DomainCondition(d, self) for d in model.domainConditions])
        self._model = model

    def append(self, type):
        cond = type.default(self.fem(), self._model)
        self._model.domainConditions.append(cond)
        super().append(_DomainCondition(cond, self))

    def remove(self, init):
        i = super().remove(init)
        self._model.domainConditions.remove(self._model.domainConditions[i])

    @ property
    def name(self):
        return "Domain Conditions"

    @ property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        for i in self._model.domainConditionTypes:
            self._menu.addAction(QtWidgets.QAction("Add " + i.className, self.treeWidget(), triggered=lambda x, y=i: self.append(y)))
        return self._menu


class _BoundaryGUI(FEMTreeItem):
    def __init__(self, model, parent):
        super().__init__(parent, children=[_BoundaryCondition(b, self) for b in model.boundaryConditions])
        self._model = model

    def append(self, type):
        cond = type.default(self.fem(), self._model)
        self._model.boundaryConditions.append(cond)
        super().append(_BoundaryCondition(cond, self))

    def remove(self, init):
        i = super().remove(init)
        self._model.boundaryConditions.remove(self._model.boundaryConditions[i])

    @ property
    def name(self):
        return "Boundary Conditions"

    @ property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        for i in self._model.boundaryConditionTypes:
            self._menu.addAction(QtWidgets.QAction("Add " + i.className, self.treeWidget(), triggered=lambda x, y=i: self.append(y)))
        return self._menu


class _InitialConditionGUI(FEMTreeItem):
    def __init__(self, model, parent):
        super().__init__(parent, children=[_InitialCondition(i, self) for i in model.initialConditions])
        self._model = model

    def append(self, type):
        cond = type.default(self.fem(), self._model)
        self._model.initialConditions.append(cond)
        super().append(_InitialCondition(cond, self))

    def remove(self, init):
        i = super().remove(init)
        self._model.initialConditions.remove(self._model.initialConditions[i])

    @ property
    def name(self):
        return "Initial Conditions"

    @ property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        for i in self._model.initialConditionTypes:
            self._menu.addAction(QtWidgets.QAction("Add " + i.className, self.treeWidget(), triggered=lambda x, y=i: self.append(y)))
        return self._menu

class _EquationCondition(FEMTreeItem):
    def __init__(self, cond, parent):
        super().__init__(parent)
        self._cond = cond

    @ property
    def name(self):
        return self._cond.objName

    @ property
    def widget(self):
        return self._cond.widget(self.fem(), self.canvas())

    @ property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return self._menu


class _DomainCondition(FEMTreeItem):
    def __init__(self, cond, parent):
        super().__init__(parent)
        self._cond = cond

    @ property
    def name(self):
        return self._cond.objName

    @ property
    def widget(self):
        return self._cond.widget(self.fem(), self.canvas())

    @ property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return self._menu


class _BoundaryCondition(FEMTreeItem):
    def __init__(self, bdr, parent):
        super().__init__(parent)
        self._bdr = bdr

    @ property
    def name(self):
        return self._bdr.objName

    @ property
    def widget(self):
        return self._bdr.widget(self.fem(), self.canvas())

    @ property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return self._menu


class _InitialCondition(FEMTreeItem):
    def __init__(self, init, parent):
        super().__init__(parent)
        self._init = init

    @ property
    def name(self):
        return self._init.objName

    @ property
    def widget(self):
        return self._init.widget(self.fem(), self.canvas())

    @ property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return self._menu


class FEMModelWidget(QtWidgets.QWidget):
    def __init__(self, model):
        super().__init__()
        self.__model = model
        self.__initlayout(model)

    def __initlayout(self, model):
        self._method = MethodComboBox(model)
        self._dim = QtWidgets.QSpinBox()
        self._dim.setRange(1, 3)
        self._dim.setValue(model.variableDimension())
        self._dim.valueChanged.connect(self.__changeDim)
        self._order = QtWidgets.QSpinBox()
        self._order.setValue(model.order)
        self._order.setRange(0,100)
        self._order.valueChanged.connect(self.__change)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(QtWidgets.QLabel("Variable dimension"), 0, 0)
        layout.addWidget(QtWidgets.QLabel("Element Order"), 1, 0)
        layout.addWidget(QtWidgets.QLabel("Discretization"), 2, 0)
        layout.addWidget(self._dim, 0, 1)
        layout.addWidget(self._order, 0, 1)
        layout.addWidget(self._method, 2, 1)
        self.setLayout(layout)

    def __changeDim(self, value):
        self.__model.setVariableDimension(self._dim.value())

    def __change(self):
        self.__model.order = self._order.value()


class FEMFixedModelWidget(QtWidgets.QWidget):
    def __init__(self, model):
        super().__init__()
        self.__initLayout(model)
        self._model = model

    def __initLayout(self, model):
        self._method = MethodComboBox(model)
        self._order = QtWidgets.QSpinBox()
        self._order.setValue(model.order)
        self._order.setRange(0,100)
        self._order.valueChanged.connect(self.__change)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(QtWidgets.QLabel("Discretization"), 0, 0)
        layout.addWidget(QtWidgets.QLabel("Element Order"), 1, 0)
        layout.addWidget(self._method, 0, 1)
        layout.addWidget(self._order, 1, 1)
        self.setLayout(layout)

    def __change(self):
        self._model.order = self._order.value()


class MethodComboBox(QtWidgets.QComboBox):
    def __init__(self, model):
        super().__init__()
        self._model = model
        self.addItems(model.discretizationTypes)
        self.setCurrentText(self._model.discretization)
        self.currentIndexChanged.connect(self.__change)

    def __change(self):
        self._model._disc = self.currentText()
