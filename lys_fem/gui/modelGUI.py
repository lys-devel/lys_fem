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
                sub.addAction(QtWidgets.QAction("Add " + m.name, self.treeWidget(), triggered=lambda x, y=m: self.append(y())))
        return self._menu


class _ModelGUI(FEMTreeItem):
    def __init__(self, model, parent):
        super().__init__(parent, children=[_DomainGUI(model, self), _BoundaryGUI(model, self), _InitialConditionGUI(model, self)])
        self._model = model

    @ property
    def name(self):
        return self._model.name

    @ property
    def widget(self):
        return self._model.widget(self.fem(), self.canvas())

    @property
    def menu(self):
        menu = self.parent.menu
        menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return menu


class _DomainGUI(FEMTreeItem):
    def __init__(self, model, parent):
        super().__init__(parent, children=[_DomainCondition(d, self) for d in model.domainConditions])
        self._model = model

    def append(self, type):
        cond = type.default(self._model)
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
        i = self._model.addBoundaryCondition(type)
        super().append(_BoundaryCondition(i, self))

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
            self._menu.addAction(QtWidgets.QAction("Add " + i.name, self.treeWidget(), triggered=lambda x, y=i: self.append(y)))
        return self._menu


class _InitialConditionGUI(FEMTreeItem):
    def __init__(self, model, parent):
        super().__init__(parent, children=[_InitialCondition(i, self) for i in model.initialConditions])
        self._model = model

    def append(self, type):
        i = self._model.addInitialCondition(type)
        super().append(_InitialCondition(i, self))

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
            self._menu.addAction(QtWidgets.QAction("Add " + i.name, self.treeWidget(), triggered=lambda x, y=i: self.append(y)))
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
        self._model = model
        self.__initlayout()

    def __initlayout(self):
        self._dim = QtWidgets.QSpinBox()
        self._dim.setRange(1, 3)
        self._dim.setValue(self._model.variableDimension())
        self._dim.valueChanged.connect(self.__changeDim)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Variable dimension"))
        layout.addWidget(self._dim)
        self.setLayout(layout)

    def __changeDim(self, value):
        self._model.setVariableDimension(self._dim.value())


class FEMFixedModelWidget(QtWidgets.QWidget):
    def __init__(self, model):
        super().__init__()
        self._model = model
