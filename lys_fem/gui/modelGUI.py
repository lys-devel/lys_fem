from lys.Qt import QtWidgets
from ..fem.model import models
from ..widgets import FEMTreeItem, GeometrySelector


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
        super().__init__(parent, children=[_DomainGUI(model, self), _BoundaryGUI(model, self), _InitialConditionGUI(model, self)])
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
    def __init__(self, model, fem, canvas, varDim=True, discretization=True, order=True):
        super().__init__()
        self._model = model
        self.__initLayout(model, fem, canvas, varDim, order, discretization)

    def __initLayout(self, model, fem, canvas, dim, order, disc):
        self._geom = GeometrySelector(canvas, fem, model.geometries)
        layout = QtWidgets.QGridLayout()

        i = 0
        if dim:
            self._dim = QtWidgets.QSpinBox()
            self._dim.setRange(1, 3)
            self._dim.setValue(model.variableDimension)
            self._dim.valueChanged.connect(self.__changeDim)
            layout.addWidget(QtWidgets.QLabel("Variable dimension"), i, 0)
            layout.addWidget(self._dim, i, 1)
            i += 1
        
        if order:
            self._order = QtWidgets.QSpinBox()
            self._order.setValue(model.order)
            self._order.setRange(0,100)
            self._order.valueChanged.connect(self.__changeOrder)
            layout.addWidget(QtWidgets.QLabel("Element Order"), i, 0)
            layout.addWidget(self._order, i, 1)
            i += 1

        if disc:
            self._method = QtWidgets.QComboBox()
            self._method.addItems(model.discretizationTypes)
            self._method.setCurrentText(self._model.discretization)
            self._method.currentIndexChanged.connect(self.__changeDisc)
            layout.addWidget(QtWidgets.QLabel("Discretization"), i, 0)
            layout.addWidget(self._method, i, 1)

        v1 = QtWidgets.QVBoxLayout()
        v1.addWidget(self._geom)
        v1.addLayout(layout)
        self.setLayout(v1)

    def __changeOrder(self):
        self._model.equation.order = self._order.value()

    def __changeDisc(self):
        self._model.equation.discretization = self._method.currentText()

    def __changeDim(self):
        self._model.variableDimension = self._dim.value()
