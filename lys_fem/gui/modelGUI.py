from lys.Qt import QtWidgets
from ..fem.model import models
from ..widgets import FEMTreeItem


class ModelTree(FEMTreeItem):
    def __init__(self, obj, canvas):
        super().__init__(fem=obj, canvas=canvas)
        self.setModels(obj.models)

    @property
    def children(self):
        return self._children

    def setModels(self, models):
        self._models = models
        self._children = [_ModelGUI(m, self) for m in models]

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        for group, ms in models.items():
            sub = self._menu.addMenu(group)
            for m in ms:
                sub.addAction(QtWidgets.QAction("Add " + m.name, self.treeWidget(), triggered=lambda x, y=m: self.__add(y)))
        return self._menu

    def __add(self, model):
        self.beginInsertRow(len(self._children))
        m = model()
        self._models.append(m)
        self._children.append(_ModelGUI(m, self))
        self.endInsertRow()


class _ModelGUI(FEMTreeItem):
    def __init__(self, model, parent):
        super().__init__(parent)
        self._model = model
        self._children = [_DomainGUI(model, self), _BoundaryGUI(model, self), _InitialConditionGUI(model, self), _SolverGUI(model, self)]

    @property
    def name(self):
        return self._model.name

    @property
    def children(self):
        return self._children

    @property
    def widget(self):
        return self._model.widget(self.fem(), self.canvas())


class _DomainGUI(FEMTreeItem):
    def __init__(self, model, parent):
        super().__init__(parent)
        self._model = model
        self._children = [_DomainCondition(d, self) for d in model.domainConditions]

    @property
    def name(self):
        return "Domain Conditions"

    @property
    def children(self):
        return self._children

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        for i in self._model.domainConditionTypes:
            self._menu.addAction(QtWidgets.QAction("Add " + i.name, self.treeWidget(), triggered=lambda x, y=i: self.__add(y)))
        return self._menu

    def __add(self, type):
        self.beginInsertRow(len(self._children))
        i = self._model.addDomainCondition(type)
        self._children.append(_DomainCondition(i, self))
        self.endInsertRow()

    def remove(self, init):
        i = self._children.index(init)
        self.beginRemoveRow(i)
        self._model.domainConditions.remove(self._model.domainConditions[i])
        self._children.remove(init)
        self.endRemoveRow()


class _BoundaryGUI(FEMTreeItem):
    def __init__(self, model, parent):
        super().__init__(parent)
        self._model = model
        self._children = [_BoundaryCondition(b, self) for b in model.boundaryConditions]

    @property
    def name(self):
        return "Boundary Conditions"

    @property
    def children(self):
        return self._children

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        for i in self._model.boundaryConditionTypes:
            self._menu.addAction(QtWidgets.QAction("Add " + i.name, self.treeWidget(), triggered=lambda x, y=i: self.__add(y)))
        return self._menu

    def __add(self, type):
        self.beginInsertRow(len(self._children))
        i = self._model.addBoundaryCondition(type)
        self._children.append(_BoundaryCondition(i, self))
        self.endInsertRow()

    def remove(self, init):
        i = self._children.index(init)
        self.beginRemoveRow(i)
        self._model.boundaryConditions.remove(self._model.boundaryConditions[i])
        self._children.remove(init)
        self.endRemoveRow()


class _InitialConditionGUI(FEMTreeItem):
    def __init__(self, model, parent):
        super().__init__(parent)
        self._model = model
        self._children = [_InitialCondition(i, self) for i in model.initialConditions]

    @property
    def name(self):
        return "Initial Conditions"

    @property
    def children(self):
        return self._children

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        for i in self._model.initialConditionTypes:
            self._menu.addAction(QtWidgets.QAction("Add " + i.name, self.treeWidget(), triggered=lambda x, y=i: self.__add(y)))
        return self._menu

    def __add(self, type):
        self.beginInsertRow(len(self._children))
        i = self._model.addInitialCondition(type)
        self._children.append(_InitialCondition(i, self))
        self.endInsertRow()

    def remove(self, init):
        i = self._children.index(init)
        self.beginRemoveRow(i)
        self._model.initialConditions.remove(self._model.initialConditions[i])
        self._children.remove(init)
        self.endRemoveRow()


class _SolverGUI(FEMTreeItem):
    def __init__(self, model, parent):
        super().__init__(parent)
        self._model = model

    @property
    def name(self):
        return "Solver"

    @property
    def widget(self):
        return self._model.solverWidget()


class _DomainCondition(FEMTreeItem):
    def __init__(self, cond, parent):
        super().__init__(parent)
        self._cond = cond

    @property
    def name(self):
        return self._cond.objName

    @property
    def widget(self):
        return self._cond.widget(self.fem(), self.canvas())

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return self._menu


class _BoundaryCondition(FEMTreeItem):
    def __init__(self, bdr, parent):
        super().__init__(parent)
        self._bdr = bdr

    @property
    def name(self):
        return self._bdr.objName

    @property
    def widget(self):
        return self._bdr.widget(self.fem(), self.canvas())

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return self._menu


class _InitialCondition(FEMTreeItem):
    def __init__(self, init, parent):
        super().__init__(parent)
        self._init = init

    @property
    def name(self):
        return self._init.objName

    @property
    def widget(self):
        return self._init.widget(self.fem(), self.canvas())

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return self._menu
