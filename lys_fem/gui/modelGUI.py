from lys.Qt import QtWidgets
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


class _ModelGUI(FEMTreeItem):
    def __init__(self, model, parent):
        super().__init__(parent)
        self._model = model
        self._children = [_DomainGUI(model, self), _BoundaryGUI(model, self), _InitialConditionGUI(model, self)]

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
        self._children = []

    @property
    def name(self):
        return "Domains"

    @property
    def children(self):
        return self._children


class _BoundaryGUI(FEMTreeItem):
    def __init__(self, model, parent):
        super().__init__(parent)
        self._children = []

    @property
    def name(self):
        return "Boundary Conditions"

    @property
    def children(self):
        return self._children


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
        if not self._default:
            self._menu = QtWidgets.QMenu()
            self._menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
            return self._menu
