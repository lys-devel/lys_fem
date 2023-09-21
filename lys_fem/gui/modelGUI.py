from lys.Qt import QtWidgets, QtCore
from ..widgets import FEMTreeItem, GeometrySelector


class ModelTree(FEMTreeItem):
    def __init__(self, obj, canvas):
        super().__init__(fem=obj, canvas=canvas)
        self.setModels(obj.models)

    @property
    def children(self):
        return self._children

    def setModels(self, models):
        self._models = models
        self._children = [_ElasticGUI(m, self) for m in models]


class _ElasticGUI(FEMTreeItem):
    def __init__(self, model, parent):
        super().__init__(parent)
        self._model = model
        self._children = [_ElasticDomain(model, self), _ElasticBoundary(model, self), _InitialConditions(model, self)]

    @property
    def name(self):
        return "Elasticity"

    @property
    def children(self):
        return self._children

    @property
    def widget(self):
        return _ElasticityWidget(self._model)


class _ElasticityWidget(QtWidgets.QWidget):
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


class _ElasticDomain(FEMTreeItem):
    def __init__(self, model, parent):
        super().__init__(parent)
        self._children = []

    @property
    def name(self):
        return "Domains"

    @property
    def children(self):
        return self._children


class _ElasticBoundary(FEMTreeItem):
    def __init__(self, model, parent):
        super().__init__(parent)
        self._children = []

    @property
    def name(self):
        return "Boundary Conditions"

    @property
    def children(self):
        return self._children


class _InitialConditions(FEMTreeItem):
    def __init__(self, model, parent):
        super().__init__(parent)
        self._model = model
        self._children = [_InitialCondition(i, self) for i in model.initialConditions]
        self._children[0].setDefault(True)

    @property
    def name(self):
        return "Initial Conditions"

    @property
    def children(self):
        return self._children

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Add initial condition", self.treeWidget(), triggered=self.__add))
        return self._menu

    def __add(self):
        self.beginInsertRow(len(self._children))
        i = self._model.addInitialCondition()
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
        self._default = False

    def setDefault(self, b):
        self._default = b

    @property
    def name(self):
        return self._init.name

    @property
    def widget(self):
        w = _InitialConditionWidget(self._init, self.fem(), self.canvas(), self._default)
        w.selectionChanged.connect(self.__domainChanged)
        w.valueChanged.connect(self.__valueChanged)
        return w

    def __domainChanged(self, selected):
        self._init.domains = selected

    def __valueChanged(self, value):
        self._init.values = value

    @property
    def menu(self):
        if not self._default:
            self._menu = QtWidgets.QMenu()
            self._menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
            return self._menu


class _InitialConditionWidget(QtWidgets.QWidget):
    selectionChanged = QtCore.pyqtSignal(object)
    valueChanged = QtCore.pyqtSignal(object)

    def __init__(self, init, fem, canvas, default):
        super().__init__()
        self._init = init
        self.__initlayout(fem, canvas, default)

    def __initlayout(self, fem, canvas, default):
        dim = len(self._init.values)
        if default:
            self._selector = GeometrySelector("Target Domains", fem.dimension, canvas, fem, self._init.domains, acceptedTypes=["All"])
        else:
            self._selector = GeometrySelector("Target Domains", fem.dimension, canvas, fem, self._init.domains)
        self._selector.selectionChanged.connect(self.selectionChanged)

        self._widgets = [QtWidgets.QLineEdit() for _ in range(dim)]
        for wid, val in zip(self._widgets, self._init.values):
            wid.setText(val)
            wid.textChanged.connect(self.__valueChanged)
        grid = QtWidgets.QGridLayout()
        for i, wid in enumerate(self._widgets):
            grid.addWidget(QtWidgets.QLabel("Dim " + str(i)), i, 0, 1, 1)
            grid.addWidget(wid, i, 1, 1, 3)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._selector)
        layout.addLayout(grid)
        self.setLayout(layout)

    def __valueChanged(self):
        self.valueChanged.emit([w.text() for w in self._widgets])
