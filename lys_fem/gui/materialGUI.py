import numpy as np
from lys.Qt import QtWidgets, QtCore, QtGui

from ..widgets import FEMTreeItem, GeometrySelector, MatrixFunctionWidget
from ..fem import Material, materialParameters, UserDefinedParameters, Coef


class MaterialTree(FEMTreeItem):
    def __init__(self, parent, materials):
        super().__init__(parent)
        self.setMaterials(materials)

    @property
    def name(self):
        return "Materials"

    def setMaterials(self, materials):
        self.clear()
        self._materials = materials
        for m in self._materials:
            super().append(_MaterialGUI(m, self))

    def append(self, material=None):
        if not isinstance(material, Material):
            material = Material()
        self._materials.append(material)
        super().append(_MaterialGUI(material, self))

    def remove(self, material):
        i = super().remove(material)
        self._materials.remove(self._materials[i])

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Add new material", self.treeWidget(), triggered=self.append))
        return self._menu


class _MaterialGUI(FEMTreeItem):
    def __init__(self, material, parent):
        super().__init__(parent, children=[_ParameterGUI(self, p) for p in material])
        self._material = material
        self._default = False

    def setDefault(self, b):
        self._default = b

    def append(self, p):
        self._material.parameters.append(p)
        super().append(_ParameterGUI(self, p))

    def remove(self, param):
        i = super().remove(param)
        self._material.parameters.remove(self._material[i])

    @property
    def name(self):
        return self._material.objName

    @property
    def widget(self):
        return _MaterialWidget(self.canvas(), self.fem(), self._material)

    @property
    def menu(self):
        menu = self.parent.menu
        if not self._default:
            menu.addAction(QtWidgets.QAction("Remove material", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        menu.addSeparator()
        for grp, params in materialParameters.items():
            sub = menu.addMenu(grp)
            for p in params:
                sub.addAction(QtWidgets.QAction(p.name, self.treeWidget(), triggered=lambda b, x=p: self.append(x())))
        return menu


class _ParameterGUI(FEMTreeItem):
    def __init__(self, parent, param):
        super().__init__(parent=parent)
        self._param = param

    @property
    def name(self):
        return self._param.name

    @property
    def menu(self):
        menu = QtWidgets.QMenu()
        menu.addAction(QtWidgets.QAction("Remove parameters", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return menu

    @property
    def widget(self):
        if isinstance(self._param, UserDefinedParameters):
            return _UserDefinedParametersWidget(self._param)
        else:
            return _ParameterWidget(self._param)


class _MaterialWidget(QtWidgets.QWidget):
    def __init__(self, canvas, fem, mat):
        super().__init__()
        self._material = mat
        self.__initlayout(canvas, fem, mat)

    def __initlayout(self, canvas, fem, mat):
        domain = GeometrySelector(canvas, fem, mat.geometries)

        self._grp = QtWidgets.QGroupBox("Define parameters on material coordinate", checkable=True)
        if mat.coordinate is not None:
            m = mat.coordinate
            self._grp.setChecked(True)
        else:
            m = np.eye(3)
            self._grp.setChecked(False)
        self._grp.toggled.connect(self.__changed)
        self._xyz = MatrixFunctionWidget(m, label="Material coordinate XYZ")
        self._xyz.valueChanged.connect(self.__changed)
        l1 = QtWidgets.QHBoxLayout()
        l1.addWidget(self._xyz)
        self._grp.setLayout(l1)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(domain)
        layout.addWidget(self._grp)
        self.setLayout(layout)

    def __changed(self):
        if self._grp.isChecked():
            self._material.coordinate = self._xyz.value()
        else:
            self._material.coordinate = None


class _ParameterWidget(QtWidgets.QTreeWidget):
    def __init__(self, param, parent=None):
        super().__init__(parent)
        self._param = param
        self.setColumnCount(2)
        self.setHeaderLabels(["Symbol", "Description"])
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._buildContextMenu)
        self.__initLayout(param)

    def __initLayout(self, param):
        self._widgets = []
        for key, c in param.items():
            if c.valid:
                self.__addItem(key, c)

    def __addItem(self, key, coef):
        child = QtWidgets.QTreeWidgetItem(["", ""])
        parent = QtWidgets.QTreeWidgetItem([key, coef.description])
        parent.addChild(child)
        widget = coef.widget()
        self.addTopLevelItem(parent)
        self.setIndexWidget(self.indexFromItem(child, column=1), widget)
        self._widgets.append(widget)

    def _buildContextMenu(self):
        self._menu = QtWidgets.QMenu()
        sub = self._menu.addMenu("Add")
        for key, coef in self._param.items():
            if not coef.valid:
                sub.addAction(QtWidgets.QAction(key+": "+coef.description, self, triggered=lambda x, y=key,z=coef: self.__add(y,z)))
        self._menu.addAction(QtWidgets.QAction("Remove", self, triggered=self.__remove))
        self._menu.exec_(QtGui.QCursor.pos())

    def __add(self, key, coef):
        coef.setDefault()
        self.__addItem(key, coef)

    def __remove(self):
        index = self.indexFromItem(self.currentItem())
        if not index.isValid():
            return
        if index.parent().isValid():
            index=index.parent()
        key = index.data(QtCore.Qt.DisplayRole)
        setattr(self._param, key, None)
        self.takeTopLevelItem(index.row())


class _UserDefinedParametersWidget(QtWidgets.QTreeWidget):
    def __init__(self, param, parent=None):
        super().__init__(parent)
        self._param = param
        self._widgets = {}
        self.setColumnCount(1)
        self.setHeaderLabels(["Symbol"])
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._buildContextMenu)
        self.__initLayout()
        self.itemClicked.connect(self.__selected)
        self.itemChanged.connect(self.__changed)

    def __initLayout(self):
        for key, item in vars(self._param).items():
            self.__addItem(key, item)

    def _buildContextMenu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Add scalar", self, triggered=lambda: self.__add("scalar")))
        self._menu.addAction(QtWidgets.QAction("Add vector", self, triggered=lambda: self.__add("vector")))
        self._menu.addAction(QtWidgets.QAction("Add matrix", self, triggered=lambda: self.__add("matrix")))
        self._menu.addAction(QtWidgets.QAction("Remove", self, triggered=self.__remove))
        self._menu.exec_(QtGui.QCursor.pos())

    def __addItem(self, key, item):
        child = QtWidgets.QTreeWidgetItem([""])
        parent = QtWidgets.QTreeWidgetItem([str(key)])
        parent.setFlags(QtCore.Qt.ItemIsEditable | parent.flags())
        parent.addChild(child)
        self.addTopLevelItem(parent)
        widget = item.widget()
        self.setIndexWidget(self.indexFromItem(child, column=0), widget)
        self._widgets[key] = widget

    def __add(self, type):
        if type == "scalar":
            obj = Coef(0)
        elif type == "vector":
            obj = Coef([0,0,0], shape=(3,))
        elif type == "matrix":
            obj= Coef(np.eye(3), shape=(3,3))
        n = 1
        while hasattr(self._param, type[0]+str(n)):
            n += 1
        name = type[0] + str(n)
        setattr(self._param, name, obj)
        self.__addItem(name, obj)

    def __remove(self):
        index = self.indexFromItem(self.currentItem())
        if not index.isValid():
            return
        if index.parent().isValid():
            index=index.parent()
        key = index.data(QtCore.Qt.DisplayRole)
        delattr(self._param, key)
        del self._widgets[key]
        self.takeTopLevelItem(index.row())

    def __selected(self, item, column):
        self._sel = self.indexFromItem(item).data(QtCore.Qt.DisplayRole)

    def __changed(self, item, column):
        symbol = self.indexFromItem(item).data(QtCore.Qt.DisplayRole)
        if symbol == self._sel:
            return
        elif hasattr(self._param, symbol):
            QtWidgets.QMessageBox.warning(self, "Warning", "You cannot specify a symbol name that is already used.")
            item.setData(0, QtCore.Qt.DisplayRole, self._sel)
        else:
            setattr(self._param, symbol, getattr(self._param, self._sel))
            delattr(self._param, self._sel)
            w = self._widgets[self._sel]
            del self._widgets[self._sel]
            self._widgets[symbol] = w
