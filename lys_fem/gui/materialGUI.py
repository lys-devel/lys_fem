from lys.Qt import QtWidgets, QtCore, QtGui

from ..widgets import FEMTreeItem, GeometrySelector
from ..fem import Material, materialParameters, UserDefinedParameter


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
        if isinstance(self._param, UserDefinedParameter):
            return _UserDefinedParameterWidget(self._param)
        else:
            return _ParameterWidget(self._param)


class _MaterialWidget(QtWidgets.QWidget):
    def __init__(self, canvas, fem, mat):
        super().__init__()
        self._material = mat
        self.__initlayout(canvas, fem, mat)

    def __initlayout(self, canvas, fem, mat):
        domain = GeometrySelector(canvas, fem, mat.geometries)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(domain)
        self.setLayout(layout)


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
        for key in param.getParameters(3).keys():
            self.__addItem(key)

    def __addItem(self, key):
        child = QtWidgets.QTreeWidgetItem(["", ""])
        parent = QtWidgets.QTreeWidgetItem([key, self._param.description[key]])
        parent.addChild(child)
        widget = self._param.widget(key)
        self.addTopLevelItem(parent)
        self.setIndexWidget(self.indexFromItem(child, column=1), widget)
        self._widgets.append(widget)

    def _buildContextMenu(self):
        self._menu = QtWidgets.QMenu()
        params = self._param.getParameters(3)
        sub = self._menu.addMenu("Add")
        for key, desc in self._param.description.items():
            if key not in params:
                sub.addAction(QtWidgets.QAction(key+": "+desc, self, triggered=lambda x, y=key: self.__add(y)))
        self._menu.addAction(QtWidgets.QAction("Remove", self, triggered=self.__remove))
        self._menu.exec_(QtGui.QCursor.pos())

    def __add(self, key):
        setattr(self._param, key, self._param.default[key])
        self.__addItem(key)

    def __remove(self):
        index = self.indexFromItem(self.currentItem())
        if not index.isValid():
            return
        if index.parent().isValid():
            index=index.parent()
        key = index.data(QtCore.Qt.DisplayRole)
        setattr(self._param, key, None)
        self.takeTopLevelItem(index.row())



class _UserDefinedParameterWidget(QtWidgets.QTreeWidget):
    def __init__(self, param, parent=None):
        super().__init__(parent)
        raise RuntimeError("Not implemented")
        self._param = param
        self.setColumnCount(2)
        self.setHeaderLabels(["Symbol", "Description"])
        #self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        #self.customContextMenuRequested.connect(self._buildContextMenu)
        self.__initLayout(param)

    def __initLayout(self, param):
        self._widgets = []
        for key in param.getParameters(3).keys():
            self.__addItem(key)

    def __addItem(self, key):
        child = QtWidgets.QTreeWidgetItem(["", ""])
        parent = QtWidgets.QTreeWidgetItem([key, self._param.description[key]])
        parent.addChild(child)
        widget = self._param.widget(key)
        self.addTopLevelItem(parent)
        self.setIndexWidget(self.indexFromItem(child, column=1), widget)
        self._widgets.append(widget)

    def _buildContextMenu(self):
        self._menu = QtWidgets.QMenu()
        params = self._param.getParameters(3)
        sub = self._menu.addMenu("Add")
        for key, desc in self._param.description.items():
            if key not in params:
                sub.addAction(QtWidgets.QAction(key+": "+desc, self, triggered=lambda x, y=key: self.__add(y)))
        self._menu.addAction(QtWidgets.QAction("Remove", self, triggered=self.__remove))
        self._menu.exec_(QtGui.QCursor.pos())

    def __add(self, key):
        setattr(self._param, key, self._param.default[key])
        self.__addItem(key)

    def __remove(self):
        index = self.indexFromItem(self.currentItem())
        if not index.isValid():
            return
        if index.parent().isValid():
            index=index.parent()
        key = index.data(QtCore.Qt.DisplayRole)
        setattr(self._param, key, None)
        self.takeTopLevelItem(index.row())
