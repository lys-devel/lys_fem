from lys.Qt import QtWidgets, QtCore

from ..widgets import FEMTreeItem, GeometrySelector
from ..fem import Material, ElasticParameters


class MaterialTree(FEMTreeItem):
    def __init__(self, obj, canvas):
        super().__init__(fem=obj, canvas=canvas)
        self._materials = obj._materials
        self._children = [_MaterialGUI(m, self) for m in self._materials]
        self._children[0].setDefault(True)

    @property
    def children(self):
        return self._children

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Add new material", self.treeWidget(), triggered=self.__add))
        return self._menu

    def __add(self):
        i = 1
        while "Material" + str(i) in [m.name for m in self._materials]:
            i += 1
        self.beginInsertRow(len(self._children))
        m = Material("Material" + str(i))
        self._materials.append(m)
        self._children.append(_MaterialGUI(m, self))
        self.endInsertRow()

    def remove(self, material):
        i = self._children.index(material)
        self.beginRemoveRow(i)
        self._materials.remove(self._materials[i])
        self._children.remove(material)
        self.endRemoveRow()


class _MaterialGUI(FEMTreeItem):
    def __init__(self, material, parent):
        super().__init__(parent)
        self._material = material
        self._children = [_ElasticParamsGUI(p) for p in material]
        self._default = False

    def setDefault(self, b):
        self._default = b

    @property
    def name(self):
        return self._material.name

    @property
    def children(self):
        return self._children

    @property
    def widget(self):
        wid = _MaterialWidget(self.canvas(), self.fem(), self._material.domains)
        wid.selectionChanged.connect(self.__set)
        return wid

    @property
    def menu(self):
        menu = self.parent.menu
        if not self._default:
            menu.addAction(QtWidgets.QAction("Remove material", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        menu.addSeparator()
        add = menu.addMenu("Add new parameter")
        add.addAction(QtWidgets.QAction("Elasticity", self.treeWidget(), triggered=self.__add))
        return menu

    def __add(self):
        self.beginInsertRow(len(self._children))
        p = ElasticParameters()
        self._material.append(p)
        self._children.append(_ElasticParamsGUI(p, self))
        self.endInsertRow()

    def remove(self, param):
        i = self._children.index(param)
        self.beginRemoveRow(i)
        self._material.remove(self._material[i])
        self._children.remove(param)
        self.endRemoveRow()

    def __set(self, selected):
        self._material.domains = selected


class _MaterialWidget(QtWidgets.QWidget):
    selectionChanged = QtCore.pyqtSignal(list)

    def __init__(self, canvas, fem, domains):
        super().__init__()
        self.__initlayout(canvas, fem, domains)

    def __initlayout(self, canvas, fem, domains):
        self._domain = GeometrySelector("Target Domains", fem.dimension, canvas, fem, domains)
        self._domain.selectionChanged.connect(self.selectionChanged)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._domain)
        self.setLayout(layout)


class _ElasticParamsGUI(FEMTreeItem):
    def __init__(self, param, parent):
        super().__init__(parent)
        self._param = param

    @property
    def name(self):
        return "Elasticity"

    @property
    def menu(self):
        menu = QtWidgets.QMenu()
        menu.addAction(QtWidgets.QAction("Remove parameters", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return menu
