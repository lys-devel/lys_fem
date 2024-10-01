from lys.Qt import QtWidgets, QtCore

from ..widgets import FEMTreeItem, GeometrySelector
from ..fem import Material, materialParameters


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
        return self._param.widget()


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