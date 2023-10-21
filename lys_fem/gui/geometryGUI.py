from lys.Qt import QtWidgets

from ..fem.geometry import geometryCommands
from ..fem import OccMesher
from ..widgets import FEMTreeItem, TreeStyleEditor


class GeometryEditor(QtWidgets.QWidget):
    def __init__(self, obj, canvas):
        super().__init__()
        self._canvas = canvas
        self._obj = obj
        self._geom = obj.geometryGenerator
        self.__initlayout()

    def __initlayout(self):
        self._gt = GeometryTree(self._obj, self._canvas)
        self._tr = TreeStyleEditor(self._gt)

        self._generateAll = QtWidgets.QCheckBox("Generate all")
        self._generateAll.setChecked(True)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QPushButton("Update", clicked=self.showGeometry))
        h.addWidget(self._generateAll)

        self._tr.layout.insertLayout(1, h)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tr)
        self.setLayout(layout)

    def showGeometry(self):
        geom = self._generate()
        with self._canvas.delayUpdate():
            self._canvas.clear()
            for m in OccMesher().getMeshWave(geom, dim=self._obj.dimension):
                obj = self._canvas.append(m)
                obj.setColor("#cccccc", type="color")

    def _generate(self):
        if self._generateAll.isChecked():
            res = self._geom.generateGeometry()
        else:
            indexes = self._tr.treeView.selectedIndexes()
            if len(indexes) != 0:
                res = self._geom.generateGeometry(indexes[0].row())
        return res

    def setGeometry(self, geom):
        self._geom = geom
        self._gt.setGeometry(geom)


class GeometryTree(FEMTreeItem):
    def __init__(self, obj, canvas):
        super().__init__(fem=obj, canvas=canvas)
        self.setGeometry(obj.geometryGenerator)

    def setGeometry(self, geom):
        self.clear()
        self._geom = geom
        for c in self._geom.commands:
            super().append(GeometryTreeItem(c, self))

    def append(self, item):
        super().append(GeometryTreeItem(item, self))
        self._geom.addCommand(item)

    def remove(self, item):
        i = super().remove(item)
        self._geom.commands.remove(self._geom.commands[i])

    @ property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        for key, commands in geometryCommands.items():
            sub = self._menu.addMenu(key)
            for item in commands:
                sub.addAction(QtWidgets.QAction(item.type, self.treeWidget(), triggered=lambda i, j=item: self.append(j())))
        return self._menu


class GeometryTreeItem(FEMTreeItem):
    def __init__(self, item, parent):
        super().__init__(parent)
        self._item = item

    @ property
    def name(self):
        return self._item.type

    @ property
    def menu(self):
        menu = self.parent.menu
        menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return menu

    @ property
    def widget(self):
        return self._item.widget()
