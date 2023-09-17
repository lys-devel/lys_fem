import functools

from lys.Qt import QtCore, QtWidgets, QtGui

from ..fem.geometry import geometryCommands
from ..widgets import TreeItem, TreeStyleEditor

from .geometryOrderGUI import geometryCommandGUIs


class GeometryEditor(QtWidgets.QWidget):
    showGeometry = QtCore.pyqtSignal()

    def __init__(self, geom):
        super().__init__()
        self._geom = geom
        self.__initlayout()

    def __initlayout(self):
        self._tr = TreeStyleEditor(GeometryTree(self._geom))
        self._geom.updated.connect(self._tr.update)

        self._generateAll = QtWidgets.QCheckBox("Generate all")
        self._generateAll.setChecked(True)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QPushButton("Update", clicked=self.showGeometry.emit))
        h.addWidget(self._generateAll)

        self._tr.layout.insertLayout(1, h)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tr)
        self.setLayout(layout)

    def generate(self):
        if self._generateAll.isChecked():
            res = self._geom.generateGeometry()
        else:
            indexes = self._tr.treeView.selectedIndexes()
            if len(indexes) != 0:
                res = self._geom.generateGeometry(indexes[0].row())
        return res


class GeometryTree(TreeItem):
    def __init__(self, geom):
        super().__init__()
        self._geom = geom
        self._children = [GeometryTreeItem(c, self) for c in self._geom.commands]

    @property
    def children(self):
        return self._children

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        for key, commands in geometryCommands.items():
            sub = self._menu.addMenu(key)
            for item in commands:
                sub.addAction(QtWidgets.QAction(item.type, self.treeWidget(), triggered=functools.partial(lambda i: self.__selected(i), i=item)))
        return self._menu

    def __selected(self, itemType):
        c = itemType()
        self._geom.addCommand(c)
        self._children.append(GeometryTreeItem(c, self))


class GeometryTreeItem(TreeItem):
    def __init__(self, item, parent):
        super().__init__(parent)
        self._item = item

    @property
    def name(self):
        return self._item.type

    @property
    def menu(self):
        return self.parent.menu

    @property
    def widget(self):
        return geometryCommandGUIs[self._item.__class__](self._item)
