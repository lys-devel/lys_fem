from lys.Qt import QtWidgets

from lys_fem.geometry import geometryCommands
from ..fem import OccMesher
from ..widgets import FEMTreeItem, TreeStyleEditor, GeometrySelector


class GeometryEditor(QtWidgets.QWidget):
    def __init__(self, obj, canvas):
        super().__init__()
        self._canvas = canvas
        self._obj = obj
        self._geom = obj.geometries
        self.__initlayout()

    def __initlayout(self):
        self._gt = _ParentTree(self._obj, self._canvas)
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
            for m in OccMesher().generate(geom).getMeshWave(dim=self._obj.dimension):
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


class _ParentTree(FEMTreeItem):
    def __init__(self, obj, canvas):
        super().__init__(fem=obj, canvas=canvas)
        self._geom = GeometryTree(self, obj)
        self._group = GroupTree(self)
        self.append(self._geom)
        self.append(self._group)

    def setGeometry(self, *args, **kwargs):
        self._geom.setGeometry(*args, **kwargs)
        self._group.update()

    @ property
    def actions(self):
        res = []
        res.append(QtWidgets.QAction("Clear", self.treeWidget(), triggered=self._clear))
        res.append(QtWidgets.QAction("Export to file", self.treeWidget(), triggered=self._export))
        res.append(QtWidgets.QAction("Import from file", self.treeWidget(), triggered=self._import))
        return res

    def _export(self):
        path, type = QtWidgets.QFileDialog.getSaveFileName(self.treeWidget(), "Export Geometry", filter="lys_fem Geometry (*.lfg);;All files (*.*)")
        if len(path) != 0:
            if not path.endswith(".lfg"):
                path = path + ".lfg"
            self.fem().geometries.save(path)

    def _import(self):
        path, type = QtWidgets.QFileDialog.getOpenFileName(self.treeWidget(), 'Import Geometry', filter="lys_fem Geometry (*.lfg);;All files (*.*)")
        if len(path) != 0:
            self.fem().geometries.load(path)
        self.setGeometry(self.fem().geometries)

    def _clear(self):
        msg = QtWidgets.QMessageBox(parent=self.treeWidget())
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setWindowTitle("Caution")
        msg.setText("All geometries will be removed. Do you really want to proceed?")
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        ok = msg.exec_()
        if ok == QtWidgets.QMessageBox.Yes:
            self._geom.clear()
            self._group.clear()


class GeometryTree(FEMTreeItem):
    def __init__(self, parent, obj):
        super().__init__(parent)
        self.setGeometry(obj.geometries)

    @property
    def name(self):
        return "Geometries"

    def setGeometry(self, geom):
        self.clear()
        self._geom = geom
        for c in self._geom.commands:
            super().append(GeometryTreeItem(c, self))

    def append(self, item):
        super().append(GeometryTreeItem(item, self))
        self._geom.add(item)

    def remove(self, item):
        i = super().remove(item)
        self._geom.remove(self._geom.commands[i])

    @ property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        for key, commands in geometryCommands.items():
            sub = self._menu.addMenu(key)
            for item in commands:
                sub.addAction(QtWidgets.QAction(item.type, self.treeWidget(), triggered=lambda i, j=item: self.append(j())))
        self._menu.addSeparator()
        for act in self.parent.actions:
            self._menu.addAction(act)
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


class GroupTree(FEMTreeItem):
    def __init__(self, parent):
        super().__init__(parent)
        self.update()

    def update(self):
        for name, geom in self.fem().geometries.groups.items():
            super().append(GroupTreeItem(name, geom.geometryType, self))

    @property
    def name(self):
        return "Groups"

    @ property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Add a new group", self.treeWidget(), triggered=self._add))
        self._menu.addSeparator()
        for act in self.parent.actions:
            self._menu.addAction(act)
        return self._menu

    def _add(self):
        d = _NewGroupDialog(self.treeWidget())
        if d.exec_():
            self.append(d.name, d.type)

    def append(self, name, type):
        if name in self.fem().geometries.groups:
            return
        self.fem().geometries.addGroup(type, name)
        super().append(GroupTreeItem(name, type, self))       

    def remove(self, item, name):
        i = super().remove(item)
        self.fem().geometries.removeGroup(name)

    def clear(self):
        self.fem().geometries.groups.clear()
        for item in reversed(self.children):
            super().remove(item)


class GroupTreeItem(FEMTreeItem):
    def __init__(self, name, type, parent):
        super().__init__(parent)
        self._name = name
        self._type = type

    @ property
    def name(self):
        return self._name + " ("+self._type+")"

    @ property
    def menu(self):
        menu = self.parent.menu
        menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self, self._name)))
        return menu

    @ property
    def widget(self):
        return GeometrySelector(self.canvas(), self.fem(), self.fem().geometries.groups[self._name], acceptedTypes=["Selected"], autoStart=True)


class _NewGroupDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._name = QtWidgets.QLineEdit()
        self._combo = QtWidgets.QComboBox()
        self._combo.addItems(["Domain", "Boundary", "Volume", "Surface", "Edge", "Point"])

        g = QtWidgets.QGridLayout()
        g.addWidget(QtWidgets.QLabel("Name"), 0, 0)
        g.addWidget(self._name, 0, 1)
        g.addWidget(QtWidgets.QLabel("Type"), 1, 0)
        g.addWidget(self._combo, 1, 1)
        g.addWidget(QtWidgets.QPushButton('O K', clicked=self.accept), 2, 0)
        g.addWidget(QtWidgets.QPushButton('CALCEL', clicked=self.reject), 2, 1)

        self.setLayout(g)

    @property
    def name(self):
        return self._name.text()

    @property
    def type(self):
        return self._combo.currentText()

