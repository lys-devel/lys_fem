import os
import sympy as sp
from lys.Qt import QtWidgets, QtCore, QtGui

from ..widgets import FEMTreeItem, FEMFileDialog
from .materialGUI import MaterialTree

class VariableTree(FEMTreeItem):
    def __init__(self, obj, canvas):
        super().__init__(fem=obj, canvas=canvas)
        self.append(GlobalVariableTree(self))
        self.append(SolutionFieldTree(self))
        self.append(MaterialTree(self, obj.materials))

    def setVariables(self, materials):
        self.children[2].setMaterials(materials)


class GlobalVariableTree(FEMTreeItem):
    def __init__(self, parent):
        super().__init__(parent)

    @property
    def name(self):
        return "Global Parameters"

    @property
    def widget(self):
        return _ParametersWidget(self.fem())


class SolutionFieldTree(FEMTreeItem):
    def __init__(self, parent):
        super().__init__(parent)

    @property
    def name(self):
        return "Solution Fields"

    @property
    def widget(self):
        return _FieldWidget(self.fem())


class _ParametersWidget(QtWidgets.QTreeWidget):
    def __init__(self, fem):
        super().__init__()
        self._fem = fem
        self.setHeaderLabels(["Symbol", "Expression", "Value"])
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._buildContextMenu)
        self.reset()
        self.itemChanged.connect(self.__changed)

    def reset(self):
        while self.topLevelItemCount()>0:
            item = self.topLevelItem(0)
            index = self.indexFromItem(item)
            self.takeTopLevelItem(index.row())
        for key, item in self._fem.parameters.items():
            item = QtWidgets.QTreeWidgetItem([str(key), str(item), ""])
            item.setFlags(QtCore.Qt.ItemIsEditable | item.flags())
            self.addTopLevelItem(item)
        self.__changed()

    def _buildContextMenu(self):
        menu = QtWidgets.QMenu()
        menu.addAction(QtWidgets.QAction("Add", self, triggered=self.__add))
        menu.addAction(QtWidgets.QAction("Remove", self, triggered=self.__remove))
        menu.exec_(QtGui.QCursor.pos())
                       
    def __add(self):
        item = QtWidgets.QTreeWidgetItem(["name", "1", "1"])
        item.setFlags(QtCore.Qt.ItemIsEditable | item.flags())
        self.addTopLevelItem(item)
        self.__changed()

    def __remove(self):
        for item in self.selectedItems():
            index = self.indexFromItem(item)
            self.takeTopLevelItem(index.row())
        self.__changed()

    def __changed(self):
        d = {}
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            d[sp.parse_expr(item.text(0))] = sp.parse_expr(item.text(1))
        self._fem.parameters.clear()
        self._fem.parameters.update(d)
        sol = self._fem.parameters.getSolved()
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            key = sp.parse_expr(item.text(0))
            if key in sol:
                item.setText(2, str(sol[key]))
            else:
                item.setText(2, "N/A")


class _FieldWidget(QtWidgets.QTreeWidget):
    _types = {"First": 0, "Last": -1, "Time-dependent": None}

    def __init__(self, fem):
        super().__init__()
        self._fem = fem
        self.setHeaderLabels(["Symbol", "Type", "Path", "Expression"])
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._buildContextMenu)
        self.reset()
        
    def reset(self):
        while self.topLevelItemCount() > 0:
            item = self.topLevelItem(0)
            index = self.indexFromItem(item)
            self.takeTopLevelItem(index.row())
        for key, item in self._fem.solutionFields.items():
            typ = [k for k, value in self._types.items() if value==item.index][0]
            item = QtWidgets.QTreeWidgetItem([str(key), typ, str(item.path), str(item.expression)])
            self.addTopLevelItem(item)

    def _buildContextMenu(self):
        menu = QtWidgets.QMenu()
        menu.addAction(QtWidgets.QAction("Add", self, triggered=self.__add))
        menu.addAction(QtWidgets.QAction("Edit", self, triggered=self.__edit))
        menu.addAction(QtWidgets.QAction("Remove", self, triggered=self.__remove))
        menu.exec_(QtGui.QCursor.pos())
                       
    def __add(self):
        d = _SolutionFieldEditDialog(self)
        ok = d.exec_()
        if ok:
            name, typ, path, expr = d.result()
            self._fem.solutionFields.add(name, path, expr, index=self._types[typ])
            self.reset()

    def __edit(self):
        key = self.indexFromItem(self.selectedItems()[0]).data(QtCore.Qt.DisplayRole)
        obj = self._fem.solutionFields[key]
        d = _SolutionFieldEditDialog(self, key, obj)
        ok = d.exec_()
        if ok:
            name, typ, path, expr = d.result()
            obj.set(path, expr, self._types[typ])
            self.reset()

    def __remove(self):
        for item in self.selectedItems():
            index = self.indexFromItem(item)
            self.takeTopLevelItem(index.row())
            self._fem.solutionFields.pop(index.data(QtCore.Qt.DisplayRole))
        self.reset()


class _SolutionFieldEditDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, key=None, obj=None):
        super().__init__(parent)
        self.__initLayout()
        if key is not None:
            self._name.setText(key)
        if obj is not None:
            self._expr.setText(str(obj.expression))
            self._path.setText(obj.path)
            if obj.index == 0:
                self._type.setCurrentIndex(1)
            elif obj.index == -1:
                self._type.setCurrentIndex(0)
            else:
                self._type.setCurrentIndex(2)        

    def __initLayout(self):
        self._name = QtWidgets.QLineEdit()
        self._expr = QtWidgets.QLineEdit()
        self._path = QtWidgets.QLineEdit()
        self._type = QtWidgets.QComboBox()
        self._type.addItems(["Last", "First", "Time-dependent"])

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Symbol name"), 0, 0)
        grid.addWidget(self._name, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Type"), 1, 0)
        grid.addWidget(self._type, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Path"), 2, 0)
        grid.addWidget(self._path, 2, 1)
        grid.addWidget(QtWidgets.QPushButton("...", clicked=self.__click), 2, 2)
        grid.addWidget(QtWidgets.QLabel("Expression"), 3, 0)
        grid.addWidget(self._expr, 3, 1, 1, 2)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 6)
        grid.setColumnStretch(2, 1)

        h1 = QtWidgets.QHBoxLayout()
        h1.addWidget(QtWidgets.QPushButton('O K', clicked=self.accept))
        h1.addWidget(QtWidgets.QPushButton('CALCEL', clicked=self.reject))

        v1 = QtWidgets.QVBoxLayout()
        v1.addLayout(grid)
        v1.addLayout(h1)

        self.setLayout(v1)

    def __click(self):
        d = FEMFileDialog(self)
        ok = d.exec_()
        if ok:
            self._path.setText(os.path.abspath("FEM/"+d.result))

    def result(self):
        return self._name.text(), self._type.currentText(), self._path.text(), self._expr.text()