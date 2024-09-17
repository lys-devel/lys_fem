import numpy as np
import sympy as sp
from lys.Qt import QtWidgets, QtCore, QtGui

class ParameterTab(QtWidgets.QWidget):
    def __init__(self, fem):
        super().__init__()
        self.__initLayout(fem)

    def __initLayout(self, model):
        self._param = ParametersWidget(model)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self._param)
        layout.addStretch()
        self.setLayout(layout)

    def reset(self):
        self._param.reset()


class ParametersWidget(QtWidgets.QTreeWidget):
    def __init__(self, fem):
        super().__init__()
        self._fem = fem
        self.setHeaderLabels(["Parameter", "Expression", "Value"])
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
