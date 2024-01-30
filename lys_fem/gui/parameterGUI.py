import numpy as np
import sympy as sp
from lys.Qt import QtWidgets, QtCore, QtGui

class ParameterTab(QtWidgets.QWidget):
    def __init__(self, fem):
        super().__init__()
        self.__initLayout(fem)

    def __initLayout(self, model):
        self._scaling = ScalingWidget(model)
        self._param = ParametersWidget(model)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self._scaling)
        layout.addWidget(self._param)
        layout.addStretch()
        self.setLayout(layout)

    def reset(self):
        self._scaling.reset()
        self._param.reset()


class ScalingWidget(QtWidgets.QGroupBox):
    def __init__(self, fem):
        super().__init__("Scaling")
        self._fem = fem
        self.__initLayout()
        self.reset()

    def __initLayout(self):
        self._length = QtWidgets.QSpinBox(valueChanged=self.__changed)
        self._time = QtWidgets.QSpinBox(valueChanged=self.__changed)
        self._mass = QtWidgets.QSpinBox(valueChanged=self.__changed)
        self._current = QtWidgets.QSpinBox(valueChanged=self.__changed)
        self._temp = QtWidgets.QSpinBox(valueChanged=self.__changed)
        self._mol = QtWidgets.QSpinBox(valueChanged=self.__changed)
        self._cd = QtWidgets.QSpinBox(valueChanged=self.__changed)

        self._length.setPrefix("10** ")
        self._time.setPrefix("10** ")
        self._mass.setPrefix("10** ")
        self._current.setPrefix("10** ")
        self._temp.setPrefix("10** ")
        self._mol.setPrefix("10** ")
        self._cd.setPrefix("10** ")

        self._length.setRange(-128, 128)
        self._time.setRange(-128, 128)
        self._mass.setRange(-128, 128)
        self._current.setRange(-128, 128)
        self._temp.setRange(-128, 128)
        self._mol.setRange(-128, 128)
        self._cd.setRange(-128, 128)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(QtWidgets.QLabel("Length (m)"), 0, 0)
        layout.addWidget(QtWidgets.QLabel("Time (s)"), 0, 2)
        layout.addWidget(QtWidgets.QLabel("Mass (kg)"), 1, 0)
        layout.addWidget(QtWidgets.QLabel("Current (A)"), 1, 2)
        layout.addWidget(QtWidgets.QLabel("Temperature (K)"), 2, 0)
        layout.addWidget(QtWidgets.QLabel("Amount (mol)"), 2, 2)
        layout.addWidget(QtWidgets.QLabel("Luminous (cd)"), 3, 0)
        layout.addWidget(self._length, 0, 1)
        layout.addWidget(self._time, 0, 3)
        layout.addWidget(self._mass, 1, 1)
        layout.addWidget(self._current, 1, 3)
        layout.addWidget(self._temp, 2, 1)
        layout.addWidget(self._mol, 2, 3)
        layout.addWidget(self._cd, 3, 1)

        self.setLayout(layout)

    def reset(self):
        norms = self._fem.scaling._norms
        self._length.setValue(int(np.log10(norms[0])))
        self._time.setValue(int(np.log10(norms[1])))
        self._mass.setValue(int(np.log10(norms[2])))
        self._current.setValue(int(np.log10(norms[3])))
        self._temp.setValue(int(np.log10(norms[4])))
        self._mol.setValue(int(np.log10(norms[5])))
        self._cd.setValue(int(np.log10(norms[6])))

    def __changed(self):
        l = 10**self._length.value()
        t = 10**self._time.value()
        m = 10**self._mass.value()
        c = 10**self._current.value()
        T = 10**self._temp.value()
        mol = 10**self._mol.value()
        cd = 10**self._cd.value()
        self._fem.scaling.set(l,t,m,c,T,mol,cd)


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
