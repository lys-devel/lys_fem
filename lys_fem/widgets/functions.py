import numpy as np
from lys.Qt import QtCore, QtWidgets, QtGui
from lys.decorators import avoidCircularReference
from .dataView import FEMFileDialog
from ..fem import SolutionField
from ..fem.base import strToExpr


class ScalarFunctionWidget(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(object)

    def __init__(self, label, value, valueChanged=None):
        super().__init__()
        self._val = value
        self.__initLayout(label, value)
        if valueChanged is not None:
            self.valueChanged.connect(valueChanged)

    def __initLayout(self, label, value):
        self._value = QtWidgets.QLineEdit()
        self._value.setText(str(value))
        self._value.textChanged.connect(self.__valueChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        if label is not None:
            layout.addWidget(QtWidgets.QLabel(label))
        layout.addWidget(self._value)
        self.setLayout(layout)

    def __valueChanged(self):
        self.valueChanged.emit(self.value())

    def value(self):
        try:
            self._val = strToExpr(self._value.text())
        except:
            pass
        return self._val


class VectorFunctionWidget(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(object)

    def __init__(self, label, value, valueChanged=None):
        super().__init__()
        self._val = np.array(value).tolist()
        self.__initLayout(label, value)
        if valueChanged is not None:
            self.valueChanged.connect(valueChanged)

    def __initLayout(self, label, value):
        self._value = [QtWidgets.QLineEdit() for v in value]
        for w, v in zip(self._value, value):
            w.setText(str(v))
            w.textChanged.connect(self.__valueChanged)

        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        if label is not None:
            layout.addWidget(QtWidgets.QLabel(label), 0, 0)
        for i, w in enumerate(self._value):
            layout.addWidget(QtWidgets.QLabel("Axis " + str(i + 1)), i + 1, 0, 1, 1)
            layout.addWidget(w, i + 1, 1, 1, 3)
        self.setLayout(layout)

    def __valueChanged(self):
        self.valueChanged.emit(self.value())

    def value(self):
        try:
            self._val = [strToExpr(v.text()) for v in self._value]
        except:
            pass
        return self._val

class MatrixFunctionWidget(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(object)

    def __init__(self, label, value, valueChanged=None):
        super().__init__()
        self._val = np.array(value).tolist()
        self.__initLayout(label, np.array(value))
        self._combo.setCurrentText(self.__checkType(np.array(value)))
        self.__typeChanged()
        if valueChanged is not None:
            self.valueChanged.connect(valueChanged)

    def __initLayout(self, label, value):
        self._combo = QtWidgets.QComboBox()
        self._combo.addItems(["Scalar", "Diagonal", "Symmetric", "Full"])
        self._combo.currentTextChanged.connect(self.__typeChanged)
        h = QtWidgets.QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        if label is not None:
            h.addWidget(QtWidgets.QLabel(label), 2)
        h.addWidget(self._combo, 1)

        self._value = [[QtWidgets.QLineEdit() for _ in range(value.shape[1])] for _ in range(value.shape[0])]
        for ws, vs in zip(self._value, value):
            for w, v in zip(ws, vs):
                w.setText(str(v))
                w.textChanged.connect(self.__valueChanged)
        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        for i in range(value.shape[0]):
            layout.addWidget(QtWidgets.QLabel(str(i + 1)), i + 1, 0, 1, 1)
        for j in range(value.shape[1]):
            layout.addWidget(QtWidgets.QLabel(str(j + 1)), 0, j + 1, 1, 1)
        for i, ws in enumerate(self._value):
            for j, w in enumerate(ws):
                layout.addWidget(w, i + 1, j + 1, 1, 1)

        v = QtWidgets.QVBoxLayout()
        v.setContentsMargins(0, 0, 0, 0)
        v.addLayout(h)
        v.addLayout(layout)
        self.setLayout(v)

    @avoidCircularReference
    def __typeChanged(self, *args):
        self.__changeType()

    def __changeType(self):
        t = self._combo.currentText()
        for i, ws in enumerate(self._value):
            for j, w in enumerate(ws):
                if t == "Scalar":
                    w.setText(str(self._value[0][0].text() if i == j else 0))
                    w.setEnabled(i == 0 and j == 0)
                elif t == "Diagonal":
                    w.setText(str(self._value[i][j].text() if i == j else 0))
                    w.setEnabled(i == j)
                elif t == "Symmetric":
                    w.setText(str(self._value[i][j].text() if i <= j else self._value[j][i].text()))
                    w.setEnabled(i <= j)
                else:
                    w.setEnabled(True)

    def __checkType(self, value):
        if value[0, 1] == value[1, 0] and value[2, 0] == value[0, 2] and value[1, 2] == value[2, 1]:
            if value[0, 1] == value[0, 2] and value[0, 1] == value[1, 2] and value[0, 1] in ["0", "0.0", 0]:
                if value[0, 0] == value[1, 1] and value[0, 0] == value[2, 2]:
                    return "Scalar"
                else:
                    return "Diagonal"
            else:
                return "Symmetric"
        else:
            return "Full"

    @avoidCircularReference
    def __valueChanged(self, *args):
        self.__changeType()
        self.valueChanged.emit(self.value())

    def value(self):
        try:
            self._val = [[strToExpr(w.text()) for w in ws] for ws in self._value]
        except:
            pass
        return self._val


class SolutionFieldWidget(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(SolutionField)
    
    def __init__(self, value=None, valueChanged=None):
        super().__init__()
        self.__initLayout()
        if value is not None:
            self._path.setText(value.path)
            self._expr.setText(value.expression)
        if valueChanged is not None:
            self.valueChanged.connect(valueChanged)

    def __initLayout(self):
        self._sel = QtWidgets.QPushButton("...", clicked=self.__clicked)
        self._path = QtWidgets.QLineEdit(textChanged=self.__changed)
        self._expr = QtWidgets.QLineEdit(textChanged=self.__changed)

        self._grid = QtWidgets.QGridLayout()
        self._grid.addWidget(QtWidgets.QLabel("Path"), 0, 0)
        self._grid.addWidget(QtWidgets.QLabel("Expression"), 1, 0)
        self._grid.addWidget(self._path, 0, 1, 1, 2)
        self._grid.addWidget(self._expr, 1, 1, 1, 2)
        self._grid.addWidget(self._sel, 0, 3)
        self.setLayout(self._grid)

    def __changed(self):
        self.valueChanged.emit(self.value)

    def __clicked(self):
        d = FEMFileDialog(self)
        ok = d.exec_()
        if ok:
            self._path.setText("../"+d.result)

    @property
    def value(self):
        return SolutionField(self._path.text(), self._expr.text())
    
