import numpy as np
from lys.Qt import QtCore, QtWidgets, QtGui
from lys.decorators import avoidCircularReference
from .dataView import FEMFileDialog
from ..fem import SolutionField


class ScalarFunctionWidget(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(object)

    def __init__(self, value, valueChanged=None, label=None):
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
        self._val = self._value.text()
        return self._val


class VectorFunctionWidget(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(object)

    def __init__(self, value, valueChanged=None, shape=None, label=None):
        super().__init__()
        if len(np.shape(value)) == 1:
            value = np.array(value).tolist()
            shape = np.shape(value)
        self.__initLayout(label, value, shape)
        self._combo.setCurrentText(self.__checkType(np.array(value)))
        self.__typeChanged()
        if valueChanged is not None:
            self.valueChanged.connect(valueChanged)

    def __initLayout(self, label, value, shape):
        self._combo = QtWidgets.QComboBox()
        self._combo.addItems(["Full", "Expression"])
        self._combo.currentTextChanged.connect(self.__valueChanged)

        h = QtWidgets.QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        if label is not None:
            h.addWidget(QtWidgets.QLabel(label), 2)
        h.addWidget(self._combo, 1)

        self._expr = QtWidgets.QLineEdit()
        if len(np.shape(value)) != 1:
            self._expr.setText(value)
        self._expr.textChanged.connect(self.__valueChanged)
        self._vec = self.__vectorGrid(value, shape)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addLayout(h)
        v.addWidget(self._expr)
        v.addWidget(self._vec)

    def __vectorGrid(self, value, shape):
        self._value = [QtWidgets.QLineEdit() for v in range(shape[0])]
        if len(np.shape(value)) == 0:
            value = ["0"]*3
        for w, v in zip(self._value, value):
            w.setText(str(v))
            w.textChanged.connect(self.__valueChanged)

        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        for i, w in enumerate(self._value):
            layout.addWidget(QtWidgets.QLabel("Axis " + str(i + 1)), i, 0, 1, 1)
            layout.addWidget(w, i, 1, 1, 3)
        w = QtWidgets.QWidget()
        w.setLayout(layout)
        return w

    @avoidCircularReference
    def __typeChanged(self, *args):
        self._changeType()

    def _changeType(self):
        t = self._combo.currentText()
        if t == "Expression":
            self._vec.hide()
            self._expr.show()
        else:
            self._vec.show()
            self._expr.hide()

    def __checkType(self, value):
        if len(np.shape(value))==0:
            return "Expression"
        else:
            return "Full"

    @avoidCircularReference
    def __valueChanged(self, *args):
        self._changeType()
        self.valueChanged.emit(self.value())

    def value(self):
        if self._combo.currentText() == "Expression":
            val = self._expr.text()
        else:
            val = [v.text() for v in self._value]
        return val


class MatrixFunctionWidget(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(object)

    def __init__(self, value, valueChanged=None, shape=None, label=None):
        super().__init__()
        if len(np.shape(value)) == 2:
            value = np.array(value).tolist()
            shape = np.shape(value)
        self.__initLayout(label, value, shape)
        self._combo.setCurrentText(self.__checkType(np.array(value)))
        self.__typeChanged()
        if valueChanged is not None:
            self.valueChanged.connect(valueChanged)

    def __initLayout(self, label, value, shape):
        self._combo = QtWidgets.QComboBox()
        self._combo.addItems(["Scalar", "Diagonal", "Symmetric", "Full", "Expression"])
        self._combo.currentTextChanged.connect(self.__valueChanged)

        h = QtWidgets.QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        if label is not None:
            h.addWidget(QtWidgets.QLabel(label), 2)
        h.addWidget(self._combo, 1)

        self._expr = QtWidgets.QLineEdit()
        if len(np.shape(value)) != 2:
            self._expr.setText(value)
        self._expr.textChanged.connect(self.__valueChanged)
        self._mat = self.__matrixGrid(value, shape)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addLayout(h)
        v.addWidget(self._expr)
        v.addWidget(self._mat)

    def __matrixGrid(self, value, shape):
        self._value = [[QtWidgets.QLineEdit() for _ in range(shape[1])] for _ in range(shape[0])]
        if len(np.shape(value)) == 0:
            value = [["0"]*3]*3
        for ws, vs in zip(self._value, value):
            for w, v in zip(ws, vs):
                w.setText(str(v))
                w.textChanged.connect(self.__valueChanged)

        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        for i in range(shape[0]):
            layout.addWidget(QtWidgets.QLabel(str(i + 1)), i + 1, 0, 1, 1)
        for j in range(shape[1]):
            layout.addWidget(QtWidgets.QLabel(str(j + 1)), 0, j + 1, 1, 1)
        for i, ws in enumerate(self._value):
            for j, w in enumerate(ws):
                layout.addWidget(w, i + 1, j + 1, 1, 1)
        w = QtWidgets.QWidget()
        w.setLayout(layout)
        return w

    @avoidCircularReference
    def __typeChanged(self, *args):
        self._changeType()
        
    def _changeType(self):
        t = self._combo.currentText()
        if t == "Expression":
            self._mat.hide()
            self._expr.show()
            return
        self._mat.show()
        self._expr.hide()
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
                elif t == "Full":
                    w.setEnabled(True)

    def __checkType(self, value):
        if len(np.shape(value))==0:
            return "Expression"
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
        self._changeType()
        self.valueChanged.emit(self.value())

    def value(self):
        if self._combo.currentText() == "Expression":
            val = self._expr.text()
        else:
            val = [[w.text() for w in ws] for ws in self._value]
        return val


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
    
