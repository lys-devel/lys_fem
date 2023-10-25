from lys.Qt import QtWidgets
from lys.widgets import ScientificSpinBox


class BoxGUI(QtWidgets.QWidget):
    def __init__(self, obj):
        super().__init__()
        self._obj = obj
        self.__initlayout()
        for v, w in zip(self._obj.args, self._values):
            w.setValue(v)

    def __initlayout(self):
        self._values = [ScientificSpinBox(valueChanged=self.__changed) for i in range(6)]
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("x"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("y"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("z"), 0, 2)
        grid.addWidget(self._values[0], 1, 0)
        grid.addWidget(self._values[1], 1, 1)
        grid.addWidget(self._values[2], 1, 2)
        grid.addWidget(QtWidgets.QLabel("dx"), 2, 0)
        grid.addWidget(QtWidgets.QLabel("dy"), 2, 1)
        grid.addWidget(QtWidgets.QLabel("dz"), 2, 2)
        grid.addWidget(self._values[3], 3, 0)
        grid.addWidget(self._values[4], 3, 1)
        grid.addWidget(self._values[5], 3, 2)
        self.setLayout(grid)

    def __changed(self):
        print([w.value() for w in self._values])
        self._obj.args = [w.value() for w in self._values]


class RectGUI(QtWidgets.QWidget):
    def __init__(self, obj):
        super().__init__()
        self._obj = obj
        self.__initlayout()
        for v, w in zip(self._obj.args, self._values):
            w.setValue(v)

    def __initlayout(self):
        self._values = [ScientificSpinBox(valueChanged=self.__changed) for i in range(5)]
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("x"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("y"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("z"), 0, 2)
        grid.addWidget(self._values[0], 1, 0)
        grid.addWidget(self._values[1], 1, 1)
        grid.addWidget(self._values[2], 1, 2)
        grid.addWidget(QtWidgets.QLabel("dx"), 2, 0)
        grid.addWidget(QtWidgets.QLabel("dy"), 2, 1)
        grid.addWidget(self._values[3], 3, 0)
        grid.addWidget(self._values[4], 3, 1)
        self.setLayout(grid)

    def __changed(self):
        self._obj.args = [w.value() for w in self._values]


class DiskGUI(QtWidgets.QWidget):
    def __init__(self, obj):
        super().__init__()
        self._obj = obj
        self.__initlayout()
        for v, w in zip(self._obj.args, self._values):
            w.setValue(v)

    def __initlayout(self):
        self._values = [ScientificSpinBox(valueChanged=self.__changed) for i in range(5)]
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("x"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("y"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("z"), 0, 2)
        grid.addWidget(self._values[0], 1, 0)
        grid.addWidget(self._values[1], 1, 1)
        grid.addWidget(self._values[2], 1, 2)
        grid.addWidget(QtWidgets.QLabel("rx"), 2, 0)
        grid.addWidget(QtWidgets.QLabel("ry"), 2, 1)
        grid.addWidget(self._values[3], 3, 0)
        grid.addWidget(self._values[4], 3, 1)
        self.setLayout(grid)

    def __changed(self):
        self._obj.args = [w.value() for w in self._values]


class LineGUI(QtWidgets.QWidget):
    def __init__(self, obj):
        super().__init__()
        self._obj = obj
        self.__initlayout()
        for v, w in zip(self._obj.p1, self._values[:3]):
            w.setValue(v)
        for v, w in zip(self._obj.p2, self._values[3:6]):
            w.setValue(v)

    def __initlayout(self):
        self._values = [ScientificSpinBox(valueChanged=self.__changed) for i in range(6)]
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("x1"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("y1"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("z1"), 0, 2)
        grid.addWidget(self._values[0], 1, 0)
        grid.addWidget(self._values[1], 1, 1)
        grid.addWidget(self._values[2], 1, 2)
        grid.addWidget(QtWidgets.QLabel("x2"), 2, 0)
        grid.addWidget(QtWidgets.QLabel("y2"), 2, 1)
        grid.addWidget(QtWidgets.QLabel("z2"), 2, 2)
        grid.addWidget(self._values[3], 3, 0)
        grid.addWidget(self._values[4], 3, 1)
        grid.addWidget(self._values[5], 3, 2)
        self.setLayout(grid)

    def __changed(self):
        self._obj.p1 = [w.value() for w in self._values[:3]]
        self._obj.p2 = [w.value() for w in self._values[3:6]]
