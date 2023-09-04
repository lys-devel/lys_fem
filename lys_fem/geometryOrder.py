from lys.Qt import QtWidgets
from lys.widgets import ScientificSpinBox


class AddBox(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def execute(self, model):
        model.occ.addBox(*self.args, **self.kwargs)

    def widget(self):
        return _AddBoxWidget(self)

    @classmethod
    @property
    def type(cls):
        return "Add box"

    @classmethod
    @property
    def default(cls):
        return AddBox(0, 0, 0, 1, 1, 1)


class AddRect(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def execute(self, model):
        model.occ.addRectangle(*self.args, **self.kwargs)

    def widget(self):
        return QtWidgets.QWidget(self)

    @classmethod
    @property
    def type(cls):
        return "Add rectangle"

    @classmethod
    @property
    def default(cls):
        return AddRect(0, 0, 0, 1, 1)


class _AddBoxWidget(QtWidgets.QWidget):
    def __init__(self, obj):
        super().__init__()
        self._obj = obj
        self.__initLayout()
        for v, w in zip(obj.args, self._values):
            w.setValue(v)

    def __initLayout(self):
        grid = QtWidgets.QGridLayout()
        self._values = [ScientificSpinBox(valueChanged=self.__changed) for i in range(6)]
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
        self._obj.args = [w.value() for w in self._values]


geometryCommands = [AddBox, AddRect]
