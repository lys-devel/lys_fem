from lys.Qt import QtWidgets
from lys_fem.gui import MethodComboBox

class LLGModelWidget(QtWidgets.QWidget):
    def __init__(self, model):
        super().__init__()
        self.__initLayout(model)
        self._model = model

    def __initLayout(self, model):
        self._method = MethodComboBox(model)
        self._const = QtWidgets.QComboBox()
        self._const.addItems(["Projection", "Lagrange", "Alouges"])
        self._const.setCurrentText(model.constraint)
        self._const.currentTextChanged.connect(self.__change)
        self._order = QtWidgets.QSpinBox()
        self._order.setValue(model.order)
        self._order.setRange(0,100)
        self._order.valueChanged.connect(self.__change)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(QtWidgets.QLabel("Constraint"), 0, 0)
        layout.addWidget(QtWidgets.QLabel("Discretization"), 1, 0)
        layout.addWidget(QtWidgets.QLabel("Element Order"), 2, 0)
        layout.addWidget(self._const, 0, 1)
        layout.addWidget(self._method, 1, 1)
        layout.addWidget(self._order, 2, 1)
        self.setLayout(layout)

    def __change(self):
        self._model.order=self._order.value()
        self._model._constraint = self._const.currentText()