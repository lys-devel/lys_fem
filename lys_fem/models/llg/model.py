from lys.Qt import QtWidgets

from lys_fem import FEMFixedModel, Equation, DomainCondition
from lys_fem.gui import MethodComboBox
from . import DirichletBoundary

class LLGEquation(Equation):
    className = "LLG Equation"
    def __init__(self, varName="m", **kwargs):
        super().__init__(varName, **kwargs)


class ExternalMagneticField(DomainCondition):
    className = "External Magnetic Field"

    @classmethod
    def default(cls, fem, model):
        return ExternalMagneticField([0,0,0])

    def widget(self, fem, canvas):
        return super().widget(fem, canvas, title="Magnetic Field (T)")


class UniaxialAnisotropy(DomainCondition):
    className = "UniaxialAnisotropy"


class CubicAnisotropy(DomainCondition):
    className = "CubicAnisotropy"


class MagneticScalarPotential(DomainCondition):
    className = "MagneticScalarPotential"

    @classmethod
    def default(cls, fem, model):
        return MagneticScalarPotential(0)

    def widget(self, fem, canvas):
        return super().widget(fem, canvas, title="Magnetic scalar potential (A)")


class SpinTransferTorque(DomainCondition):
    className = "SpinTransferTorque"

    @classmethod
    def default(cls, fem, model):
        return cls([0]*fem.dimension)

    def widget(self, fem, canvas):
        return super().widget(fem, canvas, title="Spin polarized current (A/m^2)")


class LLGModel(FEMFixedModel):
    className = "LLG"
    equationTypes = [LLGEquation]
    domainConditionTypes = [ExternalMagneticField, UniaxialAnisotropy, CubicAnisotropy, MagneticScalarPotential, SpinTransferTorque]
    boundaryConditionTypes = [DirichletBoundary]

    def __init__(self, *args, discretization="LLG Asym", constraint="Lagrange", order=2, **kwargs):
        super().__init__(3, discretization=discretization, order=order, *args, **kwargs)
        self._constraint = constraint

    @property
    def discretizationTypes(self):
        return ["LLG Asym"] + super().discretizationTypes

    @property
    def constraint(self):
        return self._constraint

    @classmethod
    def loadFromDictionary(cls, d):
        m = super().loadFromDictionary(d)
        if "constraint" in d:
            m._constraint = d["constraint"]
        return m

    def saveAsDictionary(self):
        d = super().saveAsDictionary()
        d["constraint"] = self._constraint
        return d

    def widget(self, fem, canvas):
        return LLGModelWidget(self)


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