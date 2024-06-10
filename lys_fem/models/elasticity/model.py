from lys_fem import FEMModel, Equation, DomainCondition
from .. import common
from . import DirichletBoundary

import numpy as np
from lys.Qt import QtWidgets
from lys_fem.widgets import GeometrySelector, ScalarFunctionWidget


class ChristffelEquation(Equation):
    className = "Christffel Equation"
    def __init__(self, varName="u", **kwargs):
        super().__init__(varName, **kwargs)


class InitialCondition(common.InitialCondition):
    unit = "m"


class ThermoelasticStress(DomainCondition):
    className = "ThermoelasticStress"
    unit = "K"

    def __init__(self, values=300, varName="T", *args, **kwargs):
        super().__init__(values=values, *args, **kwargs)
        self.varName = varName

    @classmethod
    def default(cls, model):
        return ThermoelasticStress()

    def widget(self, fem, canvas):
        return ThermoelasticWidget(self, fem, canvas, "Ref. Temperature T0 (K)")


class DeformationPotential(DomainCondition):
    className = "DeformationPotential"

    def __init__(self, varNames=["n", "p"], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.varNames = varNames

    @classmethod
    def default(cls, model):
        return DeformationPotential()

    def widget(self, fem, canvas):
        return DeformationPotentialWidget(self, fem, canvas)


class ThermoelasticWidget(QtWidgets.QWidget):
    def __init__(self, cond, fem, canvas, title):
        super().__init__()
        self._cond = cond
        self.__initlayout(fem, canvas, title)

    def __initlayout(self, fem, canvas, title):
        self._selector = GeometrySelector(canvas, fem, self._cond.geometries)
        self._value = ScalarFunctionWidget(title, self._cond.values, valueChanged=self.__valueChanged)
        self._varName = QtWidgets.QLineEdit()
        self._varName.setText(self._cond.varName)
        self._varName.textChanged.connect(self.__textChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._selector)
        layout.addWidget(self._value)
        self.setLayout(layout)

    def __valueChanged(self, vector):
        self._cond.values = vector

    def __textChanged(self, txt):
        self._cond.varName = txt


class DeformationPotentialWidget(QtWidgets.QWidget):
    def __init__(self, cond, fem, canvas):
        super().__init__()
        self._cond = cond
        self.__initlayout(fem, canvas)

    def __initlayout(self, fem, canvas):
        self._selector = GeometrySelector(canvas, fem, self._cond.geometries)
        self._varName1 = QtWidgets.QLineEdit()
        self._varName1.setText(self._cond.varNames[0])
        self._varName1.textChanged.connect(self.__textChanged)
        self._varName2 = QtWidgets.QLineEdit()
        self._varName2.setText(self._cond.varNames[1])
        self._varName2.textChanged.connect(self.__textChanged)

        h1 = QtWidgets.QGridLayout()
        h1.addWidget(QtWidgets.QLabel("Electron Name"), 0, 0)
        h1.addWidget(QtWidgets.QLabel("Hole Name"), 1, 0)
        h1.addWidget(self._varName1, 0, 1)
        h1.addWidget(self._varName2, 1, 1)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._selector)
        layout.addLayout(h1)
        self.setLayout(layout)

    def __textChanged(self, txt):
        self._cond.varNames = [self._varName1.text(), self._varName2.text()]


class ElasticModel(FEMModel):
    className = "Elasticity"
    equationTypes = [ChristffelEquation]
    boundaryConditionTypes = [DirichletBoundary]
    domainConditionTypes = [ThermoelasticStress, DeformationPotential]
    initialConditionTypes = [InitialCondition]

    def __init__(self, nvar=3, *args, **kwargs):
        super().__init__(nvar, *args, **kwargs)


