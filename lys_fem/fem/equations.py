import numpy as np
from .base import FEMObject, FEMObjectList
from .geometry import GeometrySelection


class Equations(FEMObjectList):
    def append(self, equation):
        if equation.objName is None:
            names_used = [c.objName for c in self.parent.equations]
            i = 1
            while equation.className + str(i) in names_used:
                i += 1
            equation.objName = equation.className + str(i)
        varNames_used = [c.variableName for c in self.parent.equations]
        if equation.variableName in varNames_used:
            i = 1
            while equation.variableName + str(i) in varNames_used:
                i += 1
            equation.variableName = equation.variableName + str(i)
        super().append(equation)

    def saveAsDictionary(self):
        return [item.saveAsDictionary() for item in self]

    @classmethod
    def loadFromDictionary(self, dic, types):
        cls_dict = {t.className: t for t in types}
        result = []
        for d in dic:
            c = cls_dict[d["type"]]
            del d["type"]
            result.append(c.loadFromDictionary(d))
        return result


class Equation(FEMObject):
    isScalar = False
    def __init__(self, varName, varDim=None, geometries="all", geometryType="Domain", objName=None, **kwargs):
        super().__init__(objName)
        self._varName = varName
        self._varDim = varDim
        self._geometries = GeometrySelection(geometryType, geometries, parent=self)
        self._values = {key: self.__parseValues(v) for key, v in kwargs.items()}

    def __parseValues(self, value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return [self.__parseValues(v) for v in value]
        if isinstance(value, (bool, int, float, complex)):
            return value
        else:
            return str(value)

    def __getattr__(self, key):
        return self._values.get(key, None)
    
    def set(self, name, value):
        self._values[name] = value

    @property
    def variableName(self):
        return self._varName
    
    @variableName.setter
    def variableName(self, value):
        self._varName = value

    @property
    def variableDimension(self):
        if self._varDim is None:
            return self.parent.parent.variableDimension()
        else:
            return self._varDim

    @property
    def geometries(self):
        return self._geometries

    @geometries.setter
    def geometries(self, value):
        self._geometries = GeometrySelection(value.geometryType, value, parent=self)

    def saveAsDictionary(self):
        return {"type": self.className, "objName": self.objName, "varName": self._varName, "dim": self._varDim, "geometries": self.geometries.saveAsDictionary(), "values": self._values}

    @classmethod
    def loadFromDictionary(cls, d):
        geometries = GeometrySelection.loadFromDictionary(d["geometries"])
        return cls(d["varName"], varDim = d["dim"], geometries=geometries, objName=d["objName"], **d.get("values", {}))

    def widget(self, fem, canvas):
        from lys_fem.gui import EquationWidget
        return EquationWidget(self, fem, canvas)
