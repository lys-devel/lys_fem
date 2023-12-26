from .base import FEMObject, FEMObjectList
from .geometry import GeometrySelection


class Equations(FEMObjectList):
    pass


class Equation(FEMObject):
    def __init__(self, name, varName, varDim=None, geometries="all", geometryType="Domain"):
        self._varName = varName
        self._varDim = varDim
        self._name = name
        if isinstance(geometries, GeometrySelection):
            self._geometries = geometries
        else:
            self._geometries = GeometrySelection(geometryType, geometries)
        self._geometries.setParent(self)

    @classmethod
    @property
    def name(cls):
        raise NotImplementedError

    @property
    def objName(self):
        return self._name

    @property
    def variableName(self):
        return self._varName

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
        self._geometries = value
