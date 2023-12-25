from .base import FEMObject, FEMObjectList
from .geometry import GeometrySelection


class Equations(FEMObjectList):
    pass


class Equation(FEMObject):
    def __init__(self, name, varName, varDim, geometries="all", geometryType="Domain"):
        self._varName = varName
        self._varDim = varDim
        self._name = name
        if isinstance(geometries, GeometrySelection):
            self._domains = geometries
        else:
            self._domains = GeometrySelection(geometryType, geometries)
        self._domains.setParent(self)

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
        return self._varDim

    @property
    def geometries(self):
        return self._domains

    @geometries.setter
    def geometries(self, value):
        self._domains = value

    @property
    def domains(self):
        return self._domains

    @domains.setter
    def domains(self, value):
        self._domains = value

    @property
    def boundaries(self):
        return self._domains

    @boundaries.setter
    def boundaries(self, value):
        self._domains = value
