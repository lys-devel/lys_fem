from .base import FEMObject, FEMObjectList
from .geometry import GeometrySelection


class DomainConditions(FEMObjectList):
    def get(self, cls):
        return [condition for condition in self if isinstance(condition, cls)]
    
    def have(self, cls):
        return len(self.get(cls)) > 0

    def append(self, condition):
        if condition.objName is None:
            names_used = [c.objName for c in self.parent.domainConditions]
            i = 1
            while condition.className + str(i) in names_used:
                i += 1
            condition.objName = condition.className + str(i)
        super().append(condition)


class ConditionBase(FEMObject):
    def __init__(self, geomType, values=None, objName=None, geometries=None):
        super().__init__(objName)
        self._geomType = geomType
        self._geom = GeometrySelection(self._geomType, geometries, parent=self)
        self._values = values

    @property
    def geometries(self):
        return self._geom

    @geometries.setter
    def geometries(self, value):
        self._geom = GeometrySelection(self._geomType, value, parent=self)

    @property
    def values(self):
        return self._values
    
    @values.setter
    def values(self, values):
        self._values = values

    def saveAsDictionary(self):
        return {"type": self.className, "objName": self.objName, "values": self.values, "geometries": self.geometries.saveAsDictionary()}

    @classmethod
    def loadFromDictionary(cls, d):
        domains = GeometrySelection.loadFromDictionary(d["geometries"])
        return cls(values = d["values"], geometries=domains, objName=d["objName"])

    @classmethod
    def default(self, model):
        raise NotImplementedError

    def widget(self, fem, canvas):
        raise NotImplementedError


class DomainCondition(ConditionBase):
    def __init__(self, *args, **kwargs):
        super().__init__("Domain", *args, **kwargs)


