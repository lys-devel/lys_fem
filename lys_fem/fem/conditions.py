from .base import FEMObject, FEMObjectList
from .geometry import GeometrySelection


class ModelConditionBase(FEMObjectList):
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


class DomainConditions(ModelConditionBase):
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


class BoundaryConditions(ModelConditionBase):
    def get(self, cls):
        return [condition for condition in self if isinstance(condition, cls)]

    def have(self, cls):
        return len(self.get(cls)) > 0

    def append(self, condition):
        if condition.objName is None:
            names_used = [c.objName for c in self.parent.boundaryConditions]
            i = 1
            while condition.className + str(i) in names_used:
                i += 1
            condition.objName = condition.className + str(i)
        super().append(condition)


class InitialConditions(ModelConditionBase):
    def append(self, condition):
        if condition.objName is None:
            names_used = [c.objName for c in self.parent.initialConditions]
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
        geometries = GeometrySelection.loadFromDictionary(d["geometries"])
        return cls(values = d["values"], geometries=geometries, objName=d["objName"])

    @classmethod
    def default(self, model):
        raise NotImplementedError

    def widget(self, fem, canvas, title="Value"):
        from lys_fem.gui import ConditionWidget
        return ConditionWidget(self, fem, canvas, title)


class DomainCondition(ConditionBase):
    def __init__(self, *args, **kwargs):
        super().__init__("Domain", *args, **kwargs)


class BoundaryCondition(ConditionBase):
    def __init__(self, *args, **kwargs):
        super().__init__("Boundary", *args, **kwargs)


class InitialCondition(ConditionBase):
    className="Initial Condition"
    def __init__(self, *args, **kwargs):
        super().__init__("Domain", *args, **kwargs)

    @classmethod
    def default(cls, model):
        return InitialCondition([0]*model.variableDimension())

    def widget(self, fem, canvas, title="Initial Value"):
        super().widget(fem, canvas, title)
