import numpy as np

from lys_fem import util
from .base import FEMObject, FEMObjectList
from .geometry import GeometrySelection


class ModelConditionBase(FEMObjectList):
    def get(self, cls):
        return [condition for condition in self if isinstance(condition, cls)]
    
    def have(self, cls):
        return len(self.get(cls)) > 0

    def coef(self, cls):
        if not self.have(cls):
            return None
        coefs = {}
        for c in self.get(cls):
            for d in c.geometries:
                coefs[d] = c.values
        return coefs

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
    def append(self, condition):
        if condition.objName is None:
            names_used = [c.objName for c in self.parent.domainConditions]
            i = 1
            while condition.className + str(i) in names_used:
                i += 1
            condition.objName = condition.className + str(i)
        super().append(condition)


class BoundaryConditions(ModelConditionBase):
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
    """
    Base class for conditions in FEM.

    The condition (Domain, Boundary, and Initial conditions) in FEM is defined as values defined on geometries.

    As values, general string expression or sequence of string expression is acceptable.
    Even if the single condition requires several parameters (such as temperature and electric field), it is recommended to put all these values into single vector.
    """

    def __init__(self, geomType, values=None, objName=None, geometries=None, **kwargs):
        super().__init__(objName)
        self._geomType = geomType
        self._geom = GeometrySelection(self._geomType, geometries, parent=self)
        self._values = {key: self.__parseValues(v) for key, v in kwargs.items()}
        if values is not None:
            self._values["values"] = self.__parseValues(values)

    def __parseValues(self, value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return [self.__parseValues(v) for v in value]
        if isinstance(value, (bool, int, float, complex)):
            return value
        else:
            return str(value)
        
    def eval(self, key, dic):
        return util.eval(self._values.get(key, None), dic, name=key)

    def __getattr__(self, key):
        return self._values.get(key, None)
    
    def __setattr__(self, key, value):
        if "_values" in self.__dict__:
            if key in self._values:
                self._values[key] = value
                return
        super().__setattr__(key, value)

    @property
    def geometries(self):
        return self._geom

    @geometries.setter
    def geometries(self, value):
        self._geom = GeometrySelection(self._geomType, value, parent=self)

    def saveAsDictionary(self):
        return {"type": self.className, "objName": self.objName, "values": self._values, "geometries": self.geometries.saveAsDictionary()}

    @classmethod
    def loadFromDictionary(cls, d):
        geometries = GeometrySelection.loadFromDictionary(d["geometries"])
        values = d.get("values", {})
        if not isinstance(values, dict): # For backward compability
            values = {"values": values}
        return cls(geometries=geometries, objName=d["objName"], **values)

    @classmethod
    def default(cls, fem, model):
        return cls()

    def widget(self, fem, canvas, title="Value", shape=None):
        from lys_fem.gui import ConditionWidget
        return ConditionWidget(self, fem, canvas, title=title, shape=shape)


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
    def default(cls, fem, model):
        return InitialCondition([0]*model.variableDimension())

    def widget(self, fem, canvas, title="Initial Value"):
        return super().widget(fem, canvas, title, shape=(self.model.variableDimension(),))

