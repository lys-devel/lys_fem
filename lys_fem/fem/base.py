import weakref
import numpy as np
import sympy as sp

class FEMObject:
    def __init__(self, objName=None):
        self._objName = objName

    @property
    def objName(self):
        return self._objName
    
    @objName.setter
    def objName(self, value):
        self._objName = value

    @classmethod
    @property
    def className(cls):
        raise NotImplementedError

    @property
    def fem(self):
        from .FEM import FEMProject
        obj = self
        while not isinstance(obj, FEMProject):
            obj = obj.parent
        return obj
    
    @property
    def model(self):
        from .model import FEMModel
        obj = self
        while not isinstance(obj, FEMModel):
            obj = obj.parent
        return obj

    @property
    def parent(self):
        return self._parent()

    def setParent(self, parent):
        self._parent = weakref.ref(parent)


class FEMObjectList(list, FEMObject):
    def __init__(self, parent, items=[]):
        super().__init__(items)
        self.setParent(parent)
        for item in items:
            if isinstance(item, FEMObject):
                item.setParent(self)

    def append(self, item):
        super().append(item)
        item.setParent(self)


class FEMCoefficient:
    """
    Sympy-based coefficient object for FEM.
    value is string expression, list of string expression, or dict of them.
    """
    def __init__(self, value=0, geomType="Domain"):
        super().__init__()
        if not isinstance(value, dict):
            geomType = "Const"
        self._type = geomType
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def geometryType(self):
        return self._type

    @property
    def default(self):
        if not isinstance(self._value, dict):
            return None
        if "default" not in self._value:
            return None
        return self._value["default"]
    
    def __getitem__(self, index):
        if isinstance(self._value, dict):
            return FEMCoefficient({key: value[index] for key, value in self._value.items()}, geomType=self._type)
        return FEMCoefficient(self.value[index], geomType=self._type)
      