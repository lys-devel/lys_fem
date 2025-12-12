from .base import FEMObjectDict, FEMObjectList, Coef
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
                coefs[d] = c.value.expression
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

    @property
    def dirichlet(self):
        from ..models.common import DirichletBoundary
        dirichlet = self.coef(DirichletBoundary)
        if dirichlet is None:
            return None
        if isinstance(list(dirichlet.values())[0], bool):
            size = 1
        else:
            size = len(list(dirichlet.values())[0])
        dirichlet = self.__dirichlet(dirichlet, size)
        return [dirichlet[i] for i in range(size)]

    def __dirichlet(self, coef, vdim):
        bdr_dir = [[] for _ in range(vdim)] 
        if coef is None:
            return bdr_dir
        for key, value in coef.items():
            for i, bdr in enumerate(bdr_dir):
                if hasattr(value, "__iter__"):
                    if value[i]:
                        bdr.append(key)
                elif value:
                    bdr.append(key)
        return list(bdr_dir)


class InitialConditions(ModelConditionBase):
    def append(self, condition):
        if condition.objName is None:
            names_used = [c.objName for c in self.parent.initialConditions]
            i = 1
            while condition.className + str(i) in names_used:
                i += 1
            condition.objName = condition.className + str(i)
        super().append(condition)


class ConditionBase(FEMObjectDict):
    """
    Base class for conditions in FEM.

    The condition (Domain, Boundary, and Initial conditions) in FEM is defined as values defined on geometries.

    As values, general string expression or sequence of string expression is acceptable.
    Even if the single condition requires several parameters (such as temperature and electric field), it is recommended to put all these values into single vector.
    """

    def __init__(self, geomType, value=None, objName=None, geometries=None):
        super().__init__(objName=objName)
        self._geom = GeometrySelection(geomType, geometries, parent=self)
        if value is not None:
            self["value"] = Coef(value)
     
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            raise AttributeError
        
    @property
    def geometries(self):
        return self._geom

    @geometries.setter
    def geometries(self, value):
        self._geom = GeometrySelection(self._geom.selectionType(), value, parent=self)

    def saveAsDictionary(self):
        values = {key: value.expression if isinstance(value, Coef) else value for key, value in self.items()}
        return {"type": self.className, "objName": self.objName, "values": values, "geometries": self.geometries.saveAsDictionary()}

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

    def widget(self, fem, canvas):
        from lys_fem.gui import ConditionWidget
        return ConditionWidget(self, fem, canvas)


class DomainCondition(ConditionBase):
    def __init__(self, *args, **kwargs):
        super().__init__("Domain", *args, **kwargs)


class BoundaryCondition(ConditionBase):
    def __init__(self, *args, **kwargs):
        super().__init__("Boundary", *args, **kwargs)


class InitialCondition(ConditionBase):
    className="Initial Condition"
    def __init__(self, value, geometries="all", *args, **kwargs):
        super().__init__("Domain", geometries=geometries, *args, **kwargs)
        self["value"] = Coef(value, shape=("V",), description="Initial value")

    @classmethod
    def default(cls, fem, model):
        return InitialCondition([0]*model.variableDimension)

