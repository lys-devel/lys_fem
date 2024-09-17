from .base import FEMObject, FEMObjectList, FEMCoefficient, strToExpr, exprToStr
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
            typ = c.geometries.geometryType
            unit = c.unit
        return FEMCoefficient(coefs, typ, xscale=self.fem.geometries.scale, vars=self.fem.parameters.getSolved())

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

    As values, general sympy expression or sequence of expression is acceptable.
    Even if the single condition requires several parameters (such as temperature and electric field), it is recommended to put all these values into single vector.
    In addition, values are loaded from calculation result if values is instance of CalculatedResult.
    """
    unit = "1"

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
        return {"type": self.className, "objName": self.objName, "values": exprToStr(self.values), "geometries": self.geometries.saveAsDictionary()}

    @classmethod
    def loadFromDictionary(cls, d):
        geometries = GeometrySelection.loadFromDictionary(d["geometries"])
        values = strToExpr(d["values"])
        return cls(values = values, geometries=geometries, objName=d["objName"])

    @classmethod
    def default(cls, model):
        raise NotImplementedError

    def widget(self, fem, canvas, title="Value", computed=False, shape=None):
        from lys_fem.gui import ConditionWidget
        return ConditionWidget(self, fem, canvas, title=title, computed=computed, shape=shape)


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
        return super().widget(fem, canvas, title, computed=True, shape=(self.model.variableDimension(),))


class CalculatedResult:
    def __init__(self, path="", expression="", index=-1):
        self._path = path
        self._expression = expression
        self._index = index

    @property
    def solution(self):
        from .solution import FEMSolution
        return FEMSolution(self._path)
    
    @property
    def path(self):
        return self._path
    
    @property
    def expression(self):
        return self._expression
    
    @property
    def index(self):
        return self._index
    
    def saveAsDictionary(self):
        return {"path": self._path, "expression": self._expression, "index": self._index}
    
    @classmethod
    def loadFromDictionary(cls, d):
        return CalculatedResult(**d)