import numpy as np
from lys_fem import util
from . import time
from .base import FEMObject
from .conditions import DomainConditions, BoundaryConditions, InitialConditions, InitialCondition
from .geometry import GeometrySelection

models = {}


class FEMModel(FEMObject):
    domainConditionTypes = []
    boundaryConditionTypes = []
    initialConditionTypes = [InitialCondition]
    discretizationTypes = ["BackwardEuler", "BDF2", "NewmarkBeta", "ForwardEuler"]

    def __init__(self, nvar, initialConditions=None, boundaryConditions=None, domainConditions=None, objName=None, equation=None, **kwargs):
        super().__init__(objName)
        if initialConditions is None:
            initialConditions = []
        if boundaryConditions is None:
            boundaryConditions = []
        if domainConditions is None:
            domainConditions = []
        if equation is not None:
            self._eq = equation
        else:
            self._eq= Equation(varDim=nvar, **kwargs)
        self._eq.setParent(self)
        self._init = InitialConditions(initialConditions, parent=self)
        self._bdrs = BoundaryConditions(boundaryConditions, parent=self)
        self._dcs = DomainConditions(domainConditions, parent=self)

    def __getattr__(self, key):
        return getattr(self._eq, key)
    
    def functionSpaces(self):
        return [self._eq.functionSpaces(self.boundaryConditions.dirichlet)]

    def initialValues(self, params):
        x = util.eval(self.__initialValue(), params, geom="domain")
        if x is None:
            raise RuntimeError("Invalid initial value for " + str(self.variableName))
        if not x.valid:
            x = util.NGSFunction([0] * self.size)
        return [x]

    def initialVelocities(self, params):
        v = util.eval([0]*self.size, geom="domain")
        return [v]

    def __initialValue(self):
        init = self.initialConditions.coef(self.initialConditionTypes[0])
        for type in self.initialConditionTypes[1:]:
            init.update(self.initialConditions.coef(type).value)
        return init

    def discretize(self, dti):
        d = {}
        for v in self.functionSpaces():
            trial = util.trial(v)
            if self.discretization == "ForwardEuler":
                d.update(time.ForwardEuler.generateWeakforms(trial, dti))
            elif self.discretization == "BackwardEuler":
                d.update(time.BackwardEuler.generateWeakforms(trial, dti))
            elif self.discretization == "BDF2":
                d.update(time.BDF2.generateWeakforms(trial, dti))
            elif self.discretization == "NewmarkBeta":
                d.update(time.NewmarkBeta.generateWeakforms(trial, dti))
            else:
                raise RuntimeError("Unknown discretization: "+self.discretization)
        return d

    def updater(self, dti):
        d = self.discretize(dti)
        res = {}
        for v in self.functionSpaces():
            trial = util.trial(v)
            if trial.t in d:
                res[trial.t] = d[trial.t]
            if trial.tt in d:
                res[trial.tt] = d[trial.tt]
        return res

    @property
    def geometries(self):
        return self._eq.geometries
            
    @property
    def domainConditions(self):
        return self._dcs

    @property
    def boundaryConditions(self):
        return self._bdrs

    @property
    def initialConditions(self):
        return self._init
    
    @property
    def equation(self):
        return self._eq

    def saveAsDictionary(self):
        d = {"model": self.className}
        d["nvar"] = self.variableDimension
        d["eqs"] = self._eq.saveAsDictionary()
        d["init"] = self.initialConditions.saveAsDictionary()
        d["bdr"] = self.boundaryConditions.saveAsDictionary()
        d["domain"] = self.domainConditions.saveAsDictionary()
        d["discretization"] = self._disc
        return d

    @classmethod
    def loadFromDictionary(cls, d):
        return cls(nvar=d["nvar"], **cls._loadConditions(d))

    @classmethod
    def _loadConditions(cls, d):
        eqs = Equation.loadFromDictionary(d["eqs"])
        init = InitialConditions.loadFromDictionary(d["init"], cls.initialConditionTypes)
        bdr = BoundaryConditions.loadFromDictionary(d.get("bdr", []), cls.boundaryConditionTypes)
        domain = DomainConditions.loadFromDictionary(d.get("domain", []), cls.domainConditionTypes)
        return {"equation": eqs, "initialConditions": init, "boundaryConditions": bdr, "domainConditions": domain, "discretization": d.get("discretization", "BackwardEuler")}

    def __str__(self):
        res = "\t"+self.className+": discretization = " + self.discretization + "\n"
        for i, v in enumerate(self.functionSpaces()):
            res += "\t\tVariable " + str(1+i) + ": " + str(v)  + "\n"
        return res

    def widget(self, fem, canvas):
        from ..gui import FEMModelWidget
        return FEMModelWidget(self, fem, canvas)


class Equation(FEMObject):
    def __init__(self, varName, varDim=None, geometries="all", geometryType="Domain", discretization="BackwardEuler", order=2, type="H1", isScalar=False, valType="x", **kwargs):
        super().__init__()
        self._varName = varName
        self._varDim = varDim
        self._order = order
        self._type = type
        self._isScalar = isScalar
        self._valType = valType
        self._disc = discretization
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
    
    def functionSpaces(self, dirichlet):
        kwargs = {"fetype": self._type, "isScalar": self._isScalar, "order": self._order, "size": self._varDim, "valtype": self._valType, "dirichlet": dirichlet}
        if self._geometries is not None:
            if self._geometries.selectionType() == "Selected":
                kwargs["geometries"] = list(self._geometries)
        return util.FunctionSpace(self._varName, **kwargs)

    def set(self, name, value):
        self._values[name] = value

    @property
    def size(self):
        return self._varDim

    @property
    def order(self):
        return self._order
    
    @order.setter
    def order(self, value):
        self._order = value

    @property
    def fetype(self):
        return self._type

    @property
    def variableName(self):
        return self._varName
    
    @property
    def discretization(self):
        return self._disc

    @discretization.setter
    def discretization(self, value):
        self._disc = value
    
    @variableName.setter
    def variableName(self, value):
        self._varName = value

    @property
    def variableDimension(self):
        return self._varDim
    
    @variableDimension.setter
    def variableDimension(self, value):
        self._varDim = value

    @property
    def geometries(self):
        return self._geometries

    @geometries.setter
    def geometries(self, value):
        self._geometries = GeometrySelection(value.geometryType, value, parent=self)

    def saveAsDictionary(self):
        return {"type": self._type, "objName": self.objName, "varName": self._varName, "dim": self._varDim, "geometries": self.geometries.saveAsDictionary(), "values": self._values, "order": self._order, "valType": self._valType, "isScalar": self._isScalar, "discretization": self._disc}

    @classmethod
    def loadFromDictionary(cls, d):
        geometries = GeometrySelection.loadFromDictionary(d["geometries"])
        return cls(d["varName"], varDim = d["dim"], geometries=geometries, type=d["type"], order=d["order"], valType=d["valType"], isScalar=d["isScalar"], discretization=d.get("discretization", "BackwardEuler"), **d.get("values", {}))

    
class FEMFixedModel(FEMModel):
    @classmethod
    def loadFromDictionary(cls, d):
        return cls(**cls._loadConditions(d))

    def widget(self, fem, canvas):
        from ..gui import FEMModelWidget
        return FEMModelWidget(self, fem, canvas, varDim=False)


def loadModel(d):
    cls_list = set(sum(models.values(), []))
    cls_dict = {m.className: m for m in cls_list}
    model = cls_dict[d["model"]]
    del d["model"]
    return model.loadFromDictionary(d)
