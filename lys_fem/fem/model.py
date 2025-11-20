from .base import FEMObject
from .equations import Equations
from .conditions import DomainConditions, BoundaryConditions, InitialConditions, InitialCondition

models = {}


class FEMModel(FEMObject):
    equationTypes = []
    domainConditionTypes = []
    boundaryConditionTypes = []
    initialConditionTypes = [InitialCondition]
    discretizationTypes = ["BackwardEuler", "BDF2", "NewmarkBeta", "ForwardEuler"]

    def __init__(self, nvar, equations="auto", discretization="BackwardEuler", order=2, type="H1", initialConditions=None, boundaryConditions=None, domainConditions=None, objName=None, **kwargs):
        super().__init__(objName)
        self._nvar = nvar
        if equations == "auto":
            equations=[self.equationTypes[0](objName=self.equationTypes[0].className, **kwargs)]
        if initialConditions is None:
            initialConditions = []
        if boundaryConditions is None:
            boundaryConditions = []
        if domainConditions is None:
            domainConditions = []
        self._eqs = Equations(self, equations)
        self._init = InitialConditions(self, initialConditions)
        self._bdrs = BoundaryConditions(self, boundaryConditions)
        self._dcs = DomainConditions(self, domainConditions)
        self._disc = discretization
        self._order = order
        self._type = type

    def setVariableDimension(self, dim):
        self._nvar = dim
        for i in self._init:
            i.setDimension(dim)

    def variableDimension(self):
        return self._nvar

    @property
    def equations(self):
        return self._eqs

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
    def discretization(self):
        return self._disc

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order=value

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    def saveAsDictionary(self):
        d = {"model": self.className}
        d["nvar"] = self._nvar
        d["eqs"] = self.equations.saveAsDictionary()
        d["init"] = self.initialConditions.saveAsDictionary()
        d["bdr"] = self.boundaryConditions.saveAsDictionary()
        d["domain"] = self.domainConditions.saveAsDictionary()
        d["discretization"] = self._disc
        d["order"] = self._order
        d["type"] = self._type
        return d

    @classmethod
    def loadFromDictionary(cls, d):
        return cls(nvar=d["nvar"], **cls._loadConditions(d))

    @classmethod
    def _loadConditions(cls, d):
        eqs = Equations.loadFromDictionary(d["eqs"], cls.equationTypes)
        init = InitialConditions.loadFromDictionary(d["init"], cls.initialConditionTypes)
        bdr = BoundaryConditions.loadFromDictionary(d.get("bdr", []), cls.boundaryConditionTypes)
        domain = DomainConditions.loadFromDictionary(d.get("domain", []), cls.domainConditionTypes)
        return {"equations": eqs, "initialConditions": init, "boundaryConditions": bdr, "domainConditions": domain, "discretization": d.get("discretization", "BackwardEuler"), "order": d.get("order", 2), "type": d.get("type", "H1")}

    def widget(self, fem, canvas):
        from ..gui import FEMModelWidget
        return FEMModelWidget(self)
    
    
class FEMFixedModel(FEMModel):
    @classmethod
    def loadFromDictionary(cls, d):
        return cls(**cls._loadConditions(d))

    def widget(self, fem, canvas):
        from ..gui import FEMFixedModelWidget
        return FEMFixedModelWidget(self)


def loadModel(d):
    cls_list = set(sum(models.values(), []))
    cls_dict = {m.className: m for m in cls_list}
    model = cls_dict[d["model"]]
    del d["model"]
    return model.loadFromDictionary(d)
