from .base import FEMObject, FEMObjectList
from .equations import Equations
from .conditions import DomainConditions, BoundaryConditions, InitialConditions

models = {}


class FEMModel(FEMObject):
    def __init__(self, nvar, equations, initialConditions=None, boundaryConditions=None, domainConditions=None):
        self._nvar = nvar
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

    def saveAsDictionary(self):
        d = {"model": self.name}
        d["nvar"] = self._nvar
        d["init"] = [i.saveAsDictionary() for i in self.initialConditions]
        d["bdr"] = [b.saveAsDictionary() for b in self.boundaryConditions]
        d["domain"] = [d.saveAsDictionary() for d in self.domainConditions]
        return d

    @classmethod
    def loadFromDictionary(cls, d):
        nvar = d["nvar"]
        init, bdr, domain = cls._loadConditions(d)
        return cls(nvar=nvar, initialConditions=init, boundaryConditions=bdr, domainConditions=domain)

    @classmethod
    def _loadConditions(cls, d):
        init = [cls._loadCondition(dic, cls.initialConditionTypes) for dic in d["init"]]
        bdr = [cls._loadCondition(dic, cls.boundaryConditionTypes) for dic in d.get("bdr", [])]
        domain = [cls._loadCondition(dic, cls.domainConditionTypes) for dic in d.get("domain", [])]
        return init, bdr, domain

    @classmethod
    def _loadCondition(cls, d, types):
        cls_dict = {t.name: t for t in types}
        c = cls_dict[d["type"]]
        del d["type"]
        return c.loadFromDictionary(d)

    def widget(self, fem, canvas):
        from ..gui import FEMModelWidget
        return FEMModelWidget(self)
    
    @classmethod
    @property
    def equationTypes(self):
        return []

    @classmethod
    @property
    def domainConditionTypes(self):
        return []

    @classmethod
    @property
    def boundaryConditionTypes(self):
        return []

    @classmethod
    @property
    def initialConditionTypes(self):
        return []
    
    
class FEMFixedModel(FEMModel):
    @classmethod
    def loadFromDictionary(cls, d):
        init, bdr, domain = cls._loadConditions(d)
        return cls(initialConditions=init, boundaryConditions=bdr, domainConditions=domain)

    def widget(self, fem, canvas):
        from ..gui import FEMFixedModelWidget
        return FEMFixedModelWidget(self)


def loadModel(d):
    cls_list = set(sum(models.values(), []))
    cls_dict = {m.name: m for m in cls_list}
    model = cls_dict[d["model"]]
    del d["model"]
    return model.loadFromDictionary(d)
