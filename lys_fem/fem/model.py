from lys.Qt import QtWidgets

from .initialCondition import InitialCondition
from .boundaryConditions import DirichletBoundary, NeumannBoundary

models = {}


class FEMModel:
    def __init__(self, nvar, initialConditions=None, boundaryConditions=None, domainConditions=None):
        self._nvar = nvar
        if initialConditions is None:
            initialConditions = []
        if boundaryConditions is None:
            boundaryConditions = []
        if domainConditions is None:
            domainConditions = []
        self._init = initialConditions
        self._bdrs = boundaryConditions
        self._dcs = domainConditions

    def setVariableDimension(self, dim):
        self._nvar = dim
        for i in self._init:
            i.setDimension(dim)

    def variableDimension(self):
        return self._nvar

    @classmethod
    @property
    def domainConditionTypes(self):
        return []

    @property
    def domainConditions(self):
        return self._dcs

    @classmethod
    @property
    def boundaryConditionTypes(self):
        return [DirichletBoundary, NeumannBoundary]

    @property
    def boundaryConditions(self):
        return self._bdrs

    @classmethod
    @property
    def initialConditionTypes(self):
        return [InitialCondition]

    @property
    def initialConditions(self):
        return self._init

    def addDomainCondition(self, cond):
        i = 1
        while cond.name + str(i) in [c.objName for c in self._dcs]:
            i += 1
        name = cond.name + str(i)
        obj = cond(name)
        self._dcs.append(obj)
        return obj

    def addBoundaryCondition(self, bdr):
        i = 1
        while bdr.name + str(i) in [c.objName for c in self._bdrs]:
            i += 1
        name = bdr.name + str(i)
        if bdr == DirichletBoundary:
            obj = bdr(name, [False] * self._nvar)
        elif bdr == NeumannBoundary:
            obj = bdr(name, ["0"] * self._nvar)
        self._bdrs.append(obj)
        return obj

    def addInitialCondition(self, init):
        i = 1
        while init.name + str(i) in [c.objName for c in self._init]:
            i += 1
        name = init.name + str(i)
        obj = init(name, self._nvar)
        self._init.append(obj)
        return obj

    def saveAsDictionary(self):
        d = {"model": self.name}
        d["nvar"] = self._nvar
        d["init"] = [i.saveAsDictionary() for i in self.initialConditions]
        d["bdr"] = [b.saveAsDictionary() for b in self.boundaryConditions]
        d["domain"] = [d.saveAsDictionary() for d in self.domainConditions]
        return d

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

    @classmethod
    def loadFromDictionary(cls, d):
        nvar = d["nvar"]
        init, bdr, domain = cls._loadConditions(d)
        return cls(nvar, init, bdr, domain)


def loadModel(d):
    cls_list = set(sum(models.values(), []))
    cls_dict = {m.name: m for m in cls_list}
    model = cls_dict[d["model"]]
    del d["model"]
    return model.loadFromDictionary(d)
