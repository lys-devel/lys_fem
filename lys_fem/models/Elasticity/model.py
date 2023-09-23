from lys.Qt import QtWidgets
from lys_fem import FEMModel

from ..Common import InitialCondition, DirichletBoundary, NeumannBoundary


class ElasticModel(FEMModel):
    def __init__(self, nvar=3, initialConditions=None, boundaryConditions=None, domainConditions=None):
        self._nvar = nvar
        if initialConditions is None:
            initialConditions = [InitialCondition("Default", self._nvar, domains="all")]
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
    def name(cls):
        return "Elasticity"

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
            obj = bdr(name)
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

    def widget(self, fem, canvas):
        return _ElasticityWidget(self)

    def saveAsDictionary(self):
        d = super().saveAsDictionary()
        d["nvar"] = self._nvar
        d["init"] = [i.saveAsDictionary() for i in self.initialConditions]
        d["bdr"] = [b.saveAsDictionary() for b in self.boundaryConditions]
        d["domain"] = [d.saveAsDictionary() for d in self.domainConditions]
        return d

    @classmethod
    def loadFromDictionary(cls, d):
        nvar = d["nvar"]
        init = [cls._loadCondition(dic, cls.initialConditionTypes) for dic in d["init"]]
        bdr = [cls._loadCondition(dic, cls.boundaryConditionTypes) for dic in d.get("bdr", [])]
        domain = [cls._loadCondition(dic, cls.domainConditionTypes) for dic in d.get("domain", [])]
        return ElasticModel(nvar, init, bdr, domain)

    @classmethod
    def _loadCondition(cls, d, types):
        cls_dict = {t.name: t for t in types}
        c = cls_dict[d["type"]]
        del d["type"]
        return c.loadFromDictionary(d)


class _ElasticityWidget(QtWidgets.QWidget):
    def __init__(self, model):
        super().__init__()
        self._model = model
        self.__initlayout()

    def __initlayout(self):
        self._dim = QtWidgets.QSpinBox()
        self._dim.setRange(1, 3)
        self._dim.setValue(self._model.variableDimension())
        self._dim.valueChanged.connect(self.__changeDim)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Variable dimension"))
        layout.addWidget(self._dim)
        self.setLayout(layout)

    def __changeDim(self, value):
        self._model.setVariableDimension(self._dim.value())
