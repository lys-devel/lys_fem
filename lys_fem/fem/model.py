
class FEMModel:
    def saveAsDictionary(self):
        model = None
        for key, m in models.items():
            if isinstance(self, m):
                model = key
                continue
        return {"model": model}

    @staticmethod
    def loadFromDictionary(d):
        model = models[d["model"]]
        del d["model"]
        return model.loadFromDictionary(d)


class ElasticModelGenerator(FEMModel):
    def __init__(self, nvar=3, init=None):
        self._nvar = nvar
        if init is None:
            init = [InitialCondition("Default", self._nvar, domains="all")]
        self._init = init

    def setVariableDimension(self, dim):
        self._nvar = dim

    def variableDimension(self):
        return self._nvar

    @property
    def initialConditions(self):
        return self._init

    def addInitialCondition(self, init=None):
        if init is None:
            i = 1
            while "Initial value" + str(i) in [c.name for c in self._init]:
                i += 1
            name = "Initial value" + str(i)
            init = InitialCondition(name, self._nvar)
            self._init.append(init)
        return init

    def saveAsDictionary(self):
        d = super().saveAsDictionary()
        d["nvar"] = self._nvar
        d["init"] = [i.saveAsDictionary() for i in self._init]
        return d

    @staticmethod
    def loadFromDictionary(d):
        nvar = d["nvar"]
        init = [InitialCondition.loadFromDictionary(dic) for dic in d["init"]]
        return ElasticModelGenerator(nvar, init)


class InitialCondition:
    def __init__(self, name, nvar, value=None, domains=None):
        self._name = name
        if value is None:
            value = ["0"] * nvar
        self._value = value
        if domains is None:
            domains = []
        self._domains = domains

    def setDimension(self, dim):
        if len(self._value) > dim:
            self._value = self._value[:2]
        while len(self._value) < dim:
            self._value.append("0")

    @property
    def name(self):
        return self._name

    @property
    def domains(self):
        return self._domains

    @domains.setter
    def domains(self, value):
        self._domains = value

    @property
    def values(self):
        return self._value

    @values.setter
    def values(self, value):
        self._value = value

    def saveAsDictionary(self):
        return {"name": self._name, "domains": self._domains, "values": self._value}

    @staticmethod
    def loadFromDictionary(d):
        values = d["values"]
        return InitialCondition(d["name"], len(values), value=values, domains=d["domains"])


models = {"Elasticity": ElasticModelGenerator}
