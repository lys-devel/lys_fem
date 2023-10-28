from .geometry import GeometrySelection

materialParameters = {}


class Material(list):
    def __init__(self, name, domains=None, params=None):
        self._name = name
        if isinstance(domains, GeometrySelection):
            self._domains = domains
        else:
            self._domains = GeometrySelection("Domain", domains)
        if params is None:
            params = []
        super().__init__(params)

    def __getitem__(self, i):
        if isinstance(i, str):
            for p in self:
                if p.name == i:
                    return p
        else:
            return super().__getitem__(i)

    @property
    def name(self):
        return self._name

    @property
    def domains(self):
        return self._domains

    @domains.setter
    def domains(self, value):
        self._domains = value

    def saveAsDictionary(self):
        return {"name": self._name, "domains": self.domains.saveAsDictionary(), "params": [p.saveAsDictionary() for p in self]}

    @staticmethod
    def loadFromDictionary(d):
        params = [FEMParameter.loadFromDictionary(p) for p in d["params"]]
        return Material(d["name"], GeometrySelection.loadFromDictionary(d["domains"]), params)


class FEMParameter:
    def __init__(self, name):
        self._name = name

    def saveAsDictionary(self):
        d = vars(self)
        d["paramsName"] = self.name
        return d

    def getParameters(self):
        return vars(self)

    @staticmethod
    def loadFromDictionary(d):
        cls_list = set(sum(materialParameters.values(), []))
        cls_dict = {value.name: value for value in cls_list}

        d = dict(d)
        cls = cls_dict[d["paramsName"]]
        del d["paramsName"]
        return cls(**d)
