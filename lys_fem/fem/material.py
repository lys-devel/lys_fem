import itertools
import numpy as np


class Material(list):
    def __init__(self, name, domains=None, params=None):
        self._name = name
        if domains is None:
            domains = []
        self._domains = domains
        if params is None:
            params = []
        super().__init__(params)

    def __getitem__(self, name_param):
        for p in self:
            if p.name == name_param:
                return p

    @property
    def name(self):
        return self._name

    @property
    def domains(self):
        return self._domains

    @domains.setter
    def domains(self, value):
        if value == "all":
            self._domains = "all"
        else:
            self._domains = list(value)

    def saveAsDictionary(self):
        return {"name": self._name, "domains": self.domains, "params": [p.saveAsDictionary() for p in self]}

    @staticmethod
    def loadFromDictionary(d):
        params = [FEMParameter.loadFromDictionary(p) for p in d["params"]]
        return Material(d["name"], d["domains"], params)


class FEMParameter:
    def saveAsDictionary(self):
        d = vars(self)
        d["paramsName"] = self.name
        return d

    @staticmethod
    def loadFromDictionary(d):
        cls_list = set(sum(materialParameters.values(), []))
        cls_dict = {value.name: value for value in cls_list}

        d = dict(d)
        cls = cls_dict[d["paramsName"]]
        del d["paramsName"]
        return cls(**d)


class ElasticParameters(FEMParameter):
    def __init__(self, rho, C, type="lame"):
        self.rho = rho
        self.C = C
        self.type = type

    @classmethod
    @property
    def name(cls):
        return "Elasticity"

    def getParameters(self):
        return {"rho": self.rho, "C": self._constructC()}

    def _constructC(self):
        if self.type == "lame":
            return [[self._lame(i, j) for j in range(6)] for i in range(6)]

    def _lame(self, i, j):
        res = "0"
        if i < 3 and j < 3:
            res += "+" + self.C[0]
        if i == j:
            if i < 3:
                res += "+ 2 *" + self.C[1]
            else:
                res += "+" + self.C[1]
        return res


materialParameters = {"Acoustics": [ElasticParameters]}
